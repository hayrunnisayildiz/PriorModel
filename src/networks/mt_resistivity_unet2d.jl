"""
    MTResistivityUNet2DLayers

Lux.jl 2-D U-Net for MT-only resistivity prior generation with CBAM2D attention.

# Tensor layout (column-major, `Float32`)
- Input:  `(n_stations, n_periods, C_in[, B])` — log10 apparent resistivity + phase
  (see [`MTMeshParams.MT_DATA_LAYOUT`](@ref))
- Output: `(nz, nx[, B])` — log10 resistivity on the uniform [`MeshParams`](@ref) grid

MT station–period data are projected to a coarse spatial bottleneck, then decoded
with transposed convolutions to full mesh resolution. Output is scaled with
``\\sigma`` to ``[\\mathrm{logres\\_min}, \\mathrm{logres\\_max}]``.
"""
module MTResistivityUNet2DLayers

using Lux
using Random
using Statistics: mean

if !isdefined(@__MODULE__, :MTMeshParams)
    include(joinpath(@__DIR__, "..", "synthetic", "MeshParams.jl"))
end
using .MTMeshParams: MeshParams, DEFAULT_MESH, n_periods, n_components, validate_mesh_params

export Conv2dGNBlock, DoubleConv2DBlock, EncoderStage2D, DecoderStage2D, BottleneckStage2D
export MTInputProjector, MTResistivityUNet2D, replace_nan, add_batch_dim, generate_logres_prior
export ChannelAttention2D, SpatialAttention2D, CBAM2D

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Pick a GroupNorm group count that divides `channels`."""
function _group_count(channels::Int)::Int
    for g in (8, 4, 2, 1)
        channels % g == 0 && return g
    end
    return 1
end

"""Spatial sizes after `n_down` stride-2 encoder stages (minimum 1)."""
function _downsample_spatial(nz::Int, nx::Int, n_down::Int=2)::Tuple{Int,Int}
    nz_out = max(1, nz)
    nx_out = max(1, nx)
    for _ in 1:n_down
        nz_out = max(1, nz_out ÷ 2)
        nx_out = max(1, nx_out ÷ 2)
    end
    return nz_out, nx_out
end

"""
    replace_nan(x; fill=0)

Zygote-safe replacement of non-finite MT samples before the network.
"""
function replace_nan(x::AbstractArray{T}; fill=zero(T)) where {T}
    return ifelse.(isfinite.(x), x, T(fill))
end

"""Promote `(H, W, C)` to `(H, W, C, 1)`; leave 4-D unchanged."""
function add_batch_dim(x::AbstractArray{T,3}) where {T}
    return reshape(x, size(x)..., 1)
end
add_batch_dim(x::AbstractArray{T,4}) where {T} = x

function _ensure_4d(x::AbstractArray)
    nd = ndims(x)
    nd == 4 && return x, false
    nd == 3 && return add_batch_dim(x), true
    throw(ArgumentError("expected (S,P,C) or (S,P,C,B), got ndims=$nd size=$(size(x))"))
end

"""Center-crop the leading `(H, W)` axes of a 4-D tensor."""
function _crop_hw(x::AbstractArray, target_h::Int, target_w::Int)
    H, W = size(x, 1), size(x, 2)
    (H == target_h && W == target_w) && return x
    H < target_h && error("cannot crop: tensor H=$H < target=$target_h")
    W < target_w && error("cannot crop: tensor W=$W < target=$target_w")
    oh = (H - target_h) ÷ 2
    ow = (W - target_w) ÷ 2
    return x[oh+1:oh+target_h, ow+1:ow+target_w, :, :]
end

"""Center-pad `(H, W)` to `(target_h, target_w)` with zeros."""
function _pad_hw(x::AbstractArray{T,4}, target_h::Int, target_w::Int) where {T}
    H, W = size(x, 1), size(x, 2)
    (H == target_h && W == target_w) && return x
    H > target_h && return _crop_hw(x, target_h, target_w)
    W > target_w && return _crop_hw(x, target_h, target_w)
    out = zeros(T, target_h, target_w, size(x, 3), size(x, 4))
    oh = (target_h - H) ÷ 2
    ow = (target_w - W) ÷ 2
    out[oh+1:oh+H, ow+1:ow+W, :, :] .= x
    return out
end

function _scale_log_resistivity(logits::AbstractArray{T},
                                logres_min::Float32, logres_max::Float32) where {T}
    lo = T(logres_min)
    hi = T(logres_max)
    return lo .+ (hi - lo) .* sigmoid.(logits)
end

# ─────────────────────────────────────────────────────────────────────────────
# CBAM 2-D
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChannelAttention2D(channels; reduction=4)

Channel attention on `(H, W, C, B)` feature maps.
"""
struct ChannelAttention2D{M} <: Lux.AbstractLuxContainerLayer{(:mlp,)}
    channels::Int
    reduction::Int
    mlp::M
end

function ChannelAttention2D(channels::Int; reduction::Int=4)
    channels >= 1 || throw(ArgumentError("channels must be ≥ 1, got $channels"))
    reduction >= 1 || throw(ArgumentError("reduction must be ≥ 1, got $reduction"))
    hidden = max(1, channels ÷ reduction)
    mlp = Chain(
        Dense(channels, hidden, relu),
        Dense(hidden, channels),
    )
    return ChannelAttention2D{typeof(mlp)}(channels, reduction, mlp)
end

@inline function _cvec2d(x::AbstractArray{T,4}) where {T}
    return reshape(x, size(x, 3), size(x, 4))
end

@inline function _cmap2d(v::AbstractArray{T,2}) where {T}
    return reshape(v, 1, 1, size(v, 1), size(v, 2))
end

function (m::ChannelAttention2D)(x::AbstractArray{T,4}, ps, st) where {T}
    avg = mean(x; dims=(1, 2))
    mx = maximum(x; dims=(1, 2))
    w_avg, st_mlp = m.mlp(_cvec2d(avg), ps.mlp, st.mlp)
    w_max, st_mlp = m.mlp(_cvec2d(mx), ps.mlp, st_mlp)
    w = _cmap2d(sigmoid.(w_avg .+ w_max))
    return x .* w, (; mlp=st_mlp)
end

"""
    SpatialAttention2D(; kernel=3)

Spatial attention on `(H, W, C, B)` feature maps.
"""
struct SpatialAttention2D{C} <: Lux.AbstractLuxContainerLayer{(:conv,)}
    kernel::Int
    conv::C
end

function SpatialAttention2D(; kernel::Int=3)
    isodd(kernel) && kernel >= 1 ||
        throw(ArgumentError("spatial kernel must be a positive odd integer, got $kernel"))
    conv = Conv((kernel, kernel), 2 => 1, sigmoid; pad=SamePad())
    return SpatialAttention2D{typeof(conv)}(kernel, conv)
end

function (m::SpatialAttention2D)(x::AbstractArray{T,4}, ps, st) where {T}
    avg = mean(x; dims=3)
    mx = maximum(x; dims=3)
    stats = cat(avg, mx; dims=3)
    att, st_conv = m.conv(stats, ps.conv, st.conv)
    return x .* att, (; conv=st_conv)
end

"""
    CBAM2D(channels; reduction=4, spatial_kernel=3)

Sequential channel then spatial attention on `(H, W, C, B)`.
"""
struct CBAM2D{CA,SA} <: Lux.AbstractLuxContainerLayer{(:channel, :spatial)}
    channel::CA
    spatial::SA
end

function CBAM2D(channels::Int; reduction::Int=4, spatial_kernel::Int=3)
    ca = ChannelAttention2D(channels; reduction=reduction)
    sa = SpatialAttention2D(; kernel=spatial_kernel)
    return CBAM2D{typeof(ca),typeof(sa)}(ca, sa)
end

function (m::CBAM2D)(x::AbstractArray{T,4}, ps, st) where {T}
    y, st_c = m.channel(x, ps.channel, st.channel)
    z, st_s = m.spatial(y, ps.spatial, st.spatial)
    return z, (; channel=st_c, spatial=st_s)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2-D convolution blocks
# ─────────────────────────────────────────────────────────────────────────────

"""
    Conv2dGNBlock(in_ch, out_ch; kernel=3, stride=1)

`Conv2D → GroupNorm → ELU` on `(H, W, C, B)`.
"""
struct Conv2dGNBlock{C,G} <: Lux.AbstractLuxContainerLayer{(:conv, :gn)}
    conv::C
    gn::G
end

function Conv2dGNBlock(in_ch::Int, out_ch::Int; kernel::Int=3, stride::Int=1)
    isodd(kernel) || throw(ArgumentError("kernel must be odd, got $kernel"))
    conv = Conv((kernel, kernel), in_ch => out_ch; pad=SamePad(), stride=stride)
    gn = GroupNorm(out_ch, _group_count(out_ch))
    return Conv2dGNBlock{typeof(conv),typeof(gn)}(conv, gn)
end

function (m::Conv2dGNBlock)(x::AbstractArray{T,4}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    h, st_g = m.gn(h, ps.gn, st.gn)
    return elu.(h), (; conv=st_c, gn=st_g)
end

"""
    DoubleConv2DBlock(in_ch, out_ch; kernel=3)

Two consecutive `Conv2D → GroupNorm → ELU` blocks (U-Net style).
"""
struct DoubleConv2DBlock{B1,B2} <: Lux.AbstractLuxContainerLayer{(:block1, :block2)}
    block1::B1
    block2::B2
end

function DoubleConv2DBlock(in_ch::Int, out_ch::Int; kernel::Int=3)
    block1 = Conv2dGNBlock(in_ch, out_ch; kernel=kernel)
    block2 = Conv2dGNBlock(out_ch, out_ch; kernel=kernel)
    return DoubleConv2DBlock{typeof(block1),typeof(block2)}(block1, block2)
end

function (m::DoubleConv2DBlock)(x::AbstractArray{T,4}, ps, st) where {T}
    h, st_b1 = m.block1(x, ps.block1, st.block1)
    h, st_b2 = m.block2(h, ps.block2, st.block2)
    return h, (; block1=st_b1, block2=st_b2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Encoder / decoder stages
# ─────────────────────────────────────────────────────────────────────────────

"""
    EncoderStage2D(in_ch, out_ch, next_ch; kernel=3)

`DoubleConv2D → CBAM2D` (skip tensor) then stride-2 `Conv2D` downsampling.
"""
struct EncoderStage2D{DC,CB,D} <: Lux.AbstractLuxContainerLayer{(:conv, :cbam, :down)}
    out_ch::Int
    conv::DC
    cbam::CB
    down::D
end

function EncoderStage2D(in_ch::Int, out_ch::Int, next_ch::Int; kernel::Int=3,
                        reduction::Int=4, spatial_kernel::Int=3)
    conv = DoubleConv2DBlock(in_ch, out_ch; kernel=kernel)
    cbam = CBAM2D(out_ch; reduction=reduction, spatial_kernel=spatial_kernel)
    down = Conv((kernel, kernel), out_ch => next_ch; pad=SamePad(), stride=(2, 2))
    return EncoderStage2D{typeof(conv),typeof(cbam),typeof(down)}(
        out_ch, conv, cbam, down,
    )
end

function (m::EncoderStage2D)(x::AbstractArray{T,4}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    skip, st_cb = m.cbam(h, ps.cbam, st.cbam)
    h, st_d = m.down(skip, ps.down, st.down)
    return h, skip, (; conv=st_c, cbam=st_cb, down=st_d)
end

"""
    DecoderStage2D(up_ch, skip_ch, out_ch; kernel=3)

`ConvTranspose2D` upsampling, optional skip `cat(..., dims=3)`, then `DoubleConv2D`.
When `skip_ch == 0`, skip fusion is omitted (projection-only decoder path).
"""
struct DecoderStage2D{U,DC} <: Lux.AbstractLuxContainerLayer{(:upsample_layer, :conv)}
    up_ch::Int
    skip_ch::Int
    upsample_layer::U
    conv::DC
end

function DecoderStage2D(up_ch::Int, skip_ch::Int, out_ch::Int; kernel::Int=3)
    fuse_ch = up_ch + skip_ch
    up = ConvTranspose((kernel, kernel), up_ch => up_ch; pad=SamePad(), stride=(2, 2))
    conv = DoubleConv2DBlock(fuse_ch, out_ch; kernel=kernel)
    return DecoderStage2D{typeof(up),typeof(conv)}(up_ch, skip_ch, up, conv)
end

function (m::DecoderStage2D)(x::AbstractArray{T,4}, skip::Union{AbstractArray{T,4},Nothing},
                             ps, st) where {T}
    h, st_up = m.upsample_layer(x, ps.upsample_layer, st.upsample_layer)
    if m.skip_ch > 0
        skip === nothing && error("DecoderStage2D expects a skip tensor, got nothing")
        H, W = size(h)[1:2]
        skip = _crop_hw(skip, H, W)
        h = cat(h, skip; dims=3)
    end
    h, st_c = m.conv(h, ps.conv, st.conv)
    return h, (; upsample_layer=st_up, conv=st_c)
end

"""
    BottleneckStage2D(channels; kernel=3)

`DoubleConv2D → CBAM2D` at the lowest spatial resolution.
"""
struct BottleneckStage2D{DC,CB} <: Lux.AbstractLuxContainerLayer{(:conv, :cbam)}
    conv::DC
    cbam::CB
end

function BottleneckStage2D(channels::Int; kernel::Int=3,
                           reduction::Int=4, spatial_kernel::Int=3)
    conv = DoubleConv2DBlock(channels, channels; kernel=kernel)
    cbam = CBAM2D(channels; reduction=reduction, spatial_kernel=spatial_kernel)
    return BottleneckStage2D{typeof(conv),typeof(cbam)}(conv, cbam)
end

function (m::BottleneckStage2D)(x::AbstractArray{T,4}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    h, st_cb = m.cbam(h, ps.cbam, st.cbam)
    return h, (; conv=st_c, cbam=st_cb)
end

# ─────────────────────────────────────────────────────────────────────────────
# MT input projection
# ─────────────────────────────────────────────────────────────────────────────

"""
    MTInputProjector(n_stations, n_periods, out_h, out_w, out_ch; in_channels=2)

Project `(n_stations, n_periods, C_in)` MT data to a coarse spatial tensor
`(out_h, out_w, out_ch)` via a Dense/reshape bottleneck.

MT curves are not natively `(nz, nx)`, so the network learns a spatial bottleneck
before the U-Net decoder path.
"""
struct MTInputProjector{M} <: Lux.AbstractLuxContainerLayer{(:mlp,)}
    n_stations::Int
    n_periods::Int
    in_channels::Int
    out_h::Int
    out_w::Int
    out_ch::Int
    mlp::M
end

function MTInputProjector(n_stations::Int, n_periods::Int,
                          out_h::Int, out_w::Int, out_ch::Int;
                          in_channels::Int=2, hidden_mult::Int=4)
    n_stations >= 1 || throw(ArgumentError("n_stations must be ≥ 1"))
    n_periods >= 1 || throw(ArgumentError("n_periods must be ≥ 1"))
    in_channels >= 1 || throw(ArgumentError("in_channels must be ≥ 1"))
    out_h >= 1 || throw(ArgumentError("out_h must be ≥ 1"))
    out_w >= 1 || throw(ArgumentError("out_w must be ≥ 1"))
    out_ch >= 1 || throw(ArgumentError("out_ch must be ≥ 1"))

    in_dim = n_stations * n_periods * in_channels
    out_dim = out_h * out_w * out_ch
    hidden = max(out_ch * hidden_mult, 64)
    mlp = Chain(
        Dense(in_dim, hidden, relu),
        Dense(hidden, out_dim),
    )
    return MTInputProjector{typeof(mlp)}(
        n_stations, n_periods, in_channels, out_h, out_w, out_ch, mlp,
    )
end

function (m::MTInputProjector)(x::AbstractArray{T,4}, ps, st) where {T}
    S, P = size(x, 1), size(x, 2)
    (S == m.n_stations && P == m.n_periods) || throw(DimensionMismatch(
        "expected MT shape ($(m.n_stations), $(m.n_periods), $(m.in_channels), B), got $(size(x))"))
    size(x, 3) == m.in_channels || throw(DimensionMismatch(
        "expected C_in=$(m.in_channels), got size(x)=$(size(x))"))

    B = size(x, 4)
    flat = reshape(x, S * P * m.in_channels, B)
    h, st_mlp = m.mlp(flat, ps.mlp, st.mlp)
    h = reshape(h, m.out_h, m.out_w, m.out_ch, B)
    return h, (; mlp=st_mlp)
end

# ─────────────────────────────────────────────────────────────────────────────
# MT 2-D resistivity U-Net
# ─────────────────────────────────────────────────────────────────────────────

"""
    MTResistivityUNet2D(;
        in_channels=2,
        base_channels=16,
        reduction=4,
        spatial_kernel=3,
        logres_min=0.0f0,
        logres_max=4.0f0,
        mesh=DEFAULT_MESH,
    )

# Architecture
1. **Input projection** — `(n_stations, n_periods, C_in)` MT tensor → coarse
   `(nz_b, nx_b, 4·base)` spatial bottleneck (replaces volumetric encoder input).
2. **Bottleneck** — `DoubleConv2D → CBAM2D` at `4·base` channels.
3. **Decoder (×2)** — `ConvTranspose2D` upsample, `DoubleConv2D`.
4. **Head** — `1×1` conv → sigmoid-scaled ``\\log_{10}\\rho`` in
   ``[\\mathrm{logres\\_min}, \\mathrm{logres\\_max}]``.

# Forward
`(model)(x, ps, st) -> (logres, st′)`

- `x`: `(n_stations, n_periods, C_in, B)` or 3-D without batch
- `logres`: `(nz, nx, B)` or `(nz, nx)` — log10 resistivity (Ω·m)
"""
struct MTResistivityUNet2D{P,B,D1,D2,H} <: Lux.AbstractLuxContainerLayer{
    (:projector, :bottleneck, :dec1, :dec2, :head)
}
    mesh::MeshParams
    in_channels::Int
    base_channels::Int
    logres_min::Float32
    logres_max::Float32
    nz_bottleneck::Int
    nx_bottleneck::Int
    projector::P
    bottleneck::B
    dec1::D1
    dec2::D2
    head::H
end

function MTResistivityUNet2D(;
                             in_channels::Int=2,
                             base_channels::Int=16,
                             reduction::Int=4,
                             spatial_kernel::Int=3,
                             logres_min::Float32=0.0f0,
                             logres_max::Float32=4.0f0,
                             mesh::MeshParams=DEFAULT_MESH)
    mesh = validate_mesh_params(mesh)
    in_channels >= 1 || throw(ArgumentError("in_channels must be ≥ 1"))
    base_channels >= 4 || throw(ArgumentError("base_channels must be ≥ 4"))
    logres_max > logres_min || throw(ArgumentError("logres_max must exceed logres_min"))

    c1 = base_channels
    c2 = 2 * base_channels
    c4 = 4 * base_channels

    nz_b, nx_b = _downsample_spatial(mesh.nz, mesh.nx, 2)

    projector = MTInputProjector(
        mesh.n_stations, n_periods(mesh), nz_b, nx_b, c4;
        in_channels=in_channels,
    )
    bottleneck = BottleneckStage2D(c4;
                                   reduction=reduction, spatial_kernel=spatial_kernel)
    dec1 = DecoderStage2D(c4, 0, c2)
    dec2 = DecoderStage2D(c2, 0, c1)
    head = Conv((1, 1), c1 => 1)

    return MTResistivityUNet2D{typeof(projector),typeof(bottleneck),
                               typeof(dec1),typeof(dec2),typeof(head)}(
        mesh, in_channels, base_channels, logres_min, logres_max, nz_b, nx_b,
        projector, bottleneck, dec1, dec2, head,
    )
end

function (m::MTResistivityUNet2D)(x::AbstractArray, ps, st)
    x4, squeezed = _ensure_4d(x)
    size(x4, 3) == m.in_channels || throw(DimensionMismatch(
        "expected C_in=$(m.in_channels), got size(x)=$(size(x4))"))

    x_in = replace_nan(x4)

    # ── MT → coarse spatial bottleneck ───────────────────────────────────────
    h, st_p = m.projector(x_in, ps.projector, st.projector)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    bot, st_bot = m.bottleneck(h, ps.bottleneck, st.bottleneck)

    # ── Decoder (upsample; no encoder skips on MT path) ──────────────────────
    h_dec, st_d1 = m.dec1(bot, nothing, ps.dec1, st.dec1)
    h_dec, st_d2 = m.dec2(h_dec, nothing, ps.dec2, st.dec2)

    # ── Pad/crop to exact mesh resolution ────────────────────────────────────
    h_up = _pad_hw(h_dec, m.mesh.nz, m.mesh.nx)

    # ── Log-resistivity head ─────────────────────────────────────────────────
    logits, st_head = m.head(h_up, ps.head, st.head)
    logres = _scale_log_resistivity(logits[:, :, 1, :], m.logres_min, m.logres_max)

    st_new = (;
        projector=st_p, bottleneck=st_bot,
        dec1=st_d1, dec2=st_d2, head=st_head,
    )

    if squeezed
        logres = dropdims(logres; dims=3)
    end
    return logres, st_new
end

function Base.show(io::IO, m::MTResistivityUNet2D)
    c = m.base_channels
    print(io, "MTResistivityUNet2D(mesh=$(m.mesh.nz)×$(m.mesh.nx)",
          ", C_in=", m.in_channels,
          ", base=", c, "→", 2c, "→", 4c,
          ", log10ρ∈[", m.logres_min, ",", m.logres_max, "])")
end

"""
    generate_logres_prior(model, x, ps, st) -> (logres, st′)

Convenience wrapper returning `(nz, nx[, B])` log10-resistivity.
"""
function generate_logres_prior(model::MTResistivityUNet2D, x::AbstractArray, ps, st)
    logres, st_new = model(x, ps, st)
    return logres, st_new
end

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (run when this file is executed directly)
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_smoke_test(; kwargs...) -> Bool

Forward pass on random MT data; returns `true` when all checks pass.
"""
function run_smoke_test(;
                        mesh::MeshParams=DEFAULT_MESH,
                        in_channels::Int=2,
                        batch_size::Int=2,
                        base_channels::Int=16,
                        rng::AbstractRNG=Random.default_rng())::Bool
    mesh = validate_mesh_params(mesh)
    model = MTResistivityUNet2D(; in_channels, base_channels, mesh)
    ps, st = Lux.setup(rng, model)

    S = mesh.n_stations
    P = n_periods(mesh)
    x = randn(Float32, S, P, in_channels, batch_size)
    logres, st = model(x, ps, st)

    size(logres) == (mesh.nz, mesh.nx, batch_size) ||
        error("expected output (nz,nx,B)=($(mesh.nz),$(mesh.nx),$batch_size), got $(size(logres))")
    eltype(logres) == Float32 || error("expected Float32 output")
    all(isfinite, logres) || error("non-finite values in output")
    all(logres .>= model.logres_min .- 1.0f-6) ||
        error("logres below logres_min")
    all(logres .<= model.logres_max .+ 1.0f-6) ||
        error("logres above logres_max")

    x3 = randn(Float32, S, P, in_channels)
    logres1, _ = model(x3, ps, st)
    size(logres1) == (mesh.nz, mesh.nx) ||
        error("expected unbatched output ($(mesh.nz),$(mesh.nx)), got $(size(logres1))")

    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert run_smoke_test()
    println("MTResistivityUNet2D smoke test passed.")
end

end # module MTResistivityUNet2DLayers
