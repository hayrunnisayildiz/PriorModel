"""
    PriorUNet3DLayers

Lux.jl 3-D U-Net prior generator with CBAM3D attention (`UnifiedPriorUNet3D`).

# Tensor layout (column-major, `Float32`)
- Input:  `(P_x, P_y, P_z, C_in, B)` — EM/IP/resistivity patch from [`PatchLoader`](@ref)
  (`resistivity`, `vlf_resistivity`, `slingram_real`, `ip_chargeability`, + validity mask)
- Output: `(P_x, P_y, P_z, 3, B)` with
  - channel 1: ``m_0`` — initial resistivity prior, ohm·m, in ``[\\rho_{min}, \\rho_{max}]``
  - channel 2: ``m_{min}`` — lower bound, ``m_{min} \\le m_0``
  - channel 3: ``m_{max}`` — upper bound, ``m_0 \\le m_{max}``

The head predicts ``\\log_{10}\\rho`` then converts with ``10^{\\cdot}`` so the
network trains in log space while the saved prior is in ohm·m.

Explicit `ps` / `st` throughout; compatible with Zygote.jl, Optimisers.jl, and
ComponentArrays.jl via `Lux.setup`.
"""
module PriorUNet3DLayers

using Lux
using Random

if !isdefined(@__MODULE__, :CBAMLayers)
    include(joinpath(@__DIR__, "..", "neural_prior", "CBAM3D.jl"))
end
using .CBAMLayers: CBAM3D

export Conv3dGNBlock, DoubleConv3DBlock, EncoderStage3D, DecoderStage3D, BottleneckStage3D
export UnifiedPriorUNet3D, replace_nan, add_batch_dim, generate_prior
export OUT_M0, OUT_MMIN, OUT_MMAX

const OUT_M0::Int = 1
const OUT_MMIN::Int = 2
const OUT_MMAX::Int = 3

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

"""
    replace_nan(x; fill=0)

Zygote-safe replacement of non-finite fusion samples before convolution.
"""
function replace_nan(x::AbstractArray{T}; fill=zero(T)) where {T}
    return ifelse.(isfinite.(x), x, T(fill))
end

"""Promote `(X, Y, Z, C)` to `(X, Y, Z, C, 1)`; leave 5-D unchanged."""
function add_batch_dim(x::AbstractArray{T,4}) where {T}
    return reshape(x, size(x)..., 1)
end
add_batch_dim(x::AbstractArray{T,5}) where {T} = x

function _ensure_5d(x::AbstractArray)
    nd = ndims(x)
    nd == 5 && return x, false
    nd == 4 && return add_batch_dim(x), true
    throw(ArgumentError("expected (X,Y,Z,C) or (X,Y,Z,C,B), got ndims=$nd size=$(size(x))"))
end

"""Center-crop the leading `(X, Y, Z)` axes of a 5-D tensor."""
function _crop_xyz(x::AbstractArray, target_x::Int, target_y::Int, target_z::Int)
    X, Y, Z = size(x, 1), size(x, 2), size(x, 3)
    (X == target_x && Y == target_y && Z == target_z) && return x
    X < target_x && error("cannot crop: tensor X=$X < target=$target_x")
    Y < target_y && error("cannot crop: tensor Y=$Y < target=$target_y")
    Z < target_z && error("cannot crop: tensor Z=$Z < target=$target_z")
    ox = (X - target_x) ÷ 2
    oy = (Y - target_y) ÷ 2
    oz = (Z - target_z) ÷ 2
    return x[ox+1:ox+target_x, oy+1:oy+target_y, oz+1:oz+target_z, :, :]
end

function _scale_log_resistivity(logits::AbstractArray{T},
                                log_rho_min::Float32, log_rho_max::Float32) where {T}
    lo = T(log_rho_min)
    hi = T(log_rho_max)
    return lo .+ (hi - lo) .* sigmoid.(logits)
end

function _clip(x::AbstractArray{T}, lo::T, hi::T) where {T}
    return ifelse.(x .< lo, lo, ifelse.(x .> hi, hi, x))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3-D convolution blocks
# ─────────────────────────────────────────────────────────────────────────────

"""
    Conv3dGNBlock(in_ch, out_ch; kernel=3, stride=1)

`Conv3D → GroupNorm → ELU` on `(X, Y, Z, C, B)`.
"""
struct Conv3dGNBlock{C,G} <: Lux.AbstractLuxContainerLayer{(:conv, :gn)}
    conv::C
    gn::G
end

function Conv3dGNBlock(in_ch::Int, out_ch::Int; kernel::Int=3, stride::Int=1)
    isodd(kernel) || throw(ArgumentError("kernel must be odd, got $kernel"))
    conv = Conv((kernel, kernel, kernel), in_ch => out_ch; pad=SamePad(), stride=stride)
    gn = GroupNorm(out_ch, _group_count(out_ch))
    return Conv3dGNBlock{typeof(conv),typeof(gn)}(conv, gn)
end

function (m::Conv3dGNBlock)(x::AbstractArray{T,5}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    h, st_g = m.gn(h, ps.gn, st.gn)
    return elu.(h), (; conv=st_c, gn=st_g)
end

"""
    DoubleConv3DBlock(in_ch, out_ch; kernel=3)

Two consecutive `Conv3D → GroupNorm → ELU` blocks (U-Net style).
"""
struct DoubleConv3DBlock{B1,B2} <: Lux.AbstractLuxContainerLayer{(:block1, :block2)}
    block1::B1
    block2::B2
end

function DoubleConv3DBlock(in_ch::Int, out_ch::Int; kernel::Int=3)
    block1 = Conv3dGNBlock(in_ch, out_ch; kernel=kernel)
    block2 = Conv3dGNBlock(out_ch, out_ch; kernel=kernel)
    return DoubleConv3DBlock{typeof(block1),typeof(block2)}(block1, block2)
end

function (m::DoubleConv3DBlock)(x::AbstractArray{T,5}, ps, st) where {T}
    h, st_b1 = m.block1(x, ps.block1, st.block1)
    h, st_b2 = m.block2(h, ps.block2, st.block2)
    return h, (; block1=st_b1, block2=st_b2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Encoder / decoder stages
# ─────────────────────────────────────────────────────────────────────────────

"""
    EncoderStage3D(in_ch, out_ch, next_ch; kernel=3)

`DoubleConv3D → CBAM3D` (skip tensor) then stride-2 `Conv3D` downsampling.
"""
struct EncoderStage3D{DC,CB,D} <: Lux.AbstractLuxContainerLayer{(:conv, :cbam, :down)}
    out_ch::Int
    conv::DC
    cbam::CB
    down::D
end

function EncoderStage3D(in_ch::Int, out_ch::Int, next_ch::Int; kernel::Int=3,
                        reduction::Int=4, spatial_kernel::Int=3)
    conv = DoubleConv3DBlock(in_ch, out_ch; kernel=kernel)
    cbam = CBAM3D(out_ch; reduction=reduction, spatial_kernel=spatial_kernel)
    down = Conv((kernel, kernel, kernel), out_ch => next_ch; pad=SamePad(), stride=(2, 2, 2))
    return EncoderStage3D{typeof(conv),typeof(cbam),typeof(down)}(
        out_ch, conv, cbam, down,
    )
end

function (m::EncoderStage3D)(x::AbstractArray{T,5}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    skip, st_cb = m.cbam(h, ps.cbam, st.cbam)
    h, st_d = m.down(skip, ps.down, st.down)
    return h, skip, (; conv=st_c, cbam=st_cb, down=st_d)
end

"""
    DecoderStage3D(up_ch, skip_ch, out_ch; kernel=3)

`ConvTranspose3D` upsampling, skip `cat(..., dims=4)`, then `DoubleConv3D`.
"""
struct DecoderStage3D{U,DC} <: Lux.AbstractLuxContainerLayer{(:upsample_layer, :conv)}
    up_ch::Int
    skip_ch::Int
    upsample_layer::U
    conv::DC
end

function DecoderStage3D(up_ch::Int, skip_ch::Int, out_ch::Int; kernel::Int=3)
    fuse_ch = up_ch + skip_ch
    up = ConvTranspose((kernel, kernel, kernel), up_ch => up_ch;
                       pad=SamePad(), stride=(2, 2, 2))
    conv = DoubleConv3DBlock(fuse_ch, out_ch; kernel=kernel)
    return DecoderStage3D{typeof(up),typeof(conv)}(up_ch, skip_ch, up, conv)
end

function (m::DecoderStage3D)(x::AbstractArray{T,5}, skip::AbstractArray{T,5},
                             ps, st) where {T}
    h, st_up = m.upsample_layer(x, ps.upsample_layer, st.upsample_layer)
    X, Y, Z = size(h)[1:3]
    skip = _crop_xyz(skip, X, Y, Z)
    h = cat(h, skip; dims=4)
    h, st_c = m.conv(h, ps.conv, st.conv)
    return h, (; upsample_layer=st_up, conv=st_c)
end

"""
    BottleneckStage3D(channels; kernel=3)

`DoubleConv3D → CBAM3D` at the lowest spatial resolution.
"""
struct BottleneckStage3D{DC,CB} <: Lux.AbstractLuxContainerLayer{(:conv, :cbam)}
    conv::DC
    cbam::CB
end

function BottleneckStage3D(channels::Int; kernel::Int=3,
                           reduction::Int=4, spatial_kernel::Int=3)
    conv = DoubleConv3DBlock(channels, channels; kernel=kernel)
    cbam = CBAM3D(channels; reduction=reduction, spatial_kernel=spatial_kernel)
    return BottleneckStage3D{typeof(conv),typeof(cbam)}(conv, cbam)
end

function (m::BottleneckStage3D)(x::AbstractArray{T,5}, ps, st) where {T}
    h, st_c = m.conv(x, ps.conv, st.conv)
    h, st_cb = m.cbam(h, ps.cbam, st.cbam)
    return h, (; conv=st_c, cbam=st_cb)
end

# ─────────────────────────────────────────────────────────────────────────────
# Unified 3-D U-Net + CBAM prior generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    UnifiedPriorUNet3D(;
        in_channels=5,
        base_channels=16,
        reduction=4,
        spatial_kernel=3,
        log_rho_min=-1.0f0,
        log_rho_max=5.0f0,
        delta_max=1.0f0,
        logit_scale=0.10f0,
    )

# Architecture
1. **Encoder (×2)** — `DoubleConv3D → CBAM3D` skip, stride-2 downsample;
   channels `base → 2·base → 4·base`.
2. **Bottleneck** — `DoubleConv3D → CBAM3D` at `4·base` channels.
3. **Decoder (×2)** — `ConvTranspose3D` upsample, skip `cat(..., dims=4)`,
   `DoubleConv3D`.
4. **Head** — `1×1×1` conv → ``\\log_{10} m_0, \\delta_-, \\delta_+`` then
   ``m = 10^{\\cdot}`` with geological resistivity bounds
   ``[10^{\\mathrm{log\\_rho\\_min}}, 10^{\\mathrm{log\\_rho\\_max}}]`` ohm·m
   (default 0.1 – 10⁵ Ω·m).

# Forward
`(model)(x, ps, st) -> (volume, st′)`

- `x`: `(P_x, P_y, P_z, C_in, B)` or 4-D without batch
- `volume`: `(P_x, P_y, P_z, 3, B)` — channels 1–3 = ``m_0, m_{min}, m_{max}`` (ohm·m)
"""
struct UnifiedPriorUNet3D{E1,E2,B,D1,D2,H} <: Lux.AbstractLuxContainerLayer{
    (:enc1, :enc2, :bottleneck, :dec1, :dec2, :head)
}
    in_channels::Int
    base_channels::Int
    log_rho_min::Float32
    log_rho_max::Float32
    delta_max::Float32
    logit_scale::Float32
    enc1::E1
    enc2::E2
    bottleneck::B
    dec1::D1
    dec2::D2
    head::H
end

function UnifiedPriorUNet3D(;
                            in_channels::Int=5,
                            base_channels::Int=16,
                            reduction::Int=4,
                            spatial_kernel::Int=3,
                            log_rho_min::Float32=-1.0f0,
                            log_rho_max::Float32=5.0f0,
                            delta_max::Float32=1.0f0,
                            logit_scale::Float32=0.10f0)
    in_channels >= 1 || throw(ArgumentError("in_channels must be ≥ 1"))
    base_channels >= 4 || throw(ArgumentError("base_channels must be ≥ 4"))
    log_rho_max > log_rho_min || throw(ArgumentError("log_rho_max must exceed log_rho_min"))
    delta_max > 0 || throw(ArgumentError("delta_max must be positive"))
    logit_scale > 0 || throw(ArgumentError("logit_scale must be positive"))

    c1 = base_channels
    c2 = 2 * base_channels
    c4 = 4 * base_channels

    enc1 = EncoderStage3D(in_channels, c1, c2;
                          reduction=reduction, spatial_kernel=spatial_kernel)
    enc2 = EncoderStage3D(c2, c2, c4;
                          reduction=reduction, spatial_kernel=spatial_kernel)

    bottleneck = BottleneckStage3D(c4;
                                   reduction=reduction, spatial_kernel=spatial_kernel)

    dec1 = DecoderStage3D(c4, c2, c2)
    dec2 = DecoderStage3D(c2, c1, c1)
    head = Conv((1, 1, 1), c1 => 3)

    return UnifiedPriorUNet3D{typeof(enc1),typeof(enc2),typeof(bottleneck),
                              typeof(dec1),typeof(dec2),typeof(head)}(
        in_channels, base_channels, log_rho_min, log_rho_max, delta_max, logit_scale,
        enc1, enc2, bottleneck, dec1, dec2, head,
    )
end

function (m::UnifiedPriorUNet3D)(x::AbstractArray, ps, st)
    x5, squeezed = _ensure_5d(x)
    size(x5, 4) == m.in_channels || throw(DimensionMismatch(
        "expected C_in=$(m.in_channels), got size(x)=$(size(x5))"))

    x_in = replace_nan(x5)

    # ── Encoder ──────────────────────────────────────────────────────────────
    h, skip1, st_e1 = m.enc1(x_in, ps.enc1, st.enc1)
    h, skip0, st_e2 = m.enc2(h, ps.enc2, st.enc2)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    bot, st_bot = m.bottleneck(h, ps.bottleneck, st.bottleneck)

    # ── Decoder (upsample → fuse skip → conv) ────────────────────────────────
    h_dec, st_d1 = m.dec1(bot, skip0, ps.dec1, st.dec1)
    h_dec, st_d2 = m.dec2(h_dec, skip1, ps.dec2, st.dec2)

    # ── Log-resistivity prior head (ohm·m after 10^) ─────────────────────────
    logits, st_head = m.head(h_dec, ps.head, st.head)

    T = eltype(logits)
    s = T(m.logit_scale)
    lo = T(m.log_rho_min)
    hi = T(m.log_rho_max)

    log_m0 = _scale_log_resistivity(logits[:, :, :, 1:1, :] .* s,
                                    m.log_rho_min, m.log_rho_max)
    δ = T(m.delta_max) .* sigmoid.(logits[:, :, :, 2:3, :] .* s)
    δ_lo = δ[:, :, :, 1:1, :]
    δ_hi = δ[:, :, :, 2:2, :]
    log_min = _clip(log_m0 .- δ_lo, lo, hi)
    log_max = _clip(log_m0 .+ δ_hi, lo, hi)
    m0 = exp10.(log_m0)
    m_min = exp10.(log_min)
    m_max = exp10.(log_max)
    volume = cat(m0, m_min, m_max; dims=4)

    st_new = (;
        enc1=st_e1, enc2=st_e2, bottleneck=st_bot,
        dec1=st_d1, dec2=st_d2, head=st_head,
    )

    if squeezed
        volume = dropdims(volume; dims=5)
    end
    return volume, st_new
end

function Base.show(io::IO, m::UnifiedPriorUNet3D)
    c = m.base_channels
    print(io, "UnifiedPriorUNet3D(C_in=", m.in_channels,
          ", base=", c, "→", 2c, "→", 4c,
          ", ρ∈[", exp10(m.log_rho_min), ",", exp10(m.log_rho_max), "] Ω·m)")
end

"""
    generate_prior(model, x, ps, st) -> (volume, st′)

Convenience wrapper.  Returns `(P_x, P_y, P_z, 3[, B])` with
``m_0, m_{min}, m_{max}`` in ohm·m.
"""
function generate_prior(model::UnifiedPriorUNet3D, x::AbstractArray, ps, st)
    volume, st_new = model(x, ps, st)
    return volume, st_new
end

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (run when this file is executed directly)
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_smoke_test(; kwargs...) -> Bool

Forward pass on a random patch batch; returns `true` when all checks pass.
"""
function run_smoke_test(;
                        px::Int=32, py::Int=32, pz::Int=16,
                        in_channels::Int=5, batch_size::Int=2,
                        base_channels::Int=16,
                        rng::AbstractRNG=Random.default_rng())::Bool
    model = UnifiedPriorUNet3D(; in_channels, base_channels)
    ps, st = Lux.setup(rng, model)
    x = randn(Float32, px, py, pz, in_channels, batch_size)
    volume, st = model(x, ps, st)

    size(volume) == (px, py, pz, 3, batch_size) ||
        error("expected output (px,py,pz,3,B), got $(size(volume))")
    eltype(volume) == Float32 || error("expected Float32 output")
    all(isfinite, volume) || error("non-finite values in output")

    m0 = volume[:, :, :, 1, :]
    m_min = volume[:, :, :, 2, :]
    m_max = volume[:, :, :, 3, :]
    all(m_min .<= m0 .+ 1.0f-5) || error("m_min > m0 violation")
    all(m0 .<= m_max .+ 1.0f-5) || error("m0 > m_max violation")
    rho_lo = exp10(model.log_rho_min)
    rho_hi = exp10(model.log_rho_max)
    all(m0 .>= rho_lo * 0.999f0) || error("m0 below ρ_min")
    all(m0 .<= rho_hi * 1.001f0) || error("m0 above ρ_max")

    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert run_smoke_test()
    println("UnifiedPriorUNet3D smoke test passed.")
end

end # module PriorUNet3DLayers
