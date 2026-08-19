"""
    PriorNet3D

Lightweight Lux.jl 3-D convolutional network that maps an EM/IP/resistivity
fusion tensor onto a smart resistivity prior.

# Tensors (column-major, `Float32`)
- Input:  `(X, Y, Z, C_in=4, B)` — fusion volume; `NaN32` holes are zero-filled
- `m0`:   `(X, Y, Z, 1, B)` — initial resistivity ``m_0`` in ohm·m
- bounds: `(X, Y, Z, 2, B)` — ``m_{min}`` (channel 1) and ``m_{max}`` (channel 2)

Fusion channel 1 is borehole `LUO_R` (ohm·m). Compare to `m0` in log10 space
(see `Losses.extract_resistivity_target`).

Lux `ps` / `st` are explicit throughout. Compatible with Zygote.jl and
Optimization.jl.
"""
module PriorNet3D

using Lux

if !isdefined(@__MODULE__, :CBAMLayers)
    include(joinpath(@__DIR__, "CBAM3D.jl"))
end
using .CBAMLayers

export ChannelAttention3D, SpatialAttention3D, CBAM3D
export ResidualCBAMBlock, SmartPriorNet3D
export replace_nan, nan_valid_mask, add_batch_dim
export generate_prior
export DENSITY_CHANNEL, KG_M3_TO_G_CM3, RESISTIVITY_CHANNEL

# 1-based index of `tensor_channels` name `resistivity` (YAML `id: 0`).
const RESISTIVITY_CHANNEL::Int = 1
# Legacy alias (density is no longer in the EM prior tensor).
const DENSITY_CHANNEL::Int = RESISTIVITY_CHANNEL
# ``1 g/cm^3 = 1000 kg/m^3``.
const KG_M3_TO_G_CM3::Float32 = 0.001f0

# ─────────────────────────────────────────────────────────────────────────────
# NaN masking (fusion empty cells are NaN32)
# ─────────────────────────────────────────────────────────────────────────────

"""
    nan_valid_mask(x) -> AbstractArray{Bool}

`true` where `x` is finite (not `NaN` / `Inf`). Used as the borehole
supervision mask when `x` is a density channel.
"""
nan_valid_mask(x::AbstractArray) = isfinite.(x)

"""
    replace_nan(x; fill=0f0) -> AbstractArray

Replace non-finite entries (`NaN32` empty voxels from the fusion tensor) with
`fill` so the volume can enter a Lux 3-D convolution.

Zygote-safe: uses `ifelse` rather than in-place assignment. Gradient w.r.t.
the original `NaN` locations is zero.

# Layout
Preserves `ndims(x)` and `size(x)`. Typical input `(X, Y, Z, C)` or
`(X, Y, Z, C, B)`.
"""
function replace_nan(x::AbstractArray{T}; fill=zero(T)) where {T}
    return ifelse.(isfinite.(x), x, T(fill))
end

"""
    add_batch_dim(x) -> Array{T,5}

Promote a fusion volume `(X, Y, Z, C)` to `(X, Y, Z, C, 1)`.
5-D arrays are returned unchanged.
"""
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

# ─────────────────────────────────────────────────────────────────────────────
# Residual Conv3D + CBAM block
# ─────────────────────────────────────────────────────────────────────────────

"""
    ResidualCBAMBlock(channels; kernel=3, reduction=4, spatial_kernel=3)

`Conv3D → ReLU → CBAM3D` with a residual skip that preserves `(X, Y, Z, C, B)`.

# Layout
- Input / output: `(X, Y, Z, C, B)` with `C = channels`
- Convolution uses `SamePad()` (stride 1) so the grid is not resampled
"""
struct ResidualCBAMBlock{C,A} <: Lux.AbstractLuxContainerLayer{(:conv, :cbam)}
    conv::C
    cbam::A
end

function ResidualCBAMBlock(channels::Int;
                           kernel::Int=3,
                           reduction::Int=4,
                           spatial_kernel::Int=3)
    isodd(kernel) || throw(ArgumentError("conv kernel must be odd, got $kernel"))
    conv = Conv((kernel, kernel, kernel), channels => channels, relu; pad=SamePad())
    cbam = CBAM3D(channels; reduction=reduction, spatial_kernel=spatial_kernel)
    return ResidualCBAMBlock{typeof(conv),typeof(cbam)}(conv, cbam)
end

function (m::ResidualCBAMBlock)(x::AbstractArray{T,5}, ps, st) where {T}
    # x: (X, Y, Z, C, B)
    h, st_conv = m.conv(x, ps.conv, st.conv)           # (X, Y, Z, C, B)
    a, st_cbam = m.cbam(h, ps.cbam, st.cbam)           # (X, Y, Z, C, B)
    return x .+ a, (; conv=st_conv, cbam=st_cbam)      # residual skip
end

function Base.show(io::IO, ::ResidualCBAMBlock)
    print(io, "ResidualCBAMBlock(Conv3D+CBAM3D, residual)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Smart prior generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    SmartPriorNet3D(; in_channels=4, hidden=16, n_blocks=2, reduction=4,
                      spatial_kernel=3, log_rho_min=-1.0f0, log_rho_max=5.0f0,
                      delta_max=1.0f0, logit_scale=0.10f0)

Lightweight 3-D Conv + CBAM prior network. A `GroupNorm` neck (batch-size-1
safe) centres features before the 1×1×1 heads.

# Keyword arguments
- `in_channels`: fusion channels `C_in` (EM prior tensor has 4)
- `hidden`: feature width of the stem / residual blocks (keep small; a
  `(292, 212, 48)` grid already has ~3×10⁶ voxels)
- `n_blocks`: number of [`ResidualCBAMBlock`](@ref)s
- `reduction`, `spatial_kernel`: forwarded to CBAM
- `log_rho_min`, `log_rho_max`: ``\\log_{10}`` resistivity box for ``m_0``
  (default 0.1 – 10⁵ Ω·m)
- `delta_max`: maximum half-width of ``[m_{min}, m_{max}]`` around ``m_0``
  in log10 decades
- `logit_scale`: multiplies head logits before sigmoid (keeps a random-init
  network in the interior of the resistivity box)

# Forward
`(model)(x, ps, st) -> ((m0, bounds), st′)`

- `x`:      `(X, Y, Z, 4, B)` or `(X, Y, Z, 4)` (batch dim added)
- `m0`:     `(X, Y, Z, 1, B)`  — ``m_0`` in ohm·m
- `bounds`: `(X, Y, Z, 2, B)`  — channel 1 = ``m_{min}``, channel 2 = ``m_{max}``
  with log-space half-widths then ``10^{\\cdot}``
"""
struct SmartPriorNet3D{B,H1,H2} <: Lux.AbstractLuxContainerLayer{(:body, :m0_head, :bounds_head)}
    in_channels::Int
    hidden::Int
    log_rho_min::Float32
    log_rho_max::Float32
    delta_max::Float32
    logit_scale::Float32
    body::B
    m0_head::H1
    bounds_head::H2
end

function _group_count(hidden::Int)::Int
    for g in (8, 4, 2, 1)
        hidden % g == 0 && return g
    end
    return 1
end

function SmartPriorNet3D(;
                         in_channels::Int=4,
                         hidden::Int=16,
                         n_blocks::Int=2,
                         reduction::Int=4,
                         spatial_kernel::Int=3,
                         log_rho_min::Float32=-1.0f0,
                         log_rho_max::Float32=5.0f0,
                         delta_max::Float32=1.0f0,
                         logit_scale::Float32=0.10f0)
    in_channels >= 1 || throw(ArgumentError("in_channels must be ≥ 1"))
    hidden >= 1 || throw(ArgumentError("hidden must be ≥ 1"))
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be ≥ 1"))
    log_rho_max > log_rho_min || throw(ArgumentError("log_rho_max must exceed log_rho_min"))
    delta_max > 0 || throw(ArgumentError("delta_max must be positive"))
    logit_scale > 0 || throw(ArgumentError("logit_scale must be positive"))

    stem = Conv((3, 3, 3), in_channels => hidden, relu; pad=SamePad())
    blocks = ntuple(_ -> ResidualCBAMBlock(hidden;
                                           reduction=reduction,
                                           spatial_kernel=spatial_kernel),
                    n_blocks)
    gn_groups = _group_count(hidden)
    neck = GroupNorm(hidden, gn_groups)
    body = Chain(stem, blocks..., neck)
    m0_head = Conv((1, 1, 1), hidden => 1)       # logits → sigmoid scale
    bounds_head = Conv((1, 1, 1), hidden => 2)   # two positive half-widths

    return SmartPriorNet3D{typeof(body),typeof(m0_head),typeof(bounds_head)}(
        in_channels, hidden, log_rho_min, log_rho_max, delta_max, logit_scale,
        body, m0_head, bounds_head,
    )
end

"""
Map unbounded logits to ``[\\log_{10}\\rho_{min}, \\log_{10}\\rho_{max}]`` via sigmoid.
AD-safe: scale factors are taken as `eltype(logits)`.
"""
function _scale_log_resistivity(logits::AbstractArray{T},
                                log_rho_min::Float32, log_rho_max::Float32) where {T}
    lo = T(log_rho_min)
    hi = T(log_rho_max)
    return lo .+ (hi - lo) .* sigmoid.(logits)
end

"""
Clip without leaving the Zygote graph (`ifelse` rather than `clamp` mutation).
"""
function _clip(x::AbstractArray{T}, lo::T, hi::T) where {T}
    return ifelse.(x .< lo, lo, ifelse.(x .> hi, hi, x))
end

function (m::SmartPriorNet3D)(x::AbstractArray, ps, st)
    x5, squeezed = _ensure_5d(x)
    size(x5, 4) == m.in_channels || throw(DimensionMismatch(
        "expected C=$(m.in_channels) channels, got size(x)=$(size(x5))"))

    # NaN32 empty voxels → 0 before any convolution
    x_in = replace_nan(x5)                              # (X, Y, Z, C_in, B)

    h, st_body = m.body(x_in, ps.body, st.body)         # (X, Y, Z, hidden, B)

    m0_logits, st_m0 = m.m0_head(h, ps.m0_head, st.m0_head)          # (X, Y, Z, 1, B)
    bnd_logits, st_bnd = m.bounds_head(h, ps.bounds_head, st.bounds_head)  # (X, Y, Z, 2, B)

    T = eltype(m0_logits)
    s = T(m.logit_scale)
    lo = T(m.log_rho_min)
    hi = T(m.log_rho_max)
    log_m0 = _scale_log_resistivity(m0_logits .* s, m.log_rho_min, m.log_rho_max)

    δ = T(m.delta_max) .* sigmoid.(bnd_logits .* s)
    δ_lo = δ[:, :, :, 1:1, :]
    δ_hi = δ[:, :, :, 2:2, :]
    log_min = _clip(log_m0 .- δ_lo, lo, hi)
    log_max = _clip(log_m0 .+ δ_hi, lo, hi)
    m0 = exp10.(log_m0)
    bounds = cat(exp10.(log_min), exp10.(log_max); dims=4)

    st_new = (; body=st_body, m0_head=st_m0, bounds_head=st_bnd)

    if squeezed
        m0 = dropdims(m0; dims=5)           # (X, Y, Z, 1)
        bounds = dropdims(bounds; dims=5)   # (X, Y, Z, 2)
    end
    return (m0, bounds), st_new
end

function Base.show(io::IO, m::SmartPriorNet3D)
    print(io, "SmartPriorNet3D(C_in=", m.in_channels,
          ", hidden=", m.hidden,
          ", ρ∈[", exp10(m.log_rho_min), ",", exp10(m.log_rho_max), "] Ω·m",
          ", δ_max=", m.delta_max, " decades)")
end

"""
    generate_prior(model, x, ps, st) -> (m0, bounds, st′)

Convenience wrapper around `(model)(x, ps, st)`.

# Units
`m0` and `bounds` are ohm·m.
"""
function generate_prior(model::SmartPriorNet3D, x::AbstractArray, ps, st)
    (m0, bounds), st_new = model(x, ps, st)
    return m0, bounds, st_new
end

end # module PriorNet3D
