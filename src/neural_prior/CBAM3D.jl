"""
    CBAMLayers

3D Convolutional Block Attention Module (Woo et al. 2018) as explicit Lux.jl
layers. Tensor layout is Julia column-major `(X, Y, Z, C, B)` / Float32.

The module is named `CBAMLayers` so it does not clash with `struct CBAM3D`.

Channel attention: global average + max pool over `(X, Y, Z)`, then a shared
MLP (`Dense`) on the channel axis. Spatial attention: channel-wise mean + max,
concatenated and mixed by a 3D convolution (`3×3×3` or `7×7×7`).

`CBAM3D` applies channel attention then spatial attention. Parameters `ps` and
state `st` are always passed explicitly (no Flux-style implicit params).
"""
module CBAMLayers

using Lux
using Statistics: mean

export ChannelAttention3D, SpatialAttention3D, CBAM3D

# ─────────────────────────────────────────────────────────────────────────────
# Channel attention
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChannelAttention3D(channels; reduction=4)

Squeeze-excitation over the spatial axes of a 5-D feature volume.

# Arguments
- `channels`: input/output channel count `C`
- `reduction`: MLP bottleneck ratio; hidden width is `max(1, C ÷ reduction)`

# Layout
- Input / output: `(X, Y, Z, C, B)`
- GAP / GMP:      `(1, 1, 1, C, B)` → reshape → `(C, B)` for `Dense`
- Weights:        `(1, 1, 1, C, B)`, broadcast-multiplied onto the input
"""
struct ChannelAttention3D{M} <: Lux.AbstractLuxContainerLayer{(:mlp,)}
    channels::Int
    reduction::Int
    mlp::M
end

function ChannelAttention3D(channels::Int; reduction::Int=4)
    channels >= 1 || throw(ArgumentError("channels must be ≥ 1, got $channels"))
    reduction >= 1 || throw(ArgumentError("reduction must be ≥ 1, got $reduction"))
    hidden = max(1, channels ÷ reduction)
    # Shared MLP on the channel axis (CBAM paper). Equivalent to 1×1×1 Conv.
    mlp = Chain(
        Dense(channels, hidden, relu),
        Dense(hidden, channels),
    )
    return ChannelAttention3D{typeof(mlp)}(channels, reduction, mlp)
end

"""
Flatten `(1, 1, 1, C, B)` channel descriptors to `(C, B)` for `Dense`.
"""
@inline function _cvec(x::AbstractArray{T,5}) where {T}
    return reshape(x, size(x, 4), size(x, 5))
end

"""
Unflatten `(C, B)` channel weights back to `(1, 1, 1, C, B)`.
"""
@inline function _cvol(v::AbstractArray{T,2}) where {T}
    return reshape(v, 1, 1, 1, size(v, 1), size(v, 2))
end

function (m::ChannelAttention3D)(x::AbstractArray{T,5}, ps, st) where {T}
    # x: (X, Y, Z, C, B)
    avg = mean(x; dims=(1, 2, 3))            # (1, 1, 1, C, B)
    mx  = maximum(x; dims=(1, 2, 3))         # (1, 1, 1, C, B)

    w_avg, st_mlp = m.mlp(_cvec(avg), ps.mlp, st.mlp)   # (C, B)
    w_max, st_mlp = m.mlp(_cvec(mx),  ps.mlp, st_mlp)    # shared weights
    w = _cvol(sigmoid.(w_avg .+ w_max))                  # (1, 1, 1, C, B)

    return x .* w, (; mlp=st_mlp)            # (X, Y, Z, C, B)
end

function Base.show(io::IO, m::ChannelAttention3D)
    hidden = max(1, m.channels ÷ m.reduction)
    print(io, "ChannelAttention3D(", m.channels,
          "; reduction=", m.reduction, ", hidden=", hidden, ")")
end

# ─────────────────────────────────────────────────────────────────────────────
# Spatial attention
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpatialAttention3D(; kernel=3)

Spatial attention map from channel-wise mean and max statistics.

# Arguments
- `kernel`: odd integer, `3` (lightweight default) or `7` (original CBAM).
  `SamePad()` keeps `(X, Y, Z)` unchanged.

# Layout
- Input / output: `(X, Y, Z, C, B)`
- Channel mean / max: `(X, Y, Z, 1, B)` each
- Concatenated map:   `(X, Y, Z, 2, B)`
- Attention:          `(X, Y, Z, 1, B)`, broadcast over `C`
"""
struct SpatialAttention3D{C} <: Lux.AbstractLuxContainerLayer{(:conv,)}
    kernel::Int
    conv::C
end

function SpatialAttention3D(; kernel::Int=3)
    isodd(kernel) && kernel >= 1 ||
        throw(ArgumentError("spatial kernel must be a positive odd integer, got $kernel"))
    conv = Conv((kernel, kernel, kernel), 2 => 1, sigmoid; pad=SamePad())
    return SpatialAttention3D{typeof(conv)}(kernel, conv)
end

function (m::SpatialAttention3D)(x::AbstractArray{T,5}, ps, st) where {T}
    # x: (X, Y, Z, C, B)
    avg = mean(x; dims=4)                    # (X, Y, Z, 1, B)
    mx  = maximum(x; dims=4)                 # (X, Y, Z, 1, B)
    stats = cat(avg, mx; dims=4)             # (X, Y, Z, 2, B)
    att, st_conv = m.conv(stats, ps.conv, st.conv)  # (X, Y, Z, 1, B)
    return x .* att, (; conv=st_conv)        # (X, Y, Z, C, B)
end

function Base.show(io::IO, m::SpatialAttention3D)
    k = m.kernel
    print(io, "SpatialAttention3D(; kernel=", k, " → ", k, "×", k, "×", k, ")")
end

# ─────────────────────────────────────────────────────────────────────────────
# CBAM block (channel → spatial)
# ─────────────────────────────────────────────────────────────────────────────

"""
    CBAM3D(channels; reduction=4, spatial_kernel=3)

Sequential channel then spatial attention. Residual skip is **not** applied
here; wrap with `ResidualCBAMBlock` (in `PriorNet3D`) if a skip is wanted.

# Layout
Input and output are both `(X, Y, Z, C, B)` with the same `C = channels`.
"""
struct CBAM3D{CA,SA} <: Lux.AbstractLuxContainerLayer{(:channel, :spatial)}
    channel::CA
    spatial::SA
end

function CBAM3D(channels::Int; reduction::Int=4, spatial_kernel::Int=3)
    ca = ChannelAttention3D(channels; reduction=reduction)
    sa = SpatialAttention3D(; kernel=spatial_kernel)
    return CBAM3D{typeof(ca),typeof(sa)}(ca, sa)
end

function (m::CBAM3D)(x::AbstractArray{T,5}, ps, st) where {T}
    # x: (X, Y, Z, C, B)
    y, st_c = m.channel(x, ps.channel, st.channel)     # (X, Y, Z, C, B)
    z, st_s = m.spatial(y, ps.spatial, st.spatial)     # (X, Y, Z, C, B)
    return z, (; channel=st_c, spatial=st_s)
end

function Base.show(io::IO, m::CBAM3D)
    print(io, "CBAM3D(", m.channel.channels,
          "; reduction=", m.channel.reduction,
          ", spatial_kernel=", m.spatial.kernel, ")")
end

end # module CBAMLayers
