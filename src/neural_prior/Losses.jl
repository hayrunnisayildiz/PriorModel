"""
    Losses

Prior-fitting losses for [`PriorNet3D.SmartPriorNet3D`](@ref).

The data term is evaluated only at known borehole density voxels (a boolean
or 0/1 mask). Spatial continuity is an anisotropic 3-D total variation (TV)
regulariser on ``m_0``.

All reductions return a scalar `eltype(m0)` so the objective is drop-in
compatible with Zygote.jl and Optimization.jl.

# Units
- `m0`, `target`: ``g/cm^3``
- Fusion-tensor density (channel 10) is ``kg/m^3``; use
  [`extract_density_target`](@ref) to convert and build the mask
"""
module Losses

using Statistics: mean

export smooth_l1, masked_data_loss, total_variation
export prior_loss, prior_loss_terms, extract_density_target
export KG_M3_TO_G_CM3, DENSITY_CHANNEL

const DENSITY_CHANNEL::Int = 10
const KG_M3_TO_G_CM3::Float32 = 0.001f0

# ─────────────────────────────────────────────────────────────────────────────
# Shape helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
Promote a volume to `(X, Y, Z, C, B)`.

- 3-D `(X, Y, Z)`     → `(X, Y, Z, 1, 1)`
- 4-D `(X, Y, Z, C)`  → `(X, Y, Z, C, 1)`
- 5-D left unchanged
"""
function _as_5d(x::AbstractArray)
    nd = ndims(x)
    nd == 5 && return x
    nd == 4 && return reshape(x, size(x)..., 1)
    nd == 3 && return reshape(x, size(x)..., 1, 1)
    throw(ArgumentError("expected 3–5D array, got ndims=$nd"))
end

function _active(mask::AbstractArray)
    return mask .> 0
end

"""
Masked mean of `vals` (already non-negative). Empty mask → `0`.
"""
function _masked_mean(vals::AbstractArray{T}, active::AbstractArray) where {T}
    n = count(active)
    n == 0 && return zero(T)
    s = sum(ifelse.(active, vals, zero(T)))
    return s / T(n)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pointwise losses
# ─────────────────────────────────────────────────────────────────────────────

"""
    smooth_l1(residual; β=1f0)

Huber / Smooth-L1 of a residual array, elementwise.

```
L =  ½ r² / β     if |r| < β
     |r| − ½ β    otherwise
```

`β` is in the same unit as `residual` (``g/cm^3`` for density).
"""
function smooth_l1(residual::AbstractArray{T}; β::Real=1.0f0) where {T}
    βT = T(β)
    ar = abs.(residual)
    quad = T(0.5) .* ar .* ar ./ βT
    lin  = ar .- T(0.5) * βT
    return ifelse.(ar .< βT, quad, lin)
end

"""
    masked_data_loss(pred, target, mask; kind=:smooth_l1, β=0.05f0)

Mean regression loss on voxels where `mask > 0`.

# Arguments
- `pred`, `target`: density, ``g/cm^3``, broadcastable to `(X, Y, Z, 1, B)`
- `mask`: `Bool` or `{0,1}` at known borehole cells
- `kind`: `:smooth_l1` (default) or `:l2` (mean squared error / 2)
- `β`: Huber threshold, ``g/cm^3`` (ignored for `:l2`)

# Returns
Scalar `eltype(pred)`.
"""
function masked_data_loss(pred::AbstractArray, target::AbstractArray, mask::AbstractArray;
                          kind::Symbol=:smooth_l1, β::Float32=0.05f0)
    p = _as_5d(pred)
    t = _as_5d(target)
    m = _as_5d(mask)
    size(p)[1:3] == size(t)[1:3] == size(m)[1:3] || throw(DimensionMismatch(
        "pred/target/mask spatial size mismatch: $(size(p)) vs $(size(t)) vs $(size(m))"))

    r = p .- t                                       # (X, Y, Z, C, B), g/cm³
    active = _active(m)
    if kind === :smooth_l1
        return _masked_mean(smooth_l1(r; β=β), active)
    elseif kind === :l2
        return _masked_mean(T_half(r) .* r .* r, active)
    else
        throw(ArgumentError("kind must be :smooth_l1 or :l2, got $(repr(kind))"))
    end
end

@inline T_half(r::AbstractArray{T}) where {T} = T(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Total variation
# ─────────────────────────────────────────────────────────────────────────────

"""
    total_variation(m; mode=:anisotropic)

Mean 3-D total variation of a volume `(X, Y, Z, C, B)`.

- `:anisotropic` — ``\\mathrm{mean}|Δx| + \\mathrm{mean}|Δy| + \\mathrm{mean}|Δz|``
- `:isotropic`   — ``\\mathrm{mean}\\sqrt{Δx^2+Δy^2+Δz^2+ε}`` (face-neighbour
  diffs are aligned on the inner `(X-1, Y-1, Z-1)` block)

# Units
If `m` is ``g/cm^3``, TV has the same unit (per voxel face). Using `mean`
keeps the scale independent of grid size so `λ_tv` transfers across
`GridSpec` resolutions.
"""
function total_variation(m::AbstractArray; mode::Symbol=:anisotropic, ε::Float32=1.0f-8)
    v = _as_5d(m)
    # v: (X, Y, Z, C, B)
    dx = v[2:end, :, :, :, :] .- v[1:end-1, :, :, :, :]     # (X-1, Y, Z, C, B)
    dy = v[:, 2:end, :, :, :] .- v[:, 1:end-1, :, :, :]     # (X, Y-1, Z, C, B)
    dz = v[:, :, 2:end, :, :] .- v[:, :, 1:end-1, :, :]     # (X, Y, Z-1, C, B)

    if mode === :anisotropic
        return mean(abs, dx) + mean(abs, dy) + mean(abs, dz)
    elseif mode === :isotropic
        T = eltype(v)
        # Align to the inner cube so the three diffs share an index
        dx_i = dx[:, 1:end-1, 1:end-1, :, :]
        dy_i = dy[1:end-1, :, 1:end-1, :, :]
        dz_i = dz[1:end-1, 1:end-1, :, :, :]
        return mean(sqrt.(dx_i .* dx_i .+ dy_i .* dy_i .+ dz_i .* dz_i .+ T(ε)))
    else
        throw(ArgumentError("mode must be :anisotropic or :isotropic, got $(repr(mode))"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Combined prior objective
# ─────────────────────────────────────────────────────────────────────────────

"""
    prior_loss(m0, target, mask; λ_tv=1f-3, kind=:smooth_l1, β=0.05f0,
               tv_mode=:anisotropic, bounds=nothing, λ_bounds=0f0)

``L = L_{data} + λ_{tv}\\, TV(m_0) [+ λ_{bounds} L_{box}]``.

# Arguments
- `m0`: predicted ``m_0``, ``g/cm^3``, `(X, Y, Z, 1[, B])`
- `target`: borehole density, ``g/cm^3``, same spatial size
- `mask`: known-cell mask (finite petrophysics voxels)
- `λ_tv`: TV weight (dimensionless relative to ``g/cm^3`` data loss)
- `bounds`: optional `(X, Y, Z, 2[, B])` with ``m_{min}, m_{max}``; if given
  and `λ_bounds > 0`, a hinge penalises ``m_0 \\notin [m_{min}, m_{max}]``
  and ``m_{min} > m_{max}``

# Returns
Scalar loss (`eltype(m0)`).
"""
function prior_loss(m0::AbstractArray, target::AbstractArray, mask::AbstractArray;
                    λ_tv::Float32=1.0f-3,
                    kind::Symbol=:smooth_l1,
                    β::Float32=0.05f0,
                    tv_mode::Symbol=:anisotropic,
                    bounds::Union{Nothing,AbstractArray}=nothing,
                    λ_bounds::Float32=0.0f0)
    terms = prior_loss_terms(m0, target, mask;
                             λ_tv=λ_tv, kind=kind, β=β, tv_mode=tv_mode,
                             bounds=bounds, λ_bounds=λ_bounds)
    return terms.total
end

"""
    prior_loss_terms(m0, target, mask; kwargs...) -> NamedTuple

Same as [`prior_loss`](@ref) but returns `(; total, data, tv, bounds)` for
logging. `total` is the Optimization.jl scalar.
"""
function prior_loss_terms(m0::AbstractArray, target::AbstractArray, mask::AbstractArray;
                          λ_tv::Float32=1.0f-3,
                          kind::Symbol=:smooth_l1,
                          β::Float32=0.05f0,
                          tv_mode::Symbol=:anisotropic,
                          bounds::Union{Nothing,AbstractArray}=nothing,
                          λ_bounds::Float32=0.0f0)
    L_data = masked_data_loss(m0, target, mask; kind=kind, β=β)
    L_tv   = total_variation(m0; mode=tv_mode)
    T = eltype(L_data)
    L_box = zero(T)
    if bounds !== nothing && λ_bounds > 0
        L_box = bounds_hinge(m0, bounds)
    end
    total = L_data + T(λ_tv) * L_tv + T(λ_bounds) * L_box
    return (; total, data=L_data, tv=L_tv, bounds=L_box)
end

"""
Hinge that is zero iff ``m_{min} \\le m_0 \\le m_{max}``.
"""
function bounds_hinge(m0::AbstractArray, bounds::AbstractArray)
    ρ  = _as_5d(m0)                          # (X, Y, Z, 1, B)
    b  = _as_5d(bounds)                      # (X, Y, Z, 2, B)
    mn = b[:, :, :, 1:1, :]
    mx = b[:, :, :, 2:2, :]
    T = eltype(ρ)
    below = ifelse.(ρ .< mn, mn .- ρ, zero(T))
    above = ifelse.(ρ .> mx, ρ .- mx, zero(T))
    flipped = ifelse.(mn .> mx, mn .- mx, zero(T))
    return mean(below .+ above .+ flipped)
end

# ─────────────────────────────────────────────────────────────────────────────
# Supervision from the fusion tensor
# ─────────────────────────────────────────────────────────────────────────────

"""
    extract_density_target(tensor; channel=10, from_unit=:kg_m3)
        -> (target, mask)

Pull the borehole density channel out of a fusion volume.

# Arguments
- `tensor`: `(X, Y, Z, C)` or `(X, Y, Z, C, B)` `Float32` fusion tensor
- `channel`: 1-based density index (`tensor_channels` YAML `id: 9` → 10)
- `from_unit`: `:kg_m3` (Voxelizer default) converts with `/1000` to
  ``g/cm^3``; `:g_cm3` leaves values unchanged

# Returns
- `target`: `(X, Y, Z, 1, B)` density in ``g/cm^3`` (non-finite → `0`)
- `mask`:   `(X, Y, Z, 1, B)` `Bool`, `true` at finite petrophysics voxels
"""
function extract_density_target(tensor::AbstractArray{T};
                                channel::Int=DENSITY_CHANNEL,
                                from_unit::Symbol=:kg_m3) where {T}
    x = _as_5d(tensor)                       # (X, Y, Z, C, B)
    C = size(x, 4)
    (1 <= channel <= C) || throw(BoundsError(x, (:, :, :, channel, :)))

    sl = x[:, :, :, channel:channel, :]      # (X, Y, Z, 1, B)
    mask = isfinite.(sl)
    target = ifelse.(mask, sl, zero(T))
    if from_unit === :kg_m3
        target = target .* T(KG_M3_TO_G_CM3)
    elseif from_unit !== :g_cm3
        throw(ArgumentError("from_unit must be :kg_m3 or :g_cm3, got $(repr(from_unit))"))
    end
    return target, mask
end

end # module Losses
