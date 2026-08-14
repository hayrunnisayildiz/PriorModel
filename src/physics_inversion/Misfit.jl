"""
    Misfit

Physics-constrained inversion objective for 3-D gravity.

Combines data misfit ``\\Phi_d``, Tikhonov prior ``\\Phi_m``, bound penalties,
and 3-D total variation on the density model ``m`` (``g/cm^3``).

All scalar reductions return `Float32` for type stability and drop-in use with
`Optimization.jl`. Individual loss terms use Zygote-friendly broadcast reductions;
[`total_inversion_loss`](@ref) also calls [`ForwardGravity.forward_gravity`](@ref)
(which is not Zygote-differentiable through its `@turbo` kernel — use
`ForwardDiff` / `Enzyme` / an adjoint wrapper for gradient-based inversion).

# Units
- gravity observations / predictions: mGal; data misfit: mGal²
- density model / prior: ``g/cm^3``; prior & TV terms: ``(g/cm^3)^2`` (mean-normalised
  where noted)
"""
module Misfit

using Statistics: mean

if !isdefined(@__MODULE__, :ForwardGravity)
    include(joinpath(@__DIR__, "ForwardGravity.jl"))
end
using .ForwardGravity
using .ForwardGravity.GridSpecs

export data_misfit, prior_regularization, bounds_penalty, total_variation
export total_inversion_loss, total_inversion_loss_terms

# ─────────────────────────────────────────────────────────────────────────────
# Shape helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Check two 3-D arrays share shape; return `size(m)` on success."""
function _same_shape3(a::Array{Float32,3}, b::Array{Float32,3}, name_a::String, name_b::String)
    sa, sb = size(a), size(b)
    sa == sb || throw(DimensionMismatch(
        "`$name_a` size $sa ≠ `$name_b` size $sb"))
    return sa
end

"""Check gravity vectors share length."""
function _same_length(a::Vector{Float32}, b::Vector{Float32}, name_a::String, name_b::String)
    la, lb = length(a), length(b)
    la == lb || throw(DimensionMismatch(
        "`$name_a` length $la ≠ `$name_b` length $lb"))
    return la
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss terms
# ─────────────────────────────────────────────────────────────────────────────

"""
    data_misfit(d_obs, g_pred) -> Float32

Squared L2 data misfit between observed and predicted gravity:

``\\Phi_d = \\| d_{obs} - g_{pred} \\|_2^2 = \\sum_i (d_i - g_i)^2``

# Arguments
- `d_obs::Vector{Float32}`: observed gravity, mGal
- `g_pred::Vector{Float32}`: forward-model prediction, mGal

# Returns
Scalar `Float32`, units mGal².
"""
function data_misfit(d_obs::Vector{Float32}, g_pred::Vector{Float32})::Float32
    _same_length(d_obs, g_pred, "d_obs", "g_pred")
    r = d_obs .- g_pred
    return sum(r .* r)
end

"""
    prior_regularization(m, m0) -> Float32

Tikhonov prior penalty anchoring ``m`` to the CBAM smart prior ``m_0``:

``\\Phi_m = \\| m - m_0 \\|_2^2 = \\sum_{ijk} (m_{ijk} - m_{0,ijk})^2``

# Arguments
- `m`, `m0::Array{Float32,3}`: current and prior density, ``g/cm^3``, `(X, Y, Z)`

# Returns
Scalar `Float32`, units ``(g/cm^3)^2`` (summed over voxels).
"""
@views function prior_regularization(m::Array{Float32,3},
                                     m0::Array{Float32,3})::Float32
    _same_shape3(m, m0, "m", "m0")
    r = m .- m0
    return sum(r .* r)
end

"""
    bounds_penalty(m, m_min, m_max) -> Float32

Soft quadratic penalty when ``m`` leaves the smart-prior box
``[m_{min}, m_{max}]``:

``\\Phi_b = \\mathrm{mean}\\bigl[\\max(0, m_{min}-m)^2 + \\max(0, m-m_{max})^2\\bigr]``

Uses `ifelse` (not `max`) for clean Zygote gradients at the boundary.

# Arguments
- `m`, `m_min`, `m_max::Array{Float32,3}`: density and bounds, ``g/cm^3``, `(X, Y, Z)`

# Returns
Scalar `Float32`, mean squared violation per voxel.
"""
@views function bounds_penalty(m::Array{Float32,3},
                               m_min::Array{Float32,3},
                               m_max::Array{Float32,3})::Float32
    _same_shape3(m, m_min, "m", "m_min")
    _same_shape3(m, m_max, "m", "m_max")
    below = ifelse.(m .< m_min, m_min .- m, zero(Float32))
    above = ifelse.(m .> m_max, m .- m_max, zero(Float32))
    return mean(below .* below .+ above .* above)
end

@inline _tv_mean(f::F, a::AbstractArray) where {F} =
    isempty(a) ? zero(Float32) : mean(f, a)

"""
    total_variation(m; mode=:anisotropic) -> Float32

Mean 3-D total variation of density volume `(X, Y, Z)`.

- `:anisotropic` — ``\\mathrm{mean}|Δx| + \\mathrm{mean}|Δy| + \\mathrm{mean}|Δz|``
- `:isotropic`   — ``\\mathrm{mean}\\sqrt{Δx^2+Δy^2+Δz^2+ε}`` on the inner cube

Grid-size normalisation via `mean` keeps ``λ_{tv}`` portable across resolutions
(see also [`Losses.total_variation`](@ref) for the 5-D prior-net variant).

# Units
Same as `m` (``g/cm^3`` when `m` is density).
"""
@views function total_variation(m::Array{Float32,3};
                                mode::Symbol=:anisotropic,
                                ε::Float32=1.0f-8)::Float32
    dx = m[2:end, :, :] .- m[1:end-1, :, :]
    dy = m[:, 2:end, :] .- m[:, 1:end-1, :]
    dz = m[:, :, 2:end] .- m[:, :, 1:end-1]

    if mode === :anisotropic
        return _tv_mean(abs, dx) + _tv_mean(abs, dy) + _tv_mean(abs, dz)
    elseif mode === :isotropic
        if size(m, 1) < 2 || size(m, 2) < 2 || size(m, 3) < 2
            return zero(Float32)
        end
        dx_i = dx[:, 1:end-1, 1:end-1]
        dy_i = dy[1:end-1, :, 1:end-1]
        dz_i = dz[1:end-1, 1:end-1, :]
        return mean(sqrt.(dx_i .* dx_i .+ dy_i .* dy_i .+ dz_i .* dz_i .+ ε))
    else
        throw(ArgumentError("mode must be :anisotropic or :isotropic, got $(repr(mode))"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Combined inversion objective
# ─────────────────────────────────────────────────────────────────────────────

"""
    total_inversion_loss_terms(m, d_obs, m0, m_min, m_max, grid_spec,
                               obs_x, obs_y, obs_z; kwargs...) -> NamedTuple

Evaluate all inversion loss components.

``L = \\Phi_d + λ_{prior}\\,\\Phi_m + λ_{tv}\\,TV(m) + λ_{bounds}\\,\\Phi_b``

# Returns
`(; total, data, prior, tv, bounds, g_pred)` — all scalars except `g_pred`
(`Vector{Float32}`, mGal) from the forward model.
"""
function total_inversion_loss_terms(m::Array{Float32,3},
                                    d_obs::Vector{Float32},
                                    m0::Array{Float32,3},
                                    m_min::Array{Float32,3},
                                    m_max::Array{Float32,3},
                                    grid_spec::GridSpec,
                                    obs_x::Vector{Float32},
                                    obs_y::Vector{Float32},
                                    obs_z::Vector{Float32};
                                    λ_prior::Float32=1.0f-2,
                                    λ_tv::Float32=1.0f-3,
                                    λ_bounds::Float32=1.0f2,
                                    tv_mode::Symbol=:anisotropic,
                                    kernel_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    _same_shape3(m, m0, "m", "m0")
    _same_shape3(m, m_min, "m", "m_min")
    _same_shape3(m, m_max, "m", "m_max")

    # Prefer the precomputed linear operator (BLAS matvec). The prism loop is
    # only used when no kernel is supplied (e.g. a one-off forward check).
    g_pred = kernel_matrix === nothing ?
        forward_gravity(m, grid_spec, obs_x, obs_y, obs_z) :
        apply_gravity_kernel(kernel_matrix, m)
    L_data   = data_misfit(d_obs, g_pred)
    L_prior  = prior_regularization(m, m0)
    L_tv     = total_variation(m; mode=tv_mode)
    L_bounds = bounds_penalty(m, m_min, m_max)

    total = L_data +
            λ_prior * L_prior +
            λ_tv * L_tv +
            λ_bounds * L_bounds

    return (; total, data=L_data, prior=L_prior, tv=L_tv, bounds=L_bounds, g_pred)
end

"""
    total_inversion_loss(m, d_obs, m0, m_min, m_max, grid_spec,
                         obs_x, obs_y, obs_z; kwargs...) -> Float32

Full physics-constrained inversion objective for `Optimization.jl`:

``L = \\|d_{obs} - g(m)\\|_2^2
      + λ_{prior}\\|m - m_0\\|_2^2
      + λ_{tv}\\, TV(m)
      + λ_{bounds}\\,\\Phi_b(m)``

# Arguments
- `m::Array{Float32,3}`: invertible density model, ``g/cm^3``, `(X, Y, Z)`
- `d_obs::Vector{Float32}`: observed gravity, mGal
- `m0`, `m_min`, `m_max::Array{Float32,3}`: smart prior centre and bounds, ``g/cm^3``
- `grid_spec::GridSpec`: voxel geometry for [`forward_gravity`](@ref)
- `obs_x`, `obs_y`, `obs_z::Vector{Float32}`: station coordinates, m

# Keyword weights
- `λ_prior=1f-2`: Tikhonov anchor to ``m_0``
- `λ_tv=1f-3`: total-variation smoothness
- `λ_bounds=1f2`: box constraint softness
- `tv_mode=:anisotropic`: `:anisotropic` or `:isotropic` TV

# Returns
Scalar `Float32` — pass to `Optimization.jl` as the objective.

See [`total_inversion_loss_terms`](@ref) for per-term logging.
"""
function total_inversion_loss(m::Array{Float32,3},
                              d_obs::Vector{Float32},
                              m0::Array{Float32,3},
                              m_min::Array{Float32,3},
                              m_max::Array{Float32,3},
                              grid_spec::GridSpec,
                              obs_x::Vector{Float32},
                              obs_y::Vector{Float32},
                              obs_z::Vector{Float32};
                              λ_prior::Float32=1.0f-2,
                              λ_tv::Float32=1.0f-3,
                              λ_bounds::Float32=1.0f2,
                              tv_mode::Symbol=:anisotropic,
                              kernel_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing)::Float32
    return total_inversion_loss_terms(
        m, d_obs, m0, m_min, m_max, grid_spec, obs_x, obs_y, obs_z;
        λ_prior=λ_prior, λ_tv=λ_tv, λ_bounds=λ_bounds, tv_mode=tv_mode,
        kernel_matrix=kernel_matrix).total
end

end # module Misfit
