"""
    InversionSolver

Physics-constrained 3-D gravity inversion driver built on `Optimization.jl` /
`OptimizationOptimJL`.

Maps the smart-prior centre ``m_0`` and bounds ``[m_{min}, m_{max}]`` onto a
refined density model ``m`` by minimising [`Misfit.total_inversion_loss`](@ref)
(data misfit + prior + TV + bounds).

The 3-D model is stored column-major `(X, Y, Z)` and vectorised with `vec(m)`
for the optimizer (same layout as Lux / fusion tensors).

# Automatic differentiation
Gravity is **linear** in density (``g = G m``). The solver precomputes the
Nagy sensitivity matrix ``G`` once, then uses an analytic gradient
(``\\nabla \\Phi_d = 2 G^\\top (G m - d_{obs})`` plus prior / TV / bounds).
Do **not** use `AutoFiniteDiff` on the voxel vector — that costs
``n_{voxels}`` prism forwards per L-BFGS step and will hang a laptop.

# Units
- density: ``g/cm^3``
- gravity: mGal
"""
module InversionSolver

using Logging
using Optimization
using OptimizationOptimJL

if !isdefined(@__MODULE__, :Misfit)
    include(joinpath(@__DIR__, "Misfit.jl"))
end
using .Misfit
using .Misfit.ForwardGravity: gravity_kernel_matrix
using .Misfit.ForwardGravity.GridSpecs

import OptimizationOptimJL: LBFGS

export InversionHistory, InversionParams
export prepare_inversion_vectors, build_inversion_problem
export run_physics_inversion

# ─────────────────────────────────────────────────────────────────────────────
# Parameter bundle & history
# ─────────────────────────────────────────────────────────────────────────────

"""
    InversionParams

Type-stable problem data passed as `p` to `OptimizationFunction`.

All arrays use `Float32`; `m_shape` is `(X, Y, Z)` column-major.
"""
struct InversionParams
    m_shape::NTuple{3,Int}
    d_obs::Vector{Float32}
    m0::Array{Float32,3}
    m_min::Array{Float32,3}
    m_max::Array{Float32,3}
    grid_spec::GridSpec
    obs_x::Vector{Float32}
    obs_y::Vector{Float32}
    obs_z::Vector{Float32}
    kernel::Matrix{Float32}          # G: (n_obs, n_voxels), mGal / (g/cm³)
    λ_prior::Float32
    λ_tv::Float32
    λ_bounds::Float32
    tv_mode::Symbol
end

"""
    InversionHistory

Loss trajectory recorded during [`run_physics_inversion`](@ref).

Fields mirror [`Misfit.total_inversion_loss_terms`](@ref) plus `iter` count.
Index `1` is the initial model (before the first optimizer step).
"""
mutable struct InversionHistory
    iter::Int
    total::Vector{Float32}
    data::Vector{Float32}
    prior::Vector{Float32}
    tv::Vector{Float32}
    bounds::Vector{Float32}
end

function InversionHistory()
    return InversionHistory(0, Float32[], Float32[], Float32[], Float32[], Float32[])
end

function _record!(hist::InversionHistory, terms)
    push!(hist.total, terms.total)
    push!(hist.data, terms.data)
    push!(hist.prior, terms.prior)
    push!(hist.tv, terms.tv)
    push!(hist.bounds, terms.bounds)
    return hist
end

# ─────────────────────────────────────────────────────────────────────────────
# Vectorisation (column-major)
# ─────────────────────────────────────────────────────────────────────────────

"""
    prepare_inversion_vectors(m0, m_min, m_max) -> (u0, lb, ub, shape)

Flatten 3-D density arrays for `OptimizationProblem` (column-major `vec`).

Returns `(u0, lb, ub)` as `Vector{Float32}` and `shape = size(m0)`.
"""
function prepare_inversion_vectors(m0::Array{Float32,3},
                                   m_min::Array{Float32,3},
                                   m_max::Array{Float32,3})
    shape = size(m0)
    shape == size(m_min) == size(m_max) ||
        throw(DimensionMismatch(
            "m0 size $shape ≠ m_min $(size(m_min)) or m_max $(size(m_max))"))
    return copy(vec(m0)), copy(vec(m_min)), copy(vec(m_max)), shape
end

function _validate_inversion_setup(m0::Array{Float32,3},
                                   m_min::Array{Float32,3},
                                   m_max::Array{Float32,3},
                                   d_obs::Vector{Float32},
                                   grid_spec::GridSpec,
                                   obs_x::Vector{Float32},
                                   obs_y::Vector{Float32},
                                   obs_z::Vector{Float32})
    nxx, nyy, nzz = nxyz(grid_spec)
    shape = size(m0)
    shape == (nxx, nyy, nzz) ||
        error("model size $shape ≠ GridSpec $(nxyz(grid_spec))")
    prepare_inversion_vectors(m0, m_min, m_max)
    nobs = length(obs_x)
    length(obs_y) == nobs && length(obs_z) == nobs ||
        throw(DimensionMismatch("observation coordinate lengths differ"))
    length(d_obs) == nobs ||
        throw(DimensionMismatch("d_obs length $(length(d_obs)) ≠ nobs $nobs"))
    return shape, nobs
end

@inline function _reshape_model(u::AbstractVector, shape::NTuple{3,Int})
    return reshape(copy(_as_f32(u)), shape)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analytic gradient (linear gravity + prior + anisotropic TV + bounds)
# ─────────────────────────────────────────────────────────────────────────────

@inline _as_f32(u::AbstractVector) = convert(Vector{Float32}, u)

"""Anisotropic TV: ``mean|Δx| + mean|Δy| + mean|Δz|``. Accumulates ``λ ∇TV`` into `g`."""
function _accum_tv_anisotropic!(g::Array{Float32,3}, m::Array{Float32,3}, λ::Float32)
    λ == 0 && return g
    nxx, nyy, nzz = size(m)
    if nxx >= 2
        invN = λ / Float32((nxx - 1) * nyy * nzz)
        @inbounds for k in 1:nzz, j in 1:nyy, i in 1:(nxx - 1)
            s = sign(m[i + 1, j, k] - m[i, j, k])
            g[i, j, k]     -= invN * s
            g[i + 1, j, k] += invN * s
        end
    end
    if nyy >= 2
        invN = λ / Float32(nxx * (nyy - 1) * nzz)
        @inbounds for k in 1:nzz, j in 1:(nyy - 1), i in 1:nxx
            s = sign(m[i, j + 1, k] - m[i, j, k])
            g[i, j, k]     -= invN * s
            g[i, j + 1, k] += invN * s
        end
    end
    if nzz >= 2
        invN = λ / Float32(nxx * nyy * (nzz - 1))
        @inbounds for k in 1:(nzz - 1), j in 1:nyy, i in 1:nxx
            s = sign(m[i, j, k + 1] - m[i, j, k])
            g[i, j, k]     -= invN * s
            g[i, j, k + 1] += invN * s
        end
    end
    return g
end

"""Soft box penalty gradient of ``λ mean(hinge²)`` accumulated into `g`."""
function _accum_bounds_grad!(g::Array{Float32,3},
                             m::Array{Float32,3},
                             m_min::Array{Float32,3},
                             m_max::Array{Float32,3},
                             λ::Float32)
    λ == 0 && return g
    s = (2.0f0 * λ) / Float32(length(m))
    @inbounds for i in eachindex(m)
        if m[i] < m_min[i]
            g[i] += s * (m[i] - m_min[i])
        elseif m[i] > m_max[i]
            g[i] += s * (m[i] - m_max[i])
        end
    end
    return g
end

"""
    _inversion_objective(u, p) -> Float32

Scalar loss; forward gravity is ``G \\, u`` (no prism loop).
"""
function _inversion_objective(u::AbstractVector, p::InversionParams)::Float32
    m = _reshape_model(u, p.m_shape)
    return total_inversion_loss(
        m, p.d_obs, p.m0, p.m_min, p.m_max, p.grid_spec,
        p.obs_x, p.obs_y, p.obs_z;
        λ_prior=p.λ_prior, λ_tv=p.λ_tv, λ_bounds=p.λ_bounds, tv_mode=p.tv_mode,
        kernel_matrix=p.kernel)
end

"""
    _inversion_grad!(grad, u, p) -> grad

Analytic ``∇L``. Data term: ``2 G^\\top (G u - d_{obs})``.
TV gradient is implemented for `:anisotropic` (the default).
"""
function _inversion_grad!(grad, u::AbstractVector, p::InversionParams)
    u32 = _as_f32(u)
    m = reshape(u32, p.m_shape)
    m0v = vec(p.m0)

    g_pred = p.kernel * u32                          # n_obs
    work = 2.0f0 .* (p.kernel' * (g_pred .- p.d_obs))  # n_voxels, data term

    λ2 = 2.0f0 * p.λ_prior
    @inbounds for i in eachindex(work)
        work[i] += λ2 * (u32[i] - m0v[i])
    end

    gm = reshape(work, p.m_shape)
    if p.tv_mode === :anisotropic
        _accum_tv_anisotropic!(gm, m, p.λ_tv)
    elseif p.λ_tv != 0
        @warn "analytic TV gradient only for :anisotropic; skipping TV grad" tv_mode=p.tv_mode
    end
    _accum_bounds_grad!(gm, m, p.m_min, p.m_max, p.λ_bounds)

    copyto!(grad, work)
    return grad
end

"""
    build_inversion_problem(m0, m_min, m_max, d_obs, grid_spec,
                            obs_x, obs_y, obs_z; kwargs...)
        -> (prob, params, history)

Construct an `OptimizationProblem` with box constraints
`lb = vec(m_min)`, `ub = vec(m_max)` and initial guess `u0 = vec(m0)`.

# Keyword arguments
Same weights / AD options as [`run_physics_inversion`](@ref).

# Returns
- `prob`: `OptimizationProblem`
- `params`: [`InversionParams`](@ref)
- `history`: [`InversionHistory`](@ref) with the initial loss logged
"""
function build_inversion_problem(m0::Array{Float32,3},
                                 m_min::Array{Float32,3},
                                 m_max::Array{Float32,3},
                                 d_obs::Vector{Float32},
                                 grid_spec::GridSpec,
                                 obs_x::Vector{Float32},
                                 obs_y::Vector{Float32},
                                 obs_z::Vector{Float32};
                                 λ_prior::Float32=1.0f-2,
                                 λ_tv::Float32=1.0f-3,
                                 λ_bounds::Float32=1.0f2,
                                 tv_mode::Symbol=:anisotropic,
                                 adtype=nothing)
    shape, nobs = _validate_inversion_setup(
        m0, m_min, m_max, d_obs, grid_spec, obs_x, obs_y, obs_z)

    u0, lb, ub, _ = prepare_inversion_vectors(m0, m_min, m_max)

    nvox = length(u0)
    @info "Assembling gravity kernel G" size=(nobs, nvox) n_obs=nobs n_voxels=nvox
    kernel = gravity_kernel_matrix(grid_spec, obs_x, obs_y, obs_z)
    @info "Gravity kernel ready" mem_MiB=round(sizeof(kernel) / 2^20; digits=2)

    params = InversionParams(
        shape, d_obs, m0, m_min, m_max, grid_spec,
        obs_x, obs_y, obs_z, kernel, λ_prior, λ_tv, λ_bounds, tv_mode)

    history = InversionHistory()
    terms0 = total_inversion_loss_terms(
        m0, d_obs, m0, m_min, m_max, grid_spec, obs_x, obs_y, obs_z;
        λ_prior=λ_prior, λ_tv=λ_tv, λ_bounds=λ_bounds, tv_mode=tv_mode,
        kernel_matrix=kernel)
    _record!(history, terms0)

    # Explicit analytic grad — never finite-difference the prism operator.
    optf = if adtype === nothing
        OptimizationFunction(_inversion_objective; grad=_inversion_grad!)
    else
        OptimizationFunction(_inversion_objective, adtype; grad=_inversion_grad!)
    end
    prob = OptimizationProblem(optf, u0, params; lb=lb, ub=ub)
    return prob, params, history
end

function _make_inversion_callback(params::InversionParams,
                                  history::InversionHistory;
                                  verbose::Bool=true)
    last_logged_iter = Ref(-1)
    return function (state, l)
        step = state.iter
        step == last_logged_iter[] && return false
        last_logged_iter[] = step
        history.iter = step

        m = _reshape_model(state.u, params.m_shape)
        terms = total_inversion_loss_terms(
            m, params.d_obs, params.m0, params.m_min, params.m_max,
            params.grid_spec, params.obs_x, params.obs_y, params.obs_z;
            λ_prior=params.λ_prior, λ_tv=params.λ_tv,
            λ_bounds=params.λ_bounds, tv_mode=params.tv_mode,
            kernel_matrix=params.kernel)
        _record!(history, terms)
        if verbose
            @info "inversion iter=$step" total=terms.total data=terms.data prior=terms.prior tv=terms.tv bounds=terms.bounds
        end
        return false
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_physics_inversion(m0, m_min, m_max, d_obs, grid_spec,
                          obs_x, obs_y, obs_z; kwargs...)
        -> (; m_final, history, sol)

Run box-constrained L-BFGS (default) physics inversion starting from the
CBAM smart prior ``m_0``.

# Arguments
- `m0`, `m_min`, `m_max::Array{Float32,3}`: prior centre and bounds, ``g/cm^3``
- `d_obs::Vector{Float32}`: observed gravity, mGal
- `grid_spec::GridSpec`: voxel geometry
- `obs_x`, `obs_y`, `obs_z::Vector{Float32}`: station coordinates, m

# Keywords
- `maxiters::Int=100`: optimizer iteration cap (outer + inner for box constraints)
- `λ_prior::Float32=1f-2`, `λ_tv::Float32=1f-3`, `λ_bounds::Float32=1f2`
- `optimizer=LBFGS()`: any `Optimization.jl` solver (e.g. `LBFGS()`)
- `adtype=nothing`: leave `nothing` to use the analytic gravity gradient.
  Passing `Optimization.AutoFiniteDiff()` will still attach the analytic `grad`
  (do not finite-difference the prism loop).
- `tv_mode::Symbol=:anisotropic`
- `verbose::Bool=true`: log loss components each callback

# Returns
`(; m_final, history, sol)` where
- `m_final::Array{Float32,3}`: inverted density model, ``g/cm^3``
- `history::InversionHistory`: loss / misfit trace
- `sol`: raw `Optimization.jl` solution object
"""
function run_physics_inversion(m0::Array{Float32,3},
                               m_min::Array{Float32,3},
                               m_max::Array{Float32,3},
                               d_obs::Vector{Float32},
                               grid_spec::GridSpec,
                               obs_x::Vector{Float32},
                               obs_y::Vector{Float32},
                               obs_z::Vector{Float32};
                               maxiters::Int=100,
                               λ_prior::Float32=1.0f-2,
                               λ_tv::Float32=1.0f-3,
                               λ_bounds::Float32=1.0f2,
                               optimizer=LBFGS(),
                               adtype=nothing,
                               tv_mode::Symbol=:anisotropic,
                               verbose::Bool=true)
    prob, params, history = build_inversion_problem(
        m0, m_min, m_max, d_obs, grid_spec, obs_x, obs_y, obs_z;
        λ_prior=λ_prior, λ_tv=λ_tv, λ_bounds=λ_bounds,
        tv_mode=tv_mode, adtype=adtype)

    callback = _make_inversion_callback(params, history; verbose=verbose)
    sol = solve(prob, optimizer;
                callback=callback,
                maxiters=maxiters,
                local_maxiters=maxiters)

    m_final = Array{Float32,3}(reshape(_as_f32(sol.u), params.m_shape))
    return (; m_final, history, sol)
end

end # module InversionSolver
