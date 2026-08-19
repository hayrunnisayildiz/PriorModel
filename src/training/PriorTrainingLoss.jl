"""
    PriorTrainingLoss

Self-constrained prior training objective for [`PriorUNet3DLayers.UnifiedPriorUNet3D`](@ref):

```
L_total = λ_well L_sondaj + λ_smooth L_TV + λ_bounds L_bounds
```

- **Well loss** — log10 masked MSE at borehole `LUO_R` voxels (ohm·m)
- **TV loss** — anisotropic 3-D total variation on ``\\log_{10}\\rho``
- **Bounds** — hinge on ``m_{min} \\le m_0 \\le m_{max}``

All terms return `Float32` scalars for Zygote / Optimisers.jl.
"""
module PriorTrainingLoss

using Statistics: mean

if !isdefined(@__MODULE__, :Losses)
    include(joinpath(@__DIR__, "..", "neural_prior", "Losses.jl"))
end
using .Losses: total_variation

export PhysicsContext, PatchTrainContext, SupervisionChannels
export volume_to_density, volume_to_susceptibility
export well_loss, forward_physics_loss, combined_loss, combined_loss_terms
export patch_combined_loss_terms, bounds_penalty
export log10_resistivity

const KG_M3_TO_G_CM3::Float32 = 0.001f0
const G_CM3_TO_KG_M3::Float32 = 1000.0f0
const LOG10_RHO_EPS::Float32 = 1.0f-3   # ohm·m floor before log10

# ─────────────────────────────────────────────────────────────────────────────
# Physics context (precomputed observation / kernel data)
# ─────────────────────────────────────────────────────────────────────────────

"""
    PhysicsContext

Precomputed forward-operator matrices and surface observations for one grid.

# Fields
- `G_grav`, `G_mag`: `(n_obs, n_voxels)` sensitivity matrices
- `d_obs_grav`: observed gravity anomaly, mGal
- `d_obs_mag`: observed TMI **anomaly** (regional removed), nT
- `rho_background`: reference density for contrast → absolute, ``g/cm^3``
- `chi_scale`: SI susceptibility per ``g/cm^3`` density contrast (empirical)
"""
struct PhysicsContext
    G_grav::Matrix{Float32}
    G_mag::Matrix{Float32}
    d_obs_grav::Vector{Float32}
    d_obs_mag::Vector{Float32}
    rho_background::Float32
    chi_scale::Float32
end

"""Absolute density ``g/cm^3`` from U-Net channel 1 (density contrast)."""
function volume_to_density(vol::AbstractArray, rho_bg::Float32)
    δρ = vol[:, :, :, 1:1, :]                       # (X, Y, Z, 1, B)
    return rho_bg .+ δρ
end

"""
SI susceptibility volume from density contrast (empirical petrophysical link).

``\\chi = \\mathrm{chi\\_scale} \\cdot \\Delta\\rho`` with ``\\Delta\\rho`` in ``g/cm^3``.
"""
function volume_to_susceptibility(vol::AbstractArray, chi_scale::Float32)
    δρ = vol[:, :, :, 1:1, :]
    return chi_scale .* δρ
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss terms
# ─────────────────────────────────────────────────────────────────────────────

"""
    well_loss(pred, target, mask) -> Float32

Normalised masked MSE (sondaj uyumu):

``L = \\| M \\odot (\\hat{m} - m) \\|^2 / \\sum M``

`pred` and `target` are broadcastable `(X, Y, Z, C, B)`.  Only cells with
`mask > 0` **and** finite `target` contribute.
"""
function well_loss(pred::AbstractArray{T}, target::AbstractArray, mask::AbstractArray)::Float32 where {T}
    p = _as_5d(pred)
    t = _as_5d(target)
    m = _as_5d(mask)
    size(p)[1:3] == size(t)[1:3] == size(m)[1:3] ||
        throw(DimensionMismatch("spatial size mismatch in well_loss"))

    active = (m .> 0) .& isfinite.(t)
    n = count(active)
    n == 0 && return zero(T)

    r = p .- t
    s = sum(ifelse.(active, r .* r, zero(T)))
    return s / T(n)
end

"""
    forward_physics_loss(density_gcm3, chi, ctx) -> Float32

``L = \\|G_g \\rho - d_{grav}\\|^2 + \\|G_m \\chi - d_{mag}\\|^2``
"""
function forward_physics_loss(density_gcm3::Array{Float32,3},
                              chi::Array{Float32,3},
                              ctx::PhysicsContext)::Float32
    size(density_gcm3) == size(chi) ||
        throw(DimensionMismatch("density / susceptibility shape mismatch"))
    g_pred = ctx.G_grav * vec(density_gcm3)
    m_pred = ctx.G_mag * vec(chi)
    rg = g_pred .- ctx.d_obs_grav
    rm = m_pred .- ctx.d_obs_mag
    return sum(rg .* rg) + sum(rm .* rm)
end

"""
    combined_loss_terms(vol, target, mask, ctx; λ_well, λ_phys, λ_smooth, tv_mode)

Returns `(; total, well, forward, tv)` NamedTuple.
"""
function combined_loss_terms(vol::AbstractArray,
                             target::AbstractArray,
                             mask::AbstractArray,
                             ctx::PhysicsContext;
                             λ_well::Float32=1.0f0,
                             λ_phys::Float32=1.0f-2,
                             λ_smooth::Float32=1.0f-3,
                             tv_mode::Symbol=:anisotropic)
# Well supervision: predicted resistivity (ohm·m) vs LUO_R / density leftover
    ρ_gcm3 = volume_to_density(vol, ctx.rho_background)
    ρ_kgm3 = ρ_gcm3 .* G_CM3_TO_KG_M3
    ρ_ohm_pred = vol[:, :, :, 1:1, :]
    tgt = _as_5d(target)
    m = _as_5d(mask)
    if size(tgt, 4) >= 2
        # Y layout: [resistivity, M_sondaj] or legacy [density, resistivity, mask]
        if size(tgt, 4) == 2
            log_pred = log10_resistivity(ρ_ohm_pred)
            log_tgt = log10_resistivity(tgt[:, :, :, 1:1, :])
            L_well = well_loss(log_pred, log_tgt, m)
        else
            pred_phys = cat(ρ_kgm3, ρ_ohm_pred; dims=4)
            L_well = well_loss(pred_phys, tgt[:, :, :, 1:2, :], m)
        end
    else
        log_pred = log10_resistivity(ρ_ohm_pred)
        log_tgt = log10_resistivity(tgt)
        L_well = well_loss(log_pred, log_tgt, m)
    end

    # TV on log-resistivity (geological continuity)
    L_tv = total_variation(log10_resistivity(ρ_ohm_pred); mode=tv_mode)

    T = eltype(L_well)
    # Gravity/mag forward no longer applies to a resistivity prior.
    L_fwd = zero(T)
    total = T(λ_well) * L_well + T(λ_smooth) * L_tv
    return (; total, well=L_well, forward=L_fwd, tv=L_tv)
end

"""Scalar alias for [`combined_loss_terms`](@ref)."""
combined_loss(args...; kwargs...) = combined_loss_terms(args...; kwargs...).total

# ─────────────────────────────────────────────────────────────────────────────
# Patch-based training (UnifiedPriorUNet3D + PatchLoader)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SupervisionChannels

Channel indices (1-based) inside the **PatchLoader** batch `(Px, Py, Pz, C_in, B)`.
The last loader channel is always the validity mask; well indices refer to
the underlying fusion / `Y` channels before the appended mask.
"""
struct SupervisionChannels
    resistivity::Int
    well_mask::Int
    resistivity_unit::Symbol
end

"""
    PatchTrainContext

Channel map for patch-based resistivity prior training.
"""
struct PatchTrainContext
    channels::SupervisionChannels
end

"""Zygote-safe ``\\log_{10}\\rho`` with a positive ohm·m floor."""
function log10_resistivity(ρ::AbstractArray{T}) where {T}
    ε = T(LOG10_RHO_EPS)
    return log10.(ifelse.(ρ .> ε, ρ, ε))
end

"""Hinge penalty enforcing ``m_{min} \\le m_0 \\le m_{max}``."""
function bounds_penalty(m0::AbstractArray, m_min::AbstractArray, m_max::AbstractArray)
    ρ = _as_5d(m0)
    lo = _as_5d(m_min)
    hi = _as_5d(m_max)
    T = eltype(ρ)
    below = ifelse.(ρ .< lo, lo .- ρ, zero(T))
    above = ifelse.(ρ .> hi, ρ .- hi, zero(T))
    return mean(below .+ above)
end

"""Convert stored resistivity to ohm·m (pass-through; kept for unit checks)."""
function _resistivity_ohm(raw::AbstractArray, unit::Symbol)
    if unit === :ohm_m || unit === :ohm_meter
        return raw
    else
        throw(ArgumentError("resistivity_unit must be :ohm_m, got $(repr(unit))"))
    end
end

"""
    patch_combined_loss_terms(vol, x, origins, ctx; λ_well, λ_tv, λ_bounds)

Combined patch objective for [`UnifiedPriorUNet3D`](@ref) output
`(Px, Py, Pz, 3, B)` — channels ``m_0, m_{min}, m_{max}`` in ohm·m.
Well mismatch and TV are evaluated in ``\\log_{10}`` resistivity.
"""
function patch_combined_loss_terms(vol::AbstractArray,
                                   x::AbstractArray,
                                   origins::Vector{NTuple{3,Int}},
                                   ctx::PatchTrainContext;
                                   λ_well::Float32=1.0f0,
                                   λ_grav::Float32=0.0f0,
                                   λ_tv::Float32=1.0f-3,
                                   λ_bounds::Float32=1.0f-4,
                                   tv_mode::Symbol=:anisotropic)
    ch = ctx.channels
    x5 = _as_5d(x)
    vol5 = _as_5d(vol)

    m0 = vol5[:, :, :, 1:1, :]
    m_min = vol5[:, :, :, 2:2, :]
    m_max = vol5[:, :, :, 3:3, :]

    raw_ρ = x5[:, :, :, ch.resistivity:ch.resistivity, :]
    target = _resistivity_ohm(raw_ρ, ch.resistivity_unit)
    well_mask = x5[:, :, :, ch.well_mask:ch.well_mask, :]
    # After PatchLoader, missing resistivity is 0; positive LUO_R marks well voxels.
    active = (well_mask .> 0) .& (target .> 0) .& isfinite.(target)

    log_pred = log10_resistivity(m0)
    log_tgt = log10_resistivity(target)
    L_well = well_loss(log_pred, log_tgt, active)

    L_tv = total_variation(log_pred; mode=tv_mode)
    L_bounds = bounds_penalty(m0, m_min, m_max)

    T = eltype(L_well)
    total = T(λ_well) * L_well + T(λ_tv) * L_tv + T(λ_bounds) * L_bounds
    return (; total, well=L_well, gravity=zero(T), tv=L_tv, bounds=L_bounds)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function _as_5d(x::AbstractArray)
    nd = ndims(x)
    nd == 5 && return x
    nd == 4 && return reshape(x, size(x)..., 1)
    nd == 3 && return reshape(x, size(x)..., 1, 1)
    throw(ArgumentError("expected 3–5D array, got ndims=$nd"))
end

end # module PriorTrainingLoss
