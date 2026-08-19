"""
    MTPriorLoss

Supervised loss for 2-D MT resistivity prior training (fully synthetic ground truth).

```
L_total = λ_data · L1(predicted, true_resistivity)
        + λ_tv  · TotalVariation(predicted)
```

All computations are in log10(ρ) space.  Unlike [`PriorTrainingLoss`](@ref),
there is no partial well supervision — we have complete synthetic ground truth.
"""
module MTPriorLoss

using Statistics: mean

if !isdefined(@__MODULE__, :Losses)
    include(joinpath(@__DIR__, "..", "neural_prior", "Losses.jl"))
end
using .Losses: total_variation

export mt_prior_loss, mt_prior_loss_terms

"""
    _as_3d(x) -> Array{T,3}

Promote `(H, W)` to `(H, W, 1)` for batch-uniform handling.
"""
function _as_3d(x::AbstractArray)
    nd = ndims(x)
    nd == 3 && return x
    nd == 2 && return reshape(x, size(x)..., 1)
    throw(ArgumentError("expected 2-D or 3-D array, got ndims=$nd"))
end

"""
    mt_prior_loss_terms(pred, target; λ_data, λ_tv, tv_mode) -> NamedTuple

Compute loss components for a batch of 2-D log10-resistivity grids.

# Arguments
- `pred`:   `(nz, nx[, B])` predicted log10(ρ) from `MTResistivityUNet2D`
- `target`: `(nz, nx[, B])` ground-truth log10(ρ)
- `λ_data`: weight for L1 data term (default `1.0f0`)
- `λ_tv`:   weight for total variation (default `1.0f-3`)

# Returns
`(; total, data, tv)` — all `Float32` scalars.
"""
function mt_prior_loss_terms(pred::AbstractArray, target::AbstractArray;
                             λ_data::Float32=1.0f0,
                             λ_tv::Float32=1.0f-3,
                             tv_mode::Symbol=:anisotropic)
    p = _as_3d(pred)
    t = _as_3d(target)
    size(p) == size(t) || throw(DimensionMismatch(
        "pred $(size(p)) vs target $(size(t))"))

    T = eltype(p)

    L_data = mean(abs, p .- t)

    # TV: promote (nz, nx, B) → (nz, nx, 1, 1, B) for the 3-D TV function
    p5 = reshape(p, size(p, 1), size(p, 2), 1, 1, size(p, 3))
    L_tv = total_variation(p5; mode=tv_mode)

    total = T(λ_data) * L_data + T(λ_tv) * L_tv
    return (; total, data=L_data, tv=L_tv)
end

"""Scalar alias."""
mt_prior_loss(args...; kw...) = mt_prior_loss_terms(args...; kw...).total

end # module MTPriorLoss
