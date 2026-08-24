"""
    MTPriorLoss

Supervised loss for 2-D MT resistivity prior training (fully synthetic ground truth).

```
L_total = λ_data · L1(predicted, true_resistivity)
        + λ_tv  · TotalVariation(predicted)
```

All computations are in log10(ρ) space. Unlike the archived 3-D well-supervised
loss (`archive/3d_keivitsa/src/training/PriorTrainingLoss.jl`), there is no
partial borehole supervision — we have complete synthetic ground truth.
"""
module MTPriorLoss

using Statistics: mean

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

    L_tv = _total_variation_2d(p; mode=tv_mode)

    total = T(λ_data) * L_data + T(λ_tv) * L_tv
    return (; total, data=L_data, tv=L_tv)
end

"""2-D anisotropic/isotropic total variation on `(H, W, B)` arrays."""
function _total_variation_2d(m::AbstractArray; mode::Symbol=:anisotropic, ε::Float32=1.0f-8)
    v = _as_3d(m)
    dh = v[2:end, :, :] .- v[1:end-1, :, :]
    dw = v[:, 2:end, :] .- v[:, 1:end-1, :]
    if mode === :anisotropic
        return mean(abs, dh) + mean(abs, dw)
    elseif mode === :isotropic
        T = eltype(v)
        dh_i = dh[:, 1:end-1, :]
        dw_i = dw[1:end-1, :, :]
        return mean(sqrt.(dh_i .* dh_i .+ dw_i .* dw_i .+ T(ε)))
    else
        throw(ArgumentError("mode must be :anisotropic or :isotropic, got $(repr(mode))"))
    end
end

"""Scalar alias."""
mt_prior_loss(args...; kw...) = mt_prior_loss_terms(args...; kw...).total

end # module MTPriorLoss
