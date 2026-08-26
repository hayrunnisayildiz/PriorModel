"""
    MTPriorLoss

Supervised loss for 2-D MT resistivity prior training (fully synthetic ground truth).

```
L_total = λ_data · L1(predicted, true_resistivity)
        + λ_tv  · TotalVariation(predicted)
```

With `weighted=true` the L1 term is pixel-weighted toward anomalies:

```
w = 1 + λ_boundary · |target − median_spatial(target)|
L_data = mean(w ⊙ |pred − target|)
```

The median is per sample (batch item), computed on CPU so CuArray has no
`median` adjoint. TV is never weighted. Default `weighted=false` is the original
unweighted `mean(abs, pred − target)`.

All computations are in log10(ρ) space. Unlike the archived 3-D well-supervised
loss (`archive/3d_keivitsa/src/training/PriorTrainingLoss.jl`), there is no
partial borehole supervision — we have complete synthetic ground truth.
"""
module MTPriorLoss

using Statistics: mean, median
using ChainRulesCore: @non_differentiable

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
    _per_sample_host_ref(t) -> AbstractArray

Spatial median of each `(H, W)` slice, shaped `(1, 1, B)` on the same device as
`t`. Median runs on a CPU copy — CUDA `median` is not used.
"""
function _per_sample_host_ref(t::AbstractArray{T,3}) where T
    _, _, B = size(t)
    t_cpu = t isa Array ? t : Array(t)
    refs = Vector{T}(undef, B)
    for b in 1:B
        refs[b] = T(median(@view t_cpu[:, :, b]))
    end
    host = similar(t, T, 1, 1, B)
    copyto!(host, reshape(refs, 1, 1, B))
    return host
end

@non_differentiable _per_sample_host_ref(::AbstractArray)

"""
    _anomaly_l1_weights(t; λ_boundary) -> AbstractArray

Continuous anomaly weights `1 + λ_boundary * |t − host|` with `host` the
per-sample spatial median of `t`.
"""
function _anomaly_l1_weights(t::AbstractArray; λ_boundary::Float32=5.0f0)
    T = eltype(t)
    host = _per_sample_host_ref(t)
    return one(T) .+ T(λ_boundary) .* abs.(t .- host)
end

"""
    mt_prior_loss_terms(pred, target; λ_data, λ_tv, tv_mode, weighted, λ_boundary)
        -> NamedTuple

Compute loss components for a batch of 2-D log10-resistivity grids.

# Arguments
- `pred`:        `(nz, nx[, B])` predicted log10(ρ) from `MTResistivityUNet2D`
- `target`:      `(nz, nx[, B])` ground-truth log10(ρ)
- `λ_data`:      weight for L1 data term (default `1.0f0`)
- `λ_tv`:        weight for total variation (default `1.0f-3`)
- `weighted`:    if `true`, up-weight pixels that deviate from the per-sample
                 spatial median of `target` (default `false` — unweighted L1)
- `λ_boundary`:  scale of that deviation weight (default `5.0f0`); unused when
                 `weighted=false`

# Returns
`(; total, data, tv)` — all `Float32` scalars.
"""
function mt_prior_loss_terms(pred::AbstractArray, target::AbstractArray;
                             λ_data::Float32=1.0f0,
                             λ_tv::Float32=1.0f-3,
                             tv_mode::Symbol=:anisotropic,
                             weighted::Bool=false,
                             λ_boundary::Float32=5.0f0)
    p = _as_3d(pred)
    t = _as_3d(target)
    size(p) == size(t) || throw(DimensionMismatch(
        "pred $(size(p)) vs target $(size(t))"))

    T = eltype(p)

    L_data = if weighted
        w = _anomaly_l1_weights(t; λ_boundary=λ_boundary)
        mean(w .* abs.(p .- t))
    else
        mean(abs, p .- t)
    end

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
