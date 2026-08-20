"""
    MTInputStandardizer

Resample an arbitrary field MT survey onto the canonical U-Net input grid
`(n_stations, n_periods, n_channels)` used by the MT-only prior network.

Training pairs are generated on [`UNET_MESH`](@ref) (30 stations × 20 log-spaced
periods, `T ∈ [10⁻³, 10³]` s). [`DEFAULT_MESH`](@ref) is the COMMEMI solver mesh
(11 × 7) and is **not** the network input contract. Call [`standardize_mt_input`](@ref)
before `predict_prior` / `generate_prior` whenever the observed survey differs.

# Tensor layout
`(n_stations, n_periods, n_channels)` `Float32`, matching
[`MTMeshParams.MT_DATA_LAYOUT`](@ref):
- `[:, :, 1] = log10(ρ_a)` (TE `rho_xy` unless the caller stacked TM too)
- `[:, :, 2] = phase` (degrees)
- extra channels (TE/TM joint, 4-component) are interpolated independently

Interpolation is bilinear in `(station x [m], log10(period [s]))`. Queries
outside the observed range are clamped to the boundary value (edge repeat)
and emit a warning — that is an out-of-distribution site relative to the
training survey.
"""
module MTInputStandardizer

if !isdefined(@__MODULE__, :MTMeshParams)
    include(joinpath(@__DIR__, "MeshParams.jl"))
end
using .MTMeshParams: MeshParams, DEFAULT_MESH, UNET_MESH, n_periods, station_positions

export standardize_mt_input, pack_te_response
export CANONICAL_N_STATIONS, CANONICAL_PERIODS, CANONICAL_MESH
export canonical_station_positions

# Canonical survey ≡ UNET_MESH (not DEFAULT_MESH). Keep these aliases in lock-step.
const CANONICAL_MESH = UNET_MESH
const CANONICAL_N_STATIONS = UNET_MESH.n_stations
const CANONICAL_PERIODS = copy(UNET_MESH.periods)

CANONICAL_N_STATIONS == UNET_MESH.n_stations ||
    error("CANONICAL_N_STATIONS drifted from MeshParams.UNET_MESH")
length(CANONICAL_PERIODS) == n_periods(UNET_MESH) ||
    error("CANONICAL_PERIODS length drifted from MeshParams.UNET_MESH")
all(CANONICAL_PERIODS .== UNET_MESH.periods) ||
    error("CANONICAL_PERIODS values drifted from MeshParams.UNET_MESH")

canonical_station_positions(mp = CANONICAL_MESH) =
    station_positions(Int(mp.nx), Float64(mp.dx), Int(mp.n_stations))

# ─────────────────────────────────────────────────────────────────────────────
# 1-D / 2-D interpolation
# ─────────────────────────────────────────────────────────────────────────────

"""1-D linear interpolation; `x` must be sorted ascending. Clamps to endpoints."""
function _interp1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::Real)::Float64
    n = length(x)
    n == 0 && return NaN
    n == 1 && return Float64(y[1])
    if xq <= x[1]
        return Float64(y[1])
    elseif xq >= x[end]
        return Float64(y[end])
    end
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, n - 1)
    dx = x[i + 1] - x[i]
    t = abs(dx) < eps(Float64) ? 0.0 : (Float64(xq) - x[i]) / dx
    return (1.0 - t) * Float64(y[i]) + t * Float64(y[i + 1])
end

"""Nearest-neighbour sample on a sorted 1-D axis; clamps to endpoints."""
function _nearest1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::Real)::Float64
    n = length(x)
    n == 0 && return NaN
    n == 1 && return Float64(y[1])
    if xq <= x[1]
        return Float64(y[1])
    elseif xq >= x[end]
        return Float64(y[end])
    end
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, n - 1)
    return abs(Float64(xq) - x[i]) <= abs(x[i + 1] - Float64(xq)) ? Float64(y[i]) : Float64(y[i + 1])
end

"""
    _bilinear(xs, ys, Z, xq, yq) -> Float64

`Z` is `(length(xs), length(ys))`. `xs` and `ys` must be sorted ascending.
Queries outside the source rectangle are clamped (edge repeat).
"""
function _bilinear(xs::Vector{Float64}, ys::Vector{Float64},
                   Z::AbstractMatrix{<:Real}, xq::Float64, yq::Float64)::Float64
    nx, ny = length(xs), length(ys)
    (nx == 0 || ny == 0) && return NaN
    if nx == 1 && ny == 1
        return Float64(Z[1, 1])
    elseif nx == 1
        return _interp1(ys, @view(Z[1, :]), yq)
    elseif ny == 1
        return _interp1(xs, @view(Z[:, 1]), xq)
    end

    xqc = clamp(xq, xs[1], xs[end])
    yqc = clamp(yq, ys[1], ys[end])
    ix = clamp(searchsortedlast(xs, xqc), 1, nx - 1)
    iy = clamp(searchsortedlast(ys, yqc), 1, ny - 1)
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = abs(x1 - x0) < eps(Float64) ? 0.0 : (xqc - x0) / (x1 - x0)
    ty = abs(y1 - y0) < eps(Float64) ? 0.0 : (yqc - y0) / (y1 - y0)
    z00 = Float64(Z[ix, iy]);     z10 = Float64(Z[ix + 1, iy])
    z01 = Float64(Z[ix, iy + 1]); z11 = Float64(Z[ix + 1, iy + 1])
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 +
           (1 - tx) * ty * z01 + tx * ty * z11
end

function _nearest2(xs::Vector{Float64}, ys::Vector{Float64},
                   Z::AbstractMatrix{<:Real}, xq::Float64, yq::Float64)::Float64
    nx, ny = length(xs), length(ys)
    (nx == 0 || ny == 0) && return NaN
    xqc = clamp(xq, xs[1], xs[end])
    yqc = clamp(yq, ys[1], ys[end])
    ix = if nx == 1
        1
    else
        i = clamp(searchsortedlast(xs, xqc), 1, nx - 1)
        abs(xqc - xs[i]) <= abs(xs[i + 1] - xqc) ? i : i + 1
    end
    iy = if ny == 1
        1
    else
        j = clamp(searchsortedlast(ys, yqc), 1, ny - 1)
        abs(yqc - ys[j]) <= abs(ys[j + 1] - yqc) ? j : j + 1
    end
    return Float64(Z[ix, iy])
end

function _outside_range(src::AbstractVector{<:Real}, tgt::AbstractVector{<:Real})::Bool
    isempty(src) && return true
    smin, smax = extrema(src)
    tmin, tmax = extrema(tgt)
    return tmin < smin - 1e-12 || tmax > smax + 1e-12
end

function _target_survey(mp)::Tuple{Vector{Float64},Vector{Float64}}
    tgt_x = station_positions(Int(mp.nx), Float64(mp.dx), Int(mp.n_stations))
    tgt_T = Float64.(collect(mp.periods))
    length(tgt_x) == Int(mp.n_stations) ||
        error("canonical station count mismatch: $(length(tgt_x)) vs $(mp.n_stations)")
    return tgt_x, tgt_T
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

"""
    pack_te_response(rho_xy, phase_xy) -> Array{Float32,3}

Convert solver-layout TE arrays `(n_periods, n_stations)` into the U-Net
layout `(n_stations, n_periods, 2)` with `[log10(ρ_a), phase_deg]`.
"""
function pack_te_response(rho_xy::AbstractMatrix, phase_xy::AbstractMatrix)::Array{Float32,3}
    size(rho_xy) == size(phase_xy) ||
        error("rho_xy $(size(rho_xy)) and phase_xy $(size(phase_xy)) differ")
    nP, nS = size(rho_xy)
    out = Array{Float32,3}(undef, nS, nP, 2)
    @inbounds for is in 1:nS, ip in 1:nP
        ρ = Float64(rho_xy[ip, is])
        out[is, ip, 1] = ρ > 0 ? Float32(log10(ρ)) : NaN32
        out[is, ip, 2] = Float32(phase_xy[ip, is])
    end
    return out
end

"""
    standardize_mt_input(raw_stations, raw_periods, raw_data;
                         mp=CANONICAL_MESH, method=:bilinear) -> Array{Float32,3}

Bilinear (or nearest) resampling of a field MT tensor onto the canonical
survey stored in `mp` (default [`UNET_MESH`](@ref) / `CANONICAL_*`).

# Arguments
- `raw_stations`: station profile coordinates, metres (length `size(raw_data, 1)`)
- `raw_periods`: periods, seconds (length `size(raw_data, 2)`)
- `raw_data`: `(n_stations, n_periods, n_channels)` — same channel order as
  training (`log10 ρ_a`, phase deg, optionally TM)

# Keyword arguments
- `mp`: mesh-like object with `nx`, `dx`, `n_stations`, `periods` (duck-typed
  so checkpoint `MeshParams` from another include works)
- `method`: `:bilinear` (default) or `:nearest`

Station interpolation is **location-based** (true x in metres), never index
mapping. Period interpolation is linear in `log10(T)`. Out-of-range queries
repeat the boundary value and log a warning.
"""
function standardize_mt_input(raw_stations::AbstractVector,
                              raw_periods::AbstractVector,
                              raw_data::AbstractArray{<:Real,3};
                              mp=CANONICAL_MESH,
                              method::Symbol=:bilinear)::Array{Float32,3}
    method in (:bilinear, :nearest) ||
        throw(ArgumentError("method must be :bilinear or :nearest, got $(repr(method))"))

    nS_raw, nP_raw, nC = size(raw_data)
    length(raw_stations) == nS_raw ||
        error("raw_stations length $(length(raw_stations)) ≠ size(raw_data, 1)=$nS_raw")
    length(raw_periods) == nP_raw ||
        error("raw_periods length $(length(raw_periods)) ≠ size(raw_data, 2)=$nP_raw")
    nC >= 1 || error("raw_data has no channels")
    all(>(0.0), raw_periods) || error("all raw_periods must be positive")

    src_x = Float64.(collect(raw_stations))
    src_T = Float64.(collect(raw_periods))
    tgt_x, tgt_T = _target_survey(mp)
    all(>(0.0), tgt_T) || error("all target periods must be positive")

    src_logT = log10.(src_T)
    tgt_logT = log10.(tgt_T)

    x_ood = _outside_range(src_x, tgt_x)
    T_query_clamp = _outside_range(src_logT, tgt_logT)  # U-Net T outside observations
    T_src_ood = _outside_range(tgt_logT, src_logT)      # observations outside training T
    if x_ood || T_query_clamp
        @warn "canonical survey extends outside the observed range; clamping query points to edge values" x_ood=x_ood T_query_clamp=T_query_clamp T_src_ood=T_src_ood src_x=(minimum(src_x), maximum(src_x)) tgt_x=(minimum(tgt_x), maximum(tgt_x)) src_T=(minimum(src_T), maximum(src_T)) tgt_T=(minimum(tgt_T), maximum(tgt_T)) method=method
    end

    x_perm = sortperm(src_x)
    T_perm = sortperm(src_logT)
    xs = src_x[x_perm]
    logT = src_logT[T_perm]

    nS, nP = length(tgt_x), length(tgt_T)
    out = Array{Float32,3}(undef, nS, nP, nC)
    sample = method === :bilinear ? _bilinear : _nearest2

    @inbounds for ic in 1:nC
        Z = Matrix{Float64}(undef, nS_raw, nP_raw)
        for is in 1:nS_raw, ip in 1:nP_raw
            Z[is, ip] = Float64(raw_data[x_perm[is], T_perm[ip], ic])
        end
        for is in 1:nS, ip in 1:nP
            out[is, ip, ic] = Float32(sample(xs, logT, Z, tgt_x[is], tgt_logT[ip]))
        end
    end
    out[.!isfinite.(out)] .= 0.0f0
    return out
end

"""Duck-typed `MTGeophysics.load_data2d` result: `.receivers`, `.periods`, `.rho_xy`, `.phase_xy`."""
function standardize_mt_input(data;
                              mp=CANONICAL_MESH,
                              method::Symbol=:bilinear)::Array{Float32,3}
    raw = pack_te_response(data.rho_xy, data.phase_xy)
    return standardize_mt_input(Float64.(collect(data.receivers)),
                                Float64.(collect(data.periods)),
                                raw; mp=mp, method=method)
end

end # module MTInputStandardizer
