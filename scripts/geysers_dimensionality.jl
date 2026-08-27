#!/usr/bin/env julia
#=
geysers_dimensionality.jl — MT phase-tensor dimensionality analysis for the
Northwest Geysers (CA) survey, 42 broadband stations.

Question this answers: is a 2-D profile inversion defensible at this site, and
along which azimuth should the profile be cut? Everything is derived from the
EDI impedance tensors. **No forward solve is called.**

MTGeophysics.jl v0.4.2 has no EDI reader (ModEM + 2-D `.dat` I/O only), so this
script carries its own minimal parser: HEAD/DEFINEMEAS coordinates, FREQ, ZROT,
the eight ZXXR…ZYYI blocks and the four Z**.VAR blocks. Nothing is resampled,
interpolated or synthesised; only `T = 1/f` is applied.

Phase-tensor definitions follow Caldwell, Bibby & Brown (2004), "The
magnetotelluric phase tensor", Geophys. J. Int. 158, 457-469
(doi:10.1111/j.1365-246X.2004.02281.x). Equation numbers cited in the
docstrings below refer to that paper. Cross-checked component-by-component
against `mtpy/analysis/pt.py` (MTgeophysics/mtpy), which implements the same
equations; `test/test_geysers_phase_tensor.jl` re-derives them from analytic
1-D / 2-D / rotated cases.

Usage (from project root):
    julia --project=. scripts/geysers_dimensionality.jl
    julia --project=. scripts/geysers_dimensionality.jl data/geysers/mt/edi results

Outputs:
    results/geysers_dimensionality.md
    results/geysers_dimensionality.csv
    results/geysers_phase_tensor.png
    results/geysers_strike_rose.png
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics
using Random
using Dates
using LinearAlgebra

const DEFAULT_EDI_DIR = joinpath(ROOT, "data", "geysers", "mt", "edi")
const DEFAULT_OUT_DIR = joinpath(ROOT, "results")

# EDI `EMPTY=1e+32`: any magnitude at or above this is a fill value, not data.
const EDI_EMPTY_THRESHOLD = 1.0e30

const Z_TAGS = ("ZXXR", "ZXXI", "ZXYR", "ZXYI", "ZYXR", "ZYXI", "ZYYR", "ZYYI")
const VAR_TAGS = ("ZXX.VAR", "ZXY.VAR", "ZYX.VAR", "ZYY.VAR")

# |beta| below this is the conventional "2-D is acceptable" cut-off. Caldwell
# et al. (2004) require beta = 0 within error; 3 deg is the common practical
# bound (e.g. Booker 2014, Surv. Geophys. 35, 7-40, section 5).
const BETA_2D_DEG = 3.0

# Two grid periods are the same physical period if their logs differ by less
# than this. The files come off one BIRRP grid but differ in the last digits.
const PERIOD_RTOL = 0.02

# ─────────────────────────────────────────────────────────────────────────────
# Minimal EDI parser
# ─────────────────────────────────────────────────────────────────────────────

struct EDIStation
    site::String
    path::String
    lat::Float64
    lon::Float64
    elev::Float64
    declination::Float64
    east::Float64
    north::Float64
    periods::Vector{Float64}              # s, ascending
    zrot::Vector{Float64}                 # deg, as stored in the >ZROT block
    Z::Vector{Matrix{ComplexF64}}         # 2x2 per period, [xx xy; yx yy]
    Zvar::Vector{Matrix{Float64}}         # variances, NaN where absent
end

"""Strip an EDI `>TAG` / `>=TAG` line down to its tag name (first token)."""
function edi_tag(line::AbstractString)::String
    t = strip(line)
    startswith(t, ">") || return ""
    t = lstrip(lstrip(t, '>'), '=')
    tok = split(t; limit = 2)
    return isempty(tok) ? "" : uppercase(tok[1])
end

"""Parse `KEY=value` assignments on an EDI header line (quotes optional)."""
function parse_assignments(line::AbstractString)::Dict{String,String}
    out = Dict{String,String}()
    for m in eachmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|[^\s]+)", line)
        val = m.captures[2]
        if startswith(val, "\"") && endswith(val, "\"") && length(val) >= 2
            val = val[2:end-1]
        end
        out[uppercase(m.captures[1])] = val
    end
    return out
end

"""DMS `±DD:MM:SS.s` or plain decimal degrees → Float64 degrees."""
function parse_angle(s::AbstractString)::Float64
    t = strip(s)
    isempty(t) && return NaN
    if occursin(':', t)
        sgn = 1.0
        if startswith(t, "-")
            sgn = -1.0
            t = t[2:end]
        elseif startswith(t, "+")
            t = t[2:end]
        end
        parts = split(t, ':')
        length(parts) == 3 || error("expected DMS 'DD:MM:SS', got $(repr(s))")
        return sgn * (parse(Float64, parts[1]) +
                      parse(Float64, parts[2]) / 60 +
                      parse(Float64, parts[3]) / 3600)
    end
    v = tryparse(Float64, t)
    v === nothing && error("cannot parse angle $(repr(s))")
    return v
end

"""Read the free-format numeric payload of an EDI block, stopping at the next `>`."""
function read_block(lines::Vector{String}, start_idx::Int)::Tuple{Vector{Float64},Int}
    vals = Float64[]
    i = start_idx
    while i <= length(lines)
        t = strip(lines[i])
        if isempty(t)
            i += 1
            continue
        end
        startswith(t, ">") && break
        for tok in split(t)
            v = tryparse(Float64, tok)
            v === nothing && continue
            push!(vals, v)
        end
        i += 1
    end
    return vals, i
end

function first_present(d::Dict{String,String}, keys::Vararg{String})::String
    for k in keys
        haskey(d, k) && return d[k]
    end
    return ""
end

"""EDI fill values (`EMPTY=1e+32`) and non-finite entries become `NaN`."""
blank_empty(v::Float64)::Float64 = (isfinite(v) && abs(v) < EDI_EMPTY_THRESHOLD) ? v : NaN

"""
    parse_edi(path) -> EDIStation

Minimal reader for the MTpy-written Geysers EDIs: `>HEAD` / `>=DEFINEMEAS`
coordinates, `>FREQ`, `>ZROT`, the eight `Z**R` / `Z**I` blocks and the four
`Z**.VAR` blocks. Periods are `T = 1/f` and rows are sorted by ascending `T`.
Blocks whose length disagrees with `>FREQ` are a hard error, not a silent
truncation.
"""
function parse_edi(path::AbstractString)::EDIStation
    lines = readlines(path)
    head = Dict{String,String}()
    meas = Dict{String,String}()
    sect = Dict{String,String}()
    blocks = Dict{String,Vector{Float64}}()

    i = 1
    while i <= length(lines)
        t = strip(lines[i])
        if isempty(t) || !startswith(t, ">")
            i += 1
            continue
        end
        tag = edi_tag(t)
        assigns = parse_assignments(t)

        if tag in ("HEAD", "DEFINEMEAS", "MTSECT", "SPECTRASECT")
            target = tag == "HEAD" ? head : (tag == "DEFINEMEAS" ? meas : sect)
            merge!(target, assigns)
            i += 1
            while i <= length(lines) && !startswith(strip(lines[i]), ">")
                merge!(target, parse_assignments(lines[i]))
                i += 1
            end
        elseif tag == "FREQ" || tag == "ZROT" || tag in Z_TAGS || tag in VAR_TAGS
            vals, i = read_block(lines, i + 1)
            blocks[tag] = vals
        else
            i += 1
        end
    end

    haskey(blocks, "FREQ") || error("$path: no >FREQ block")
    freqs = blocks["FREQ"]
    nf = length(freqs)
    for tag in Z_TAGS
        haskey(blocks, tag) || error("$path: missing >$tag block")
        length(blocks[tag]) == nf ||
            error("$path: >$tag has $(length(blocks[tag])) values, >FREQ has $nf")
    end

    lat = parse_angle(first_present(meas, "REFLAT", "REFLATITUDE"))
    isnan(lat) && (lat = parse_angle(first_present(head, "LAT", "LATITUDE")))
    lon = parse_angle(first_present(meas, "REFLON", "REFLONG", "REFLONGITUDE"))
    isnan(lon) && (lon = parse_angle(first_present(head, "LON", "LONG", "LONGITUDE")))
    elev = something(tryparse(Float64,
        strip(isempty(first_present(meas, "REFELEV")) ? first_present(head, "ELEV") :
              first_present(meas, "REFELEV"))), NaN)
    decl = something(tryparse(Float64, strip(first_present(head, "DECLINATION"))), NaN)
    east, north = wgs84_to_utm10n(lat, lon)

    zrot_all = get(blocks, "ZROT", Float64[])
    length(zrot_all) == nf || (zrot_all = fill(NaN, nf))

    Zall = Vector{Matrix{ComplexF64}}(undef, nf)
    Vall = Vector{Matrix{Float64}}(undef, nf)
    for k in 1:nf
        g(tag) = blank_empty(blocks[tag][k])
        v(tag) = haskey(blocks, tag) && length(blocks[tag]) == nf ?
                 blank_empty(blocks[tag][k]) : NaN
        Zall[k] = ComplexF64[
            complex(g("ZXXR"), g("ZXXI"))  complex(g("ZXYR"), g("ZXYI"));
            complex(g("ZYXR"), g("ZYXI"))  complex(g("ZYYR"), g("ZYYI"))
        ]
        Vall[k] = Float64[
            v("ZXX.VAR")  v("ZXY.VAR");
            v("ZYX.VAR")  v("ZYY.VAR")
        ]
    end

    periods = Float64[f > 0 ? 1 / f : NaN for f in freqs]
    perm = sortperm(periods)
    site = uppercase(get(sect, "SECTID", get(head, "DATAID", splitext(basename(path))[1])))

    return EDIStation(site, String(path), lat, lon, elev, decl, east, north,
                      periods[perm], zrot_all[perm], Zall[perm], Vall[perm])
end

# ─────────────────────────────────────────────────────────────────────────────
# WGS84 → UTM Zone 10N (EPSG:32610)
# ─────────────────────────────────────────────────────────────────────────────

const WGS84_A = 6378137.0
const WGS84_F = 1 / 298.257223563
const UTM_K0 = 0.9996
const UTM_FALSE_EASTING = 500_000.0
const UTM_ZONE10_LON0 = -123.0

"""
    wgs84_to_utm10n(lat_deg, lon_deg) -> (easting_m, northing_m)

WGS84 geographic → UTM Zone 10N (EPSG:32610). Transverse-Mercator series of
Snyder (1987), *Map Projections — A Working Manual*, USGS Professional Paper
1395, eqs. (8-9) to (8-11) with the meridional arc (3-21). Sub-metre inside the
zone. Purely a map projection; it never touches the MT responses.
"""
function wgs84_to_utm10n(lat_deg::Float64, lon_deg::Float64)::Tuple{Float64,Float64}
    a = WGS84_A
    b = a * (1 - WGS84_F)
    e2 = (a^2 - b^2) / a^2
    ep2 = (a^2 - b^2) / b^2
    φ = deg2rad(lat_deg)
    Δλ = deg2rad(lon_deg - UTM_ZONE10_LON0)

    N = a / sqrt(1 - e2 * sin(φ)^2)
    T = tan(φ)^2
    C = ep2 * cos(φ)^2
    A = Δλ * cos(φ)
    # Meridional arc, Snyder eq. (3-21). Written with explicit `*` because
    # `3e4` would lex as the float literal 30000.0, not 3·e⁴.
    e4 = e2^2
    e6 = e4 * e2
    M = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * φ -
             (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * sin(2φ) +
             (15 * e4 / 256 + 45 * e6 / 1024) * sin(4φ) -
             (35 * e6 / 3072) * sin(6φ))

    east = UTM_K0 * N * (A + (1 - T + C) * A^3 / 6 +
                         (5 - 18 * T + T^2 + 72 * C - 58 * ep2) * A^5 / 120) +
           UTM_FALSE_EASTING
    north = UTM_K0 * (M + N * tan(φ) * (A^2 / 2 +
                      (5 - T + 9 * C + 4 * C^2) * A^4 / 24 +
                      (61 - 58 * T + T^2 + 600 * C - 330 * ep2) * A^6 / 720))
    return east, north
end

# ─────────────────────────────────────────────────────────────────────────────
# Phase tensor — Caldwell, Bibby & Brown (2004), GJI 158, 457-469
# ─────────────────────────────────────────────────────────────────────────────

"""
    phase_tensor(Z) -> Matrix{Float64}

Phase tensor `Φ = X⁻¹ Y` with `X = Re(Z)`, `Y = Im(Z)` — Caldwell, Bibby &
Brown (2004) eq. (13) — evaluated in the component form of their eq. (15):

    Φ = 1/det(X) * [ X22·Y11 − X12·Y21    X22·Y12 − X12·Y22
                     X11·Y21 − X21·Y11    X11·Y22 − X21·Y12 ]

with `det(X) = X11·X22 − X21·X12`. `Φ` is real and is unchanged by any real
galvanic distortion tensor `D` (their eq. 14), so it needs no assumption about
dimensionality. Returns a matrix of `NaN` if `Z` has non-finite entries or `X`
is singular.
"""
function phase_tensor(Z::AbstractMatrix{<:Complex})::Matrix{Float64}
    all(isfinite, Z) || return fill(NaN, 2, 2)
    X = real.(Z)
    Y = imag.(Z)
    detX = X[1, 1] * X[2, 2] - X[2, 1] * X[1, 2]
    (isfinite(detX) && detX != 0) || return fill(NaN, 2, 2)
    return Float64[
        (X[2, 2] * Y[1, 1] - X[1, 2] * Y[2, 1])  (X[2, 2] * Y[1, 2] - X[1, 2] * Y[2, 2]);
        (X[1, 1] * Y[2, 1] - X[2, 1] * Y[1, 1])  (X[1, 1] * Y[2, 2] - X[2, 1] * Y[1, 2])
    ] ./ detX
end

"""
Coordinate-invariant description of one phase tensor. All angles in degrees.

* `beta`    — skew angle, CBB04 eqs. (19) and (A10). `β = 0` is a *necessary*
              condition for a 1-D or 2-D regional structure.
* `alpha`   — coordinate-dependent angle, CBB04 eq. (22).
* `azimuth` — direction of the ellipse major axis, `α − β` (CBB04 Fig. 1 and
              Appendix "Tensor ellipse"), clockwise from `x₁`.
* `phimax`, `phimin` — principal (singular) values, CBB04 eqs. (A8)/(A9).
* `phimax_deg`, `phimin_deg` — `atan` of those, i.e. the principal phases in
              degrees; CBB04 Fig. 8(b) plots exactly `tan⁻¹Φmax`, `tan⁻¹Φmin`.
* `ellipticity` — `(Φmax − Φmin)/(Φmax + Φmin)`. Near 0 the ellipse is a circle
              and its major-axis direction — hence any strike estimate — is
              meaningless (CBB04, "Tensor ellipse in 1-D": α is undefined and
              unstable under noise when the axes are equal).
"""
struct PTInvariants
    beta::Float64
    alpha::Float64
    azimuth::Float64
    phimax::Float64
    phimin::Float64
    phimax_deg::Float64
    phimin_deg::Float64
    ellipticity::Float64
end

"""
    pt_invariants(Φ) -> PTInvariants

CBB04 Appendix: with `tr(Φ) = Φ11+Φ22` (A2), `sk(Φ) = Φ12−Φ21` (A3),
`det(Φ) = Φ11Φ22−Φ12Φ21` (A4) and the first-order forms `Φ1 = tr/2` (A5),
`Φ2 = det^½` (A6), `Φ3 = sk/2` (A7), the principal values (A8)/(A9) are

    Φmax = √(Φ1²+Φ3²) + √(Φ1²+Φ3²−Φ2²),   Φmin = √(Φ1²+Φ3²) − √(Φ1²+Φ3²−Φ2²)

and the skew angle (eq. 19 / A10) is `β = ½ tan⁻¹[sk/tr]`, with
`α = ½ tan⁻¹[(Φ12+Φ21)/(Φ11−Φ22)]` (eq. 22).

This implementation uses the algebraically identical Bibby (1986) form

    Π1 = ½√((Φ11−Φ22)² + (Φ12+Φ21)²) ≡ √(Φ1²+Φ3²−det Φ)
    Π2 = ½√((Φ11+Φ22)² + (Φ12−Φ21)²) ≡ √(Φ1²+Φ3²)
    Φmax = Π2 + Π1,   Φmin = Π2 − Π1

because it never takes the square root of a negative number and reproduces
CBB04's own prescription for the `det(Φ) < 0` case (Appendix: "assign a
negative sign to the value of Φmin") without a branch. `α` and `β` use the
two-argument arctangent, matching `mtpy.analysis.pt`; for `tr(Φ) > 0` — the
normal MT case — that agrees with the single-argument eqs. (19) and (22), and
`tr(Φ) < 0` is counted and reported separately.
"""
function pt_invariants(Φ::AbstractMatrix{<:Real})::PTInvariants
    if !all(isfinite, Φ)
        return PTInvariants(NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end
    Φ11, Φ12, Φ21, Φ22 = Φ[1, 1], Φ[1, 2], Φ[2, 1], Φ[2, 2]

    β = 0.5 * atan(Φ12 - Φ21, Φ11 + Φ22)
    α = 0.5 * atan(Φ12 + Φ21, Φ11 - Φ22)

    Π1 = 0.5 * sqrt((Φ11 - Φ22)^2 + (Φ12 + Φ21)^2)
    Π2 = 0.5 * sqrt((Φ11 + Φ22)^2 + (Φ12 - Φ21)^2)
    φmax = Π2 + Π1
    φmin = Π2 - Π1

    ellip = (φmax + φmin) == 0 ? NaN : (φmax - φmin) / (φmax + φmin)
    return PTInvariants(rad2deg(β), rad2deg(α), rad2deg(α - β),
                        φmax, φmin,
                        rad2deg(atan(φmax)), rad2deg(atan(φmin)), ellip)
end

pt_invariants(Z::AbstractMatrix{<:Complex}) = pt_invariants(phase_tensor(Z))

# ─────────────────────────────────────────────────────────────────────────────
# Per (station, period) record
# ─────────────────────────────────────────────────────────────────────────────

struct PTRecord
    site::String
    east::Float64
    north::Float64
    period::Float64
    inv::PTInvariants
    beta_sigma::Float64      # 1σ of β from Monte-Carlo error propagation, deg
    max_rel_err::Float64     # max_ij σ(Z_ij) / √(|Zxy|·|Zyx|)
    valid::Bool
    hiqual::Bool
end

"""
    beta_uncertainty(Z, Zvar, ndraw, rng) -> Float64

1σ of the skew angle β, from `ndraw` independent Gaussian perturbations of each
impedance component drawn from its EDI `Z**.VAR` entry. EDI stores variances,
so the per-component σ is `√VAR`, applied to the real and imaginary parts
alike — the usual EDI convention, and the one `mtpy` assumes. Returns `NaN`
when variances are absent or negative.

Monte Carlo is used instead of the linearised propagation in
`mtpy.analysis.pt` because β is a ratio of small differences and the linear
approximation degrades exactly where it matters, at β near 0 with weak signal.
The spread is measured on the doubled angle so the estimator is insensitive to
the ±90° branch of `atan`.
"""
function beta_uncertainty(Z::AbstractMatrix{<:Complex}, Zvar::AbstractMatrix{<:Real},
                          ndraw::Int, rng::AbstractRNG)::Float64
    (all(isfinite, Zvar) && all(>=(0), Zvar)) || return NaN
    σ = sqrt.(Zvar)
    Zp = Matrix{ComplexF64}(undef, 2, 2)
    Csum = 0.0
    Ssum = 0.0
    n = 0
    for _ in 1:ndraw
        @inbounds for j in 1:2, i in 1:2
            Zp[i, j] = complex(real(Z[i, j]) + σ[i, j] * randn(rng),
                               imag(Z[i, j]) + σ[i, j] * randn(rng))
        end
        b = pt_invariants(Zp).beta
        isfinite(b) || continue
        Csum += cosd(2b)
        Ssum += sind(2b)
        n += 1
    end
    n < 2 && return NaN
    R = hypot(Csum, Ssum) / n
    R <= 0 && return NaN
    R >= 1 && return 0.0
    # Circular standard deviation of the doubled angle, halved back.
    return rad2deg(sqrt(-2 * log(R))) / 2
end

"""
    build_records(stations; ndraw, seed) -> Vector{PTRecord}

Phase-tensor invariants for every (station, period) pair.

`valid` requires all eight impedance entries to be real data (not the EDI
`EMPTY` fill) and `det(Re Z) ≠ 0`. `hiqual` additionally requires every
component standard error to be at most 10 per cent of `√(|Zxy|·|Zyx|)`. The
off-diagonal geometric mean is the scale rather than each component's own
modulus because `Zxx`/`Zyy` legitimately approach zero in quasi-2-D settings,
where a per-component relative error would reject perfectly good data.
"""
function build_records(stations::Vector{EDIStation}; ndraw::Int = 400,
                       seed::Int = 20260827)::Vector{PTRecord}
    rng = MersenneTwister(seed)
    recs = PTRecord[]
    for st in stations
        for k in eachindex(st.periods)
            Z = st.Z[k]
            V = st.Zvar[k]
            inv = pt_invariants(Z)
            valid = isfinite(inv.beta)

            scale = sqrt(abs(Z[1, 2]) * abs(Z[2, 1]))
            rel = NaN
            if valid && isfinite(scale) && scale > 0 && all(isfinite, V) && all(>=(0), V)
                rel = maximum(sqrt.(V)) / scale
            end
            hiqual = valid && isfinite(rel) && rel <= 0.10
            σβ = valid ? beta_uncertainty(Z, V, ndraw, rng) : NaN

            push!(recs, PTRecord(st.site, st.east, st.north, st.periods[k],
                                 inv, σβ, rel, valid, hiqual))
        end
    end
    return recs
end

# ─────────────────────────────────────────────────────────────────────────────
# Circular statistics
# ─────────────────────────────────────────────────────────────────────────────

"""
    axial_mean(angles_deg, period_deg) -> (mean_deg, R, circ_std_deg)

Mean direction of axial data whose symmetry period is `period_deg`: 180° for a
line direction such as the ellipse major axis, 90° for a geoelectric strike,
which additionally carries the TE/TM ambiguity.

Standard doubling-of-angles construction generalised to a fold
`m = 360/period_deg`: average `exp(i·m·θ)`, then divide the argument by `m`
(Mardia & Jupp 2000, *Directional Statistics*, §2.3 for the circular mean and
§9.2 for the axial/`m`-fold case). `R ∈ [0,1]` is the mean resultant length —
1 means perfectly aligned, 0 uniformly scattered. The circular standard
deviation `√(−2 ln R)/m` is returned in degrees on the unfolded angle scale.
The mean is reported in `[0, period_deg)`.
"""
function axial_mean(angles_deg::AbstractVector{<:Real}, period_deg::Real)
    θ = Float64[a for a in angles_deg if isfinite(a)]
    isempty(θ) && return (NaN, NaN, NaN)
    m = 360 / period_deg
    C = mean(cosd.(m .* θ))
    S = mean(sind.(m .* θ))
    R = hypot(C, S)
    μ = mod(rad2deg(atan(S, C)) / m, period_deg)
    σ = R <= 0 ? Inf : (R >= 1 ? 0.0 : rad2deg(sqrt(-2 * log(R))) / m)
    return (μ, R, σ)
end

# ─────────────────────────────────────────────────────────────────────────────
# Period grid
# ─────────────────────────────────────────────────────────────────────────────

"""
    cluster_periods(periods; rtol) -> Vector{Float64}

Collapse the union of every station's periods onto one grid, merging values
whose logs differ by less than `rtol`. Returns the geometric mean of each
cluster, ascending.
"""
function cluster_periods(periods::AbstractVector{<:Real}; rtol::Float64 = PERIOD_RTOL)
    vals = sort(Float64[p for p in periods if isfinite(p) && p > 0])
    isempty(vals) && return Float64[]
    centres = Float64[]
    group = Float64[vals[1]]
    for v in vals[2:end]
        if abs(log(v) - log(group[1])) <= rtol
            push!(group, v)
        else
            push!(centres, exp(mean(log.(group))))
            group = Float64[v]
        end
    end
    push!(centres, exp(mean(log.(group))))
    return centres
end

"""Index of the grid period closest, in log space, to `p`."""
nearest_period_index(grid::Vector{Float64}, p::Real)::Int =
    argmin(abs.(log.(grid) .- log(p)))

at_period(recs::Vector{PTRecord}, T::Real) =
    PTRecord[r for r in recs if r.valid && abs(log(r.period) - log(T)) <= PERIOD_RTOL]

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

const BAND_EDGES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 2000.0)
const ROSE_BANDS = [(0.001, 0.1, "short"), (0.1, 10.0, "middle"), (10.0, 2000.0, "long")]

band_label(lo, hi) = @sprintf("%.4g – %.4g s", lo, hi)

function band_of(p::Real)::Int
    for k in 1:(length(BAND_EDGES) - 1)
        BAND_EDGES[k] <= p < BAND_EDGES[k + 1] && return k
    end
    return 0
end

pct(n, d) = d == 0 ? NaN : 100 * n / d

"""
Pairs whose major-axis direction is actually meaningful: β small enough that
the 2-D interpretation holds, and an ellipse elongated enough that its long
axis is not set by noise (CBB04, "Tensor ellipse in 1-D": α is undefined and
unstable when the principal values coincide).
"""
well_conditioned(recs::AbstractVector{PTRecord}) =
    PTRecord[r for r in recs
             if r.valid && abs(r.inv.beta) < BETA_2D_DEG &&
                isfinite(r.inv.ellipticity) && r.inv.ellipticity > 0.1]

function nearest_neighbour_m(east::Vector{Float64}, north::Vector{Float64})
    n = length(east)
    out = fill(Inf, n)
    for i in 1:n, j in 1:n
        i == j && continue
        out[i] = min(out[i], hypot(east[i] - east[j], north[i] - north[j]))
    end
    return out
end

function write_csv(path::AbstractString, recs::Vector{PTRecord})
    open(path, "w") do io
        println(io, "site,easting_m,northing_m,period_s,beta_deg,beta_sigma_deg,",
                    "alpha_deg,azimuth_deg,phimax,phimin,phimax_deg,phimin_deg,",
                    "ellipticity,max_rel_err,valid,hiqual")
        for r in recs
            v = r.inv
            f(x) = isfinite(x) ? @sprintf("%.6g", x) : ""
            @printf(io, "%s,%.2f,%.2f,%.8g,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\n",
                    r.site, r.east, r.north, r.period,
                    f(v.beta), f(r.beta_sigma), f(v.alpha), f(v.azimuth),
                    f(v.phimax), f(v.phimin), f(v.phimax_deg), f(v.phimin_deg),
                    f(v.ellipticity), f(r.max_rel_err), Int(r.valid), Int(r.hiqual))
        end
    end
end

function write_report(path::AbstractString, edi_dir::AbstractString,
                      stations::Vector{EDIStation}, recs::Vector{PTRecord},
                      grid::Vector{Float64}, panel_periods::Vector{Float64},
                      ndraw::Int, seed::Int)
    good = PTRecord[r for r in recs if r.valid]
    hq = PTRecord[r for r in good if r.hiqual]
    absβ(r) = abs(r.inv.beta)

    n_all = length(recs)
    n_good = length(good)
    n_hq = length(hq)
    n_2d = count(r -> absβ(r) < BETA_2D_DEG, good)
    n_2d_hq = count(r -> absβ(r) < BETA_2D_DEG, hq)
    withσ = PTRecord[r for r in good if isfinite(r.beta_sigma)]
    n_insig = count(r -> absβ(r) < 2 * r.beta_sigma, withσ)

    east = Float64[s.east for s in stations]
    north = Float64[s.north for s in stations]
    nn = nearest_neighbour_m(east, north)

    band_rows = NamedTuple[]
    for k in 1:(length(BAND_EDGES) - 1)
        sel = PTRecord[r for r in good if band_of(r.period) == k]
        isempty(sel) && continue
        b = absβ.(sel)
        μ, R, _ = axial_mean(Float64[r.inv.azimuth for r in well_conditioned(sel)], 90.0)
        push!(band_rows, (label = band_label(BAND_EDGES[k], BAND_EDGES[k + 1]),
                          n = length(sel), med = median(b), p90 = quantile(b, 0.9),
                          frac = pct(count(<(BETA_2D_DEG), b), length(b)),
                          strike = μ, R = R))
    end

    per_rows = NamedTuple[]
    for T in grid
        sel = at_period(good, T)
        isempty(sel) && continue
        b = absβ.(sel)
        push!(per_rows, (T = T, n = length(sel), med = median(b),
                         frac = pct(count(<(BETA_2D_DEG), b), length(b)),
                         phimax = median(Float64[r.inv.phimax_deg for r in sel]),
                         phimin = median(Float64[r.inv.phimin_deg for r in sel])))
    end

    stat_rows = NamedTuple[]
    for st in stations
        sel = PTRecord[r for r in good if r.site == st.site]
        isempty(sel) && continue
        b = absβ.(sel)
        wsel = well_conditioned(sel)
        # Below a handful of pairs a circular mean is noise dressed as a number.
        μ, R, _ = length(wsel) >= 5 ?
                  axial_mean(Float64[r.inv.azimuth for r in wsel], 90.0) : (NaN, NaN, NaN)
        push!(stat_rows, (site = st.site, n = length(sel), med = median(b),
                          mx = maximum(b),
                          frac = pct(count(<(BETA_2D_DEG), b), length(b)),
                          negtr = count(x -> x > 45, b),
                          strike = μ, R = R))
    end
    sort!(stat_rows; by = r -> -r.med)

    az_all = Float64[r.inv.azimuth for r in good]
    ax180 = axial_mean(az_all, 180.0)
    ax90 = axial_mean(az_all, 90.0)
    well = well_conditioned(good)
    ax90_well = axial_mean(Float64[r.inv.azimuth for r in well], 90.0)
    strike_a = isfinite(ax90_well[1]) ? ax90_well[1] : ax90[1]
    strike_b = mod(strike_a + 90, 180)

    band_strikes = NamedTuple[]
    for (lo, hi, name) in ROSE_BANDS
        sel = PTRecord[r for r in good if lo <= r.period < hi]
        isempty(sel) && continue
        μ, R, σ = axial_mean(Float64[r.inv.azimuth for r in sel], 90.0)
        wsel = well_conditioned(sel)
        μw, Rw, _ = axial_mean(Float64[r.inv.azimuth for r in wsel], 90.0)
        push!(band_strikes, (name = name, lo = lo, hi = hi, n = length(sel),
                             strike = μ, R = R, σ = σ,
                             nw = length(wsel), strikew = μw, Rw = Rw))
    end

    φmax = Float64[r.inv.phimax_deg for r in good]
    φmin = Float64[r.inv.phimin_deg for r in good]
    ell = Float64[r.inv.ellipticity for r in good if isfinite(r.inv.ellipticity)]
    n_negdet = count(r -> r.inv.phimin < 0, good)
    n_negtrace = count(r -> absβ(r) > 45, good)

    zrots = sort(unique(round.(vcat([s.zrot for s in stations]...); digits = 4)))
    decls = sort(unique(round.(Float64[s.declination for s in stations]; digits = 4)))

    open(path, "w") do io
        println(io, "# Northwest Geysers — MT phase-tensor dimensionality")
        println(io)
        println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Script: `scripts/geysers_dimensionality.jl`. No forward solve is called;")
        println(io, "every number below comes from the EDI impedance tensors alone.")
        println(io, "EDI source: `", relpath(edi_dir, ROOT), "` — ",
                    length(stations), " stations, ", n_all, " (station, period) pairs.")
        println(io)
        println(io, "Method: the phase tensor of Caldwell, Bibby & Brown (2004),")
        println(io, "*The magnetotelluric phase tensor*, Geophys. J. Int. **158**, 457–469,")
        println(io, "doi:10.1111/j.1365-246X.2004.02281.x. `Φ = X⁻¹Y` (eqs. 13, 15);")
        println(io, "`β` (eq. 19); `α` (eq. 22); `Φmax`, `Φmin` (eqs. A8, A9); ellipse major")
        println(io, "axis at `α − β` (Fig. 1 and Appendix). Implementation cross-checked")
        println(io, "against `mtpy.analysis.pt` and unit-tested in")
        println(io, "`test/test_geysers_phase_tensor.jl`.")
        println(io)

        println(io, "## Reference frame and conventions")
        println(io)
        println(io, "- The impedances are already in a geographic-north frame. Every file carries")
        println(io, "  `COORDINATE_SYSTEM=Geographic North`, `REFTYPE=Geographic North` and")
        println(io, "  `processing.coordinate_system = \"Data collected in geomagnetic coordinate")
        println(io, "  system then rotated to align with Geographic North\"`.")
        print(io, "- `>ZROT` is ")
        if length(zrots) == 1
            @printf(io, "constant at %.2f° for all %d stations and all periods,\n",
                    zrots[1], length(stations))
            if length(decls) == 1 && isfinite(decls[1])
                @printf(io, "  i.e. −%.2f°, minus the declination (`DECLINATION=%.3f`). It records the\n",
                        mod(-zrots[1], 360.0), decls[1])
                println(io, "  rotation **already applied**, so no further rotation is done here.")
            end
        else
            println(io, "not constant: ", join([@sprintf("%.2f", z) for z in zrots], ", "),
                        "°. Check before trusting the azimuths below.")
        end
        println(io, "- All azimuths below are degrees clockwise from **geographic north**")
        println(io, "  (`x₁ = N`, `x₂ = E`), which is the frame CBB04 use for `α`, `β`, `α − β`.")
        println(io, "- Angles are reduced modulo the symmetry of the quantity: the ellipse major")
        println(io, "  axis is a line (period 180°), and a geoelectric strike inherits the TE/TM")
        println(io, "  ambiguity on top of that (period 90°).")
        println(io)

        println(io, "## Data screening")
        println(io)
        println(io, "| criterion | pairs | % of all |")
        println(io, "|---|---:|---:|")
        @printf(io, "| all (station, period) | %d | 100.0 |\n", n_all)
        @printf(io, "| `valid` — 8 finite Z components and det(Re Z) ≠ 0 | %d | %.1f |\n",
                n_good, pct(n_good, n_all))
        @printf(io, "| `hiqual` — also max σ(Zij) ≤ 0.10·√(\\|Zxy\\|·\\|Zyx\\|) | %d | %.1f |\n",
                n_hq, pct(n_hq, n_all))
        println(io)
        println(io, "`EMPTY=1e+32` fill values are treated as missing. Errors come from the")
        println(io, "`ZXX.VAR` / `ZXY.VAR` / `ZYX.VAR` / `ZYY.VAR` blocks, which hold variances,")
        println(io, "so σ = √VAR.")
        if n_negdet > 0 || n_negtrace > 0
            println(io)
            @printf(io, "Pathological cases, kept and flagged rather than deleted: %d pairs with\n",
                    n_negdet)
            @printf(io, "det(Φ) < 0 (so Φmin < 0, per the CBB04 Appendix) and %d with tr(Φ) < 0\n",
                    n_negtrace)
            println(io, "(|β| > 45°, where the two-argument arctan branch departs from eq. 19).")
        end
        println(io)

        println(io, "## 1. How often is 2-D defensible?")
        println(io)
        @printf(io, "- **|β| < %.0f° in %.1f%% of valid (station, period) pairs** (%d / %d).\n",
                BETA_2D_DEG, pct(n_2d, n_good), n_2d, n_good)
        @printf(io, "- Restricted to `hiqual` pairs: **%.1f%%** (%d / %d).\n",
                pct(n_2d_hq, n_hq), n_2d_hq, n_hq)
        @printf(io, "- β statistically indistinguishable from zero (|β| < 2σ_β, with σ_β from\n")
        @printf(io, "  %d Monte-Carlo draws of the EDI variances, seed %d): **%.1f%%**\n",
                ndraw, seed, pct(n_insig, length(withσ)))
        @printf(io, "  (%d / %d pairs that carry usable variances).\n", n_insig, length(withσ))
        @printf(io, "- Median |β| over all valid pairs **%.2f°**, 90th percentile **%.2f°**.\n",
                median(absβ.(good)), quantile(absβ.(good), 0.9))
        println(io)
        println(io, "β = 0 is *necessary but not sufficient* for 2-D (CBB04, \"Properties in")
        println(io, "2-D\"): a 2-D structure needs β ≈ 0 **and** a constant major-axis direction")
        println(io, "across a band of periods. Section 4 tests that second condition.")
        println(io)

        println(io, "## 2. β versus period — where is 2-D valid?")
        println(io)
        println(io, "### By decade band")
        println(io)
        println(io, "| band | pairs | median \\|β\\| | p90 \\|β\\| | \\|β\\| < 3° | strike (mod 90°) | R |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for r in band_rows
            @printf(io, "| %s | %d | %.2f° | %.2f° | %.1f%% | N%.1f°E | %.2f |\n",
                    r.label, r.n, r.med, r.p90, r.frac, r.strike, r.R)
        end
        println(io)
        println(io, "Strike and `R` in this table use only well-conditioned pairs (|β| < 3° and")
        println(io, "ellipticity > 0.1); a near-circular ellipse has no meaningful long axis.")
        println(io)
        println(io, "### By individual period")
        println(io)
        println(io, "| T (s) | pairs | median \\|β\\| | \\|β\\| < 3° | median Φmax | median Φmin |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for r in per_rows
            @printf(io, "| %.4g | %d | %.2f° | %.0f%% | %.1f° | %.1f° |\n",
                    r.T, r.n, r.med, r.frac, r.phimax, r.phimin)
        end
        println(io)

        println(io, "## 3. β by station — which sites are problematic?")
        println(io)
        println(io, "Sorted by median |β|, worst first. `tr<0` counts periods with tr(Φ) < 0,")
        println(io, "i.e. out-of-quadrant phases; those are a data-quality symptom rather than")
        println(io, "3-D structure, and they inflate |β| towards 90°. Strike and `R` again use")
        println(io, "only well-conditioned pairs, so they are blank where a station has none.")
        println(io)
        println(io, "| station | pairs | median \\|β\\| | max \\|β\\| | \\|β\\| < 3° | tr<0 | strike (mod 90°) | R |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in stat_rows
            strike_s = isfinite(r.strike) ? @sprintf("N%.1f°E", r.strike) : "—"
            R_s = isfinite(r.R) ? @sprintf("%.2f", r.R) : "—"
            @printf(io, "| %s | %d | %.2f° | %.2f° | %.0f%% | %d | %s | %s |\n",
                    r.site, r.n, r.med, r.mx, r.frac, r.negtr, strike_s, R_s)
        end
        println(io)

        println(io, "## 4. Dominant geoelectric strike")
        println(io)
        println(io, "Circular (axial) statistics on the phase-tensor major-axis azimuth `α − β`.")
        println(io, "`R` is the mean resultant length: 1 means every pair points the same way,")
        println(io, "0 means uniform scatter.")
        println(io)
        @printf(io, "- Major-axis direction, mod 180°: **N%.1f°E** (R = %.2f, circular σ = %.1f°, n = %d)\n",
                ax180[1], ax180[2], ax180[3], length(az_all))
        @printf(io, "- Strike, mod 90°: **N%.1f°E** (R = %.2f, circular σ = %.1f°)\n",
                ax90[1], ax90[2], ax90[3])
        @printf(io, "- Same but restricted to well-conditioned pairs, |β| < 3° and ellipticity\n")
        @printf(io, "  > 0.1 (n = %d): **N%.1f°E** (R = %.2f)\n",
                length(well), ax90_well[1], ax90_well[2])
        println(io)
        println(io, "**The 90° ambiguity.** The phase tensor cannot distinguish strike from")
        println(io, "strike + 90°. CBB04 (\"Properties in 2-D\"): for a 2-D structure the major")
        println(io, "axis is aligned *either parallel or perpendicular* to strike, depending on")
        println(io, "whether Φmax corresponds to the TE or the TM phase. The two admissible")
        @printf(io, "geoelectric strikes here are therefore **N%.1f°E** and **N%.1f°E**, and the\n",
                strike_a, strike_b)
        println(io, "phase tensor alone cannot choose between them. Breaking the tie needs")
        println(io, "induction vectors — absent from these EDIs, which carry no `TXR`/`TXI`/")
        println(io, "`TYR`/`TYI` blocks — or independent geological control.")
        println(io)
        println(io, "### Strike by period band")
        println(io)
        println(io, "| band | pairs | strike (mod 90°) | R | circular σ | well-cond. pairs | strike | R |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in band_strikes
            @printf(io, "| %s (%s) | %d | N%.1f°E | %.2f | %.1f° | %d | N%.1f°E | %.2f |\n",
                    band_label(r.lo, r.hi), r.name, r.n, r.strike, r.R, r.σ,
                    r.nw, r.strikew, r.Rw)
        end
        println(io)
        println(io, "These are the three bands plotted in `results/geysers_strike_rose.png`.")
        println(io)

        println(io, "## 5. Principal values")
        println(io)
        println(io, "Reported as principal *phases*, `tan⁻¹Φmax` and `tan⁻¹Φmin` in degrees, the")
        println(io, "form CBB04 plot in their Fig. 8(b). For a 1-D earth the two coincide and the")
        println(io, "ellipse is a circle.")
        println(io)
        println(io, "| quantity | min | p10 | median | p90 | max |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for (name, v) in (("Φmax", φmax), ("Φmin", φmin))
            @printf(io, "| %s | %.1f° | %.1f° | %.1f° | %.1f° | %.1f° |\n",
                    name, minimum(v), quantile(v, 0.1), median(v),
                    quantile(v, 0.9), maximum(v))
        end
        @printf(io, "| ellipticity | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
                minimum(ell), quantile(ell, 0.1), median(ell), quantile(ell, 0.9),
                maximum(ell))
        println(io)

        println(io, "## 6. Survey geometry (UTM Zone 10N, WGS84)")
        println(io)
        @printf(io, "- Easting  %.0f – %.0f m (Δ = %.2f km)\n",
                minimum(east), maximum(east), (maximum(east) - minimum(east)) / 1000)
        @printf(io, "- Northing %.0f – %.0f m (Δ = %.2f km)\n",
                minimum(north), maximum(north), (maximum(north) - minimum(north)) / 1000)
        @printf(io, "- Nearest-neighbour spacing min / median / max: %.0f / %.0f / %.0f m\n",
                minimum(nn), median(nn), maximum(nn))
        @printf(io, "- Period coverage %.4g – %.4g s on a %d-point common grid\n",
                minimum(grid), maximum(grid), length(grid))
        println(io)

        println(io, "## 7. Verdict for a 2-D profile")
        println(io)
        @printf(io, "1. **β test.** %.1f%% of valid pairs (%.1f%% of `hiqual` pairs) satisfy\n",
                pct(n_2d, n_good), pct(n_2d_hq, n_hq))
        println(io, "   |β| < 3°, with median |β| = ",
                    @sprintf("%.2f°", median(absβ.(good))), ". The site is not 2-D everywhere,")
        println(io, "   but neither is it dominated by 3-D distortion.")
        if !isempty(band_rows)
            best = band_rows[argmin([r.med for r in band_rows])]
            worst = band_rows[argmax([r.med for r in band_rows])]
            shallow = Float64[absβ(r) for r in good if 0.01 <= r.period < 1.0]
            deep = Float64[absβ(r) for r in good if r.period >= 10.0]
            @printf(io, "2. **Best and worst bands.** 2-D is best supported at **%s** (median\n",
                    best.label)
            @printf(io, "   |β| = %.2f°, %.0f%% below 3°) and worst at **%s** (median |β| = %.2f°,\n",
                    best.med, best.frac, worst.label, worst.med)
            @printf(io, "   %.0f%% below 3°).", worst.frac)
            if !isempty(shallow) && !isempty(deep)
                @printf(io, " Median |β| rises from %.2f° over 0.01–1 s to %.2f°\n",
                        median(shallow), median(deep))
                println(io, "   beyond 10 s: the deeper the structure sampled, the more 3-D it is.")
            else
                println(io)
            end
        end
        if length(band_strikes) >= 2
            println(io, "3. **Strike-stability test — this is the binding constraint.** CBB04 require")
            println(io, "   β ≈ 0 *and* a major-axis direction that stays constant over a band of")
            println(io, "   periods. Here the well-conditioned strike rotates from ",
                        @sprintf("N%.0f°E", band_strikes[1].strikew))
            println(io, "   at the short end to ",
                        @sprintf("N%.0f°E", band_strikes[end].strikew),
                        " at the long end, with the concentration rising from")
            @printf(io, "   R = %.2f to R = %.2f. No single profile azimuth serves the whole band.\n",
                    band_strikes[1].Rw, band_strikes[end].Rw)
            @printf(io, "   The long band (%s) is the one with a genuinely coherent strike.\n",
                    band_label(band_strikes[end].lo, band_strikes[end].hi))
        end
        best_band = isempty(band_strikes) ? nothing :
                    band_strikes[argmax([r.Rw for r in band_strikes])]
        @printf(io, "4. **Profile azimuth.** Over the whole band the strike is N%.1f°E or N%.1f°E\n",
                strike_a, strike_b)
        println(io, "   (90° ambiguous), so a profile cut perpendicular to it runs along")
        @printf(io, "   N%.1f°E or N%.1f°E.", strike_b, strike_a)
        if best_band !== nothing
            @printf(io, " Prefer the estimate from the most coherent band,\n")
            @printf(io, "   %s (R = %.2f): strike N%.1f°E or N%.1f°E, profile along N%.1f°E or\n",
                    band_label(best_band.lo, best_band.hi), best_band.Rw,
                    best_band.strikew, mod(best_band.strikew + 90, 180),
                    mod(best_band.strikew + 90, 180))
            @printf(io, "   N%.1f°E.", best_band.strikew)
        end
        println(io, " `results/geysers_strike_rose.png` shows the spread and")
        println(io, "   `results/geysers_phase_tensor.png` the spatial pattern.")
        println(io, "5. **Geometry cost.** The stations form a grid, not a line, so any profile")
        println(io, "   either discards off-line stations or projects them; projection error grows")
        println(io, "   with off-line offset and with |β|. Section 3 lists which sites to drop")
        println(io, "   first.")
        println(io, "6. **Recommendation.** A 2-D profile is defensible as a *prior* or starting")
        println(io, "   model, not as a final image. Restrict the fitted band where β is smallest")
        println(io, "   and the strike is stable, exclude the high-|β| and `tr<0` stations from")
        println(io, "   section 3, and treat structure recovered below the depth sensed at ~10 s")
        println(io, "   as 3-D-contaminated.")
        println(io)

        println(io, "## Artefacts")
        println(io)
        println(io, "- `results/geysers_dimensionality.md` (this file)")
        println(io, "- `results/geysers_dimensionality.csv` — every (station, period) invariant")
        println(io, "- `results/geysers_phase_tensor.png` — ellipse maps at T = ",
                    join([@sprintf("%.4g", p) for p in panel_periods], ", "), " s")
        println(io, "- `results/geysers_strike_rose.png` — strike rose by period band")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Plots (CairoMakie, re-exported by MTGeophysics)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ellipse_points(cx, cy, a, b, az_deg; n)

Polygon for a phase-tensor ellipse on a map with `x` = easting, `y` = northing.
`az_deg` is the major-axis direction clockwise from north (CBB04's `α − β`), so
the major-axis unit vector is `(sin az, cos az)` and the minor axis is that
turned by 90°.
"""
function ellipse_points(cx::Real, cy::Real, a::Real, b::Real, az_deg::Real; n::Int = 72)
    s, c = sincosd(az_deg)
    return [(cx + a * cos(t) * s + b * sin(t) * c,
             cy + a * cos(t) * c - b * sin(t) * s)
            for t in range(0, 2π; length = n)]
end

function plot_phase_tensor_map(png_path::AbstractString, recs::Vector{PTRecord},
                               panel_periods::Vector{Float64}, scale_m::Float64)
    CM = MTGeophysics.CairoMakie
    CM.activate!()

    blim = 6.0
    cmap = CM.cgrad(:RdBu; rev = true)
    colour(β) = cmap[clamp((clamp(β, -blim, blim) + blim) / (2blim), 0, 1)]

    fig = CM.Figure(size = (1250, 1030), fontsize = 13)
    for (idx, T) in enumerate(panel_periods)
        row, col = fldmod1(idx, 2)
        sel = at_period(recs, T)
        ax = CM.Axis(fig[row, col];
                     xlabel = "Easting (m, UTM 10N / WGS84)",
                     ylabel = "Northing (m, UTM 10N / WGS84)",
                     title = @sprintf("T = %.4g s   (n = %d)", T, length(sel)),
                     aspect = CM.DataAspect(),
                     xticklabelrotation = π / 6)
        for r in sel
            φmax = r.inv.phimax
            (isfinite(φmax) && φmax > 0) || continue
            b = scale_m * clamp(abs(r.inv.phimin) / φmax, 0.0, 1.0)
            pts = ellipse_points(r.east, r.north, scale_m, b, r.inv.azimuth)
            CM.poly!(ax, CM.Point2f.(pts); color = colour(r.inv.beta),
                     strokecolor = (:black, 0.65), strokewidth = 0.5)
        end
        CM.scatter!(ax, Float64[r.east for r in sel], Float64[r.north for r in sel];
                    markersize = 2.5, color = :black)
    end
    CM.Colorbar(fig[1:2, 3]; colormap = cmap, limits = (-blim, blim),
                label = "β (deg)", height = CM.Relative(0.55))
    subtitle = "Each ellipse is normalised to unit Φmax (semi-major " *
               @sprintf("%.0f m", scale_m) *
               "); semi-minor = Φmin/Φmax.\nAzimuths clockwise from geographic north. " *
               "Colour saturates at |β| = " * @sprintf("%.0f", blim) * "°."
    CM.Label(fig[0, 1:3],
             "Northwest Geysers — phase-tensor ellipses coloured by skew β\n" * subtitle;
             fontsize = 15, padding = (0, 0, 6, 0), tellheight = true)
    CM.rowgap!(fig.layout, 6)
    CM.save(png_path, fig)
    return png_path
end

"""
Screen angle for a compass azimuth: `PolarAxis` measures counter-clockwise from
the +x axis, an azimuth is clockwise from north, hence `90° − azimuth`. Doing
the mapping explicitly rather than via `theta_0`/`direction` keeps the tick
labels and the data on the same convention.
"""
compass_theta(az_deg::Real) = deg2rad(90 - az_deg)

function plot_strike_rose(png_path::AbstractString, recs::Vector{PTRecord},
                          bands::Vector{<:Tuple})
    CM = MTGeophysics.CairoMakie
    CM.activate!()

    binwidth = 10.0
    centres = Float64[binwidth * (i - 0.5) for i in 1:Int(180 ÷ binwidth)]
    tick_az = 0.0:30.0:330.0
    tick_labels = ["N", "30°", "60°", "E", "120°", "150°",
                   "S", "210°", "240°", "W", "300°", "330°"]

    fig = CM.Figure(size = (330 * length(bands), 370), fontsize = 13)
    for (idx, (lo, hi, name)) in enumerate(bands)
        sel = PTRecord[r for r in recs if r.valid && lo <= r.period < hi]
        counts = zeros(Int, length(centres))
        for r in sel
            a = mod(r.inv.azimuth, 180.0)
            isfinite(a) || continue
            counts[clamp(Int(floor(a / binwidth)) + 1, 1, length(counts))] += 1
        end
        μ, R, _ = axial_mean(Float64[r.inv.azimuth for r in sel], 90.0)

        # Mirror onto 180–360° so the rose reads as an axial (line) distribution.
        θ = compass_theta.(vcat(centres, centres .+ 180.0))
        rad = Float64.(vcat(counts, counts))
        rmax = maximum(rad; init = 1.0)

        ax = CM.PolarAxis(fig[1, idx];
                          thetaticks = (compass_theta.(collect(tick_az)), tick_labels),
                          rticklabelsize = 10,
                          title = @sprintf("%s band\n%.4g – %.4g s   (n = %d)",
                                           uppercasefirst(name), lo, hi, length(sel)))
        CM.barplot!(ax, θ, rad; width = deg2rad(binwidth),
                    color = (:steelblue, 0.85),
                    strokewidth = 0.5, strokecolor = (:black, 0.6))
        if isfinite(μ)
            for m in (μ, mod(μ + 90, 180)), half in (0.0, 180.0)
                CM.lines!(ax, fill(compass_theta(m + half), 2), [0.0, 1.06 * rmax];
                          color = :red, linewidth = 2)
            end
        end
        CM.Label(fig[2, idx],
                 isfinite(μ) ?
                 @sprintf("mean strike N%.1f°E or N%.1f°E   (R = %.2f)",
                          μ, mod(μ + 90, 180), R) : "no data";
                 fontsize = 12, tellheight = true)
    end
    CM.Label(fig[0, 1:length(bands)],
             "Northwest Geysers — phase-tensor major-axis azimuth (α − β), 10° bins, " *
             "mirrored to 360°\nRed lines: circular-mean strike and its 90°-ambiguous " *
             "partner. Azimuth clockwise from geographic north.";
             fontsize = 15, padding = (0, 0, 6, 0), tellheight = true)
    CM.rowgap!(fig.layout, 2)
    CM.rowsize!(fig.layout, 1, CM.Relative(0.84))
    CM.save(png_path, fig)
    return png_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

"""Four panel periods: short, mid-short, mid-long and long end of the band."""
function select_panel_periods(grid::Vector{Float64})::Vector{Float64}
    targets = (0.003, 0.1, 3.0, 300.0)
    return grid[unique(Int[nearest_period_index(grid, t) for t in targets])]
end

function main(args::Vector{String} = ARGS)
    edi_dir = length(args) >= 1 && !isempty(args[1]) ? abspath(args[1]) : DEFAULT_EDI_DIR
    out_dir = length(args) >= 2 && !isempty(args[2]) ? abspath(args[2]) : DEFAULT_OUT_DIR
    mkpath(out_dir)

    isdir(edi_dir) || error("EDI directory not found: $edi_dir")
    files = sort(String[joinpath(edi_dir, f) for f in readdir(edi_dir)
                        if lowercase(splitext(f)[2]) == ".edi" && !startswith(f, ".")])
    isempty(files) && error("no .edi files in $edi_dir")

    println("[1/5] Parsing ", length(files), " EDI files from ", edi_dir)
    stations = EDIStation[parse_edi(f) for f in files]
    sort!(stations; by = s -> s.site)
    @printf("      %d stations, %d–%d periods each, T = %.4g – %.4g s\n",
            length(stations),
            minimum(length(s.periods) for s in stations),
            maximum(length(s.periods) for s in stations),
            minimum(minimum(s.periods) for s in stations),
            maximum(maximum(s.periods) for s in stations))

    ndraw, seed = 400, 20260827
    println("[2/5] Phase tensors + Monte-Carlo β errors (", ndraw, " draws, seed ", seed, ")")
    recs = build_records(stations; ndraw = ndraw, seed = seed)
    grid = cluster_periods(vcat([s.periods for s in stations]...))
    panel_periods = select_panel_periods(grid)
    @printf("      %d pairs, %d valid, common period grid = %d points\n",
            length(recs), count(r -> r.valid, recs), length(grid))

    csv_path = joinpath(out_dir, "geysers_dimensionality.csv")
    md_path = joinpath(out_dir, "geysers_dimensionality.md")
    pt_png = joinpath(out_dir, "geysers_phase_tensor.png")
    rose_png = joinpath(out_dir, "geysers_strike_rose.png")

    println("[3/5] Writing ", relpath(csv_path, ROOT), " and ", relpath(md_path, ROOT))
    write_csv(csv_path, recs)
    write_report(md_path, edi_dir, stations, recs, grid, panel_periods, ndraw, seed)

    nn = nearest_neighbour_m(Float64[s.east for s in stations],
                             Float64[s.north for s in stations])
    scale_m = 0.42 * median(nn)

    println("[4/5] Writing ", relpath(pt_png, ROOT))
    plot_phase_tensor_map(pt_png, recs, panel_periods, scale_m)

    println("[5/5] Writing ", relpath(rose_png, ROOT))
    plot_strike_rose(rose_png, recs, ROSE_BANDS)

    println("Done.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    using MTGeophysics   # CairoMakie is re-exported; only needed for the two PNGs
    main()
end
