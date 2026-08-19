"""
    KeivitsaStats

Empirical "realism" statistics for the Keivitsa Ni–Cu–PGE deposit, distilled into
sampling priors for the synthetic 3-D resistivity model generator.

# Sources
- `LUO_R` — core/downhole resistivity from `petro.txt` (ohm·m, ~63k records)
- `vlf_resistivity` — VLF-R apparent resistivity, column `ov` (ohm·m)
- `slingram_real` — Slingram in-phase response, column `Re%` (percent, **not** ohm·m;
  used for anomaly geometry only)

# Pipeline
1. Load the GTK tables, reject non-finite / non-positive / out-of-range samples and
   move resistivity to `log10(ρ)` space (all downstream statistics live there).
2. Separate the log10 population into conductive body / host rock / resistive body
   with a 1-D Gaussian mixture (EM), and report per-class min/median/max/p05/p95.
3. Quantify vertical structure along boreholes (layer thickness, conductive interval
   thickness) and spatial continuity (autocorrelation e-folding length + exponential
   variogram range) both vertically and horizontally.
4. Quantify plan-view anomaly geometry from a VLF raster (connected components →
   area, equivalent diameter, principal axes, aspect ratio, strike azimuth).
5. Emit `config/keivitsa_priors.json`, a structured dictionary the synthetic model
   generator reads directly (`generator_priors` block).

# Units
Resistivity `ohm.m` (log10 where stated), lengths `m`, azimuth `deg` clockwise from
grid north, coordinates KKJ / Finland Zone 3 (EPSG:2391).

# Example
```julia
include("src/analysis/keivitsa_stats.jl")
priors = KeivitsaStats.run_stats()
priors["generator_priors"]["host_rock"]["log10_mean"]
```
"""
module KeivitsaStats

using Dates
using JSON3
using Printf
using Random
using Statistics
using YAML

export StatsOptions
export compute_priors, run_stats, write_priors, load_priors
export describe, fit_gmm1d, experimental_variogram, fit_exponential_variogram
export segment_layers, low_runs, acf1d, efolding_length, plan_anomalies

const SCHEMA_VERSION = "keivitsa_priors/v1"

"""GTK TXT exports carry 4 header rows: names, types, widths, decimals."""
const GTK_HEADER_ROWS = 4

"""GTK shapefile stem → TXT mirror that is actually present under `database/`."""
const SHP_TO_TXT = Dict{String,String}(
    "petroph.shp" => "petro.txt",
    "collar.shp"  => "reiat.txt",
    "survey.shp"  => "kalte.txt",
)

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

"""
    StatsOptions

Tunable thresholds for the statistics run. Every field that affects a reported
number is echoed into the output JSON under `options`, so a prior file is always
reproducible from itself.

# Fields
- `config_path`: dataset YAML (defaults to `config/dataset_config.yaml`)
- `output_json`, `output_yaml`: artefact paths; `nothing` disables YAML mirroring
- `rho_range`: hard sanity window for resistivity, ohm·m
- `mad_sigma`: robust outlier rejection in log10 space, in MAD-σ (`0` disables)
- `n_populations`: Gaussian mixture components fitted to log10 ρ
- `conductor_ohm_m`, `resistor_ohm_m`: fixed geological cutoffs used as cross-check
- `smooth_window_m`: depth window of the along-hole moving median, m
- `log_split`: `|Δlog10 ρ|` that starts a new layer
- `max_depth_gap_m`: depth gap that always starts a new layer, m
- `min_layer_m`: layers thinner than this are treated as sub-resolution noise, m
- `variogram_lag_m`, `variogram_nlags`: vertical (along-hole) variogram binning
- `vlf_lag_m`, `vlf_nlags`: horizontal (along-line) variogram binning
- `raster_cell_m`: VLF raster cell for connected-component geometry, m; `nothing`
  derives it from the survey layout (see [`survey_spacing`](@ref))
- `min_anomaly_cells`: components smaller than this are discarded
- `acf_maxlag`: maximum autocorrelation lag, in samples
- `seed`: RNG seed for the EM initialisation
"""
Base.@kwdef struct StatsOptions
    config_path::Union{Nothing,String} = nothing
    output_json::Union{Nothing,String} = nothing
    output_yaml::Union{Nothing,String} = nothing
    rho_range::Tuple{Float64,Float64} = (1.0e-2, 1.0e7)
    mad_sigma::Float64 = 6.0
    n_populations::Int = 3
    conductor_ohm_m::Float64 = 100.0
    resistor_ohm_m::Float64 = 1.0e4
    smooth_window_m::Float64 = 3.0
    log_split::Float64 = 0.5
    max_depth_gap_m::Float64 = 5.0
    min_layer_m::Float64 = 2.0
    variogram_lag_m::Float64 = 0.5
    variogram_nlags::Int = 80
    vlf_lag_m::Float64 = 25.0
    vlf_nlags::Int = 40
    raster_cell_m::Union{Nothing,Float64} = nothing
    min_anomaly_cells::Int = 2
    acf_maxlag::Int = 80
    seed::Int = 20250818
end

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    project_root(start_path) -> String

Walk upward from `start_path` until a directory holding both `config/` and
`database/` is found.
"""
function project_root(start_path::AbstractString)::String
    d = abspath(start_path)
    isfile(d) && (d = dirname(d))
    while true
        isdir(joinpath(d, "config")) && isdir(joinpath(d, "database")) && return d
        parent = dirname(d)
        parent == d && return dirname(abspath(start_path))
        d = parent
    end
end

"""
    default_config_path(root) -> String

Return `config/dataset_config.yaml`, falling back to `config/config.yaml`.
"""
function default_config_path(root::AbstractString)::String
    for name in ("dataset_config.yaml", "config.yaml")
        p = joinpath(root, "config", name)
        isfile(p) && return p
    end
    error("No dataset YAML found under $(joinpath(root, "config"))")
end

"""
    _stringify_keys(x) -> Any

Recursively coerce YAML mapping keys to `String` for type-stable lookups.
"""
function _stringify_keys(x)
    if x isa AbstractDict
        return Dict{String,Any}(string(k) => _stringify_keys(v) for (k, v) in x)
    elseif x isa AbstractVector
        return Any[_stringify_keys(v) for v in x]
    else
        return x
    end
end

"""
    load_config(path) -> Dict{String,Any}

Parse the dataset YAML into a string-keyed dictionary.
"""
function load_config(path::AbstractString)::Dict{String,Any}
    cfg = _stringify_keys(YAML.load_file(String(path)))
    cfg isa Dict{String,Any} || error("Top-level YAML must be a mapping: $(path)")
    return cfg
end

"""
    lookup(cfg, dotted; default=nothing) -> Any

Walk a dotted YAML path such as `"ground_geophysics.vlf"`, returning `default`
when any level is missing.
"""
function lookup(cfg::AbstractDict, dotted::AbstractString; default=nothing)
    node::Any = cfg
    for part in split(String(dotted), '.')
        node isa AbstractDict || return default
        haskey(node, String(part)) || return default
        node = node[String(part)]
    end
    return node
end

"""
    resolve_table_path(root, yaml_path) -> Union{String,Nothing}

Map a YAML shapefile path onto its GTK TXT mirror. Returns `nothing` when
neither exists.
"""
function resolve_table_path(root::AbstractString, yaml_path::AbstractString)
    full = isabspath(yaml_path) ? String(yaml_path) : joinpath(root, String(yaml_path))
    if endswith(lowercase(yaml_path), ".shp")
        base = lowercase(basename(String(yaml_path)))
        if haskey(SHP_TO_TXT, base)
            txt = joinpath(dirname(full), SHP_TO_TXT[base])
            isfile(txt) && return txt
        end
    end
    isfile(full) && return full
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar / distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_f64(s) -> Float64

Parse a GTK numeric token. Blanks and the GTK null markers `-` / `*` map to `NaN`.
"""
function parse_f64(s::AbstractString)::Float64
    t = strip(s)
    (isempty(t) || t == "-" || t == "*") && return NaN
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

"""
    _norm_token(s) -> String

Normalise a column name for fuzzy header matching (`Re%` ≡ `Re_pct` ≡ `re`).
"""
function _norm_token(s::AbstractString)::String
    t = lowercase(strip(String(s)))
    t = replace(t, r"[/%].*$" => "")
    t = replace(t, r"[^a-z0-9]" => "")
    endswith(t, "pct") && (t = t[1:end-3])
    return t
end

"""
    rnd(x, digits) -> Float64

Round for reporting, propagating `NaN` (converted to `null` on serialisation).
"""
rnd(x::Real, digits::Int=3)::Float64 = isfinite(x) ? round(Float64(x); digits=digits) : NaN

"""
    q(sorted, p) -> Float64

Percentile `p ∈ [0, 100]` of an already-sorted vector.
"""
function q(sorted::AbstractVector{<:Real}, p::Real)::Float64
    isempty(sorted) && return NaN
    return Float64(quantile(sorted, p / 100; sorted=true))
end

"""
    describe(x; digits=4) -> Dict{String,Any}

Location / spread summary of a sample: `n`, `min`, `p05`, `p10`, `p25`, `p50`,
`p75`, `p90`, `p95`, `max`, `mean`, `std`, `iqr`.

Empty input yields `n = 0` and `NaN` elsewhere (serialised as `null`).
"""
function describe(x::AbstractVector{<:Real}; digits::Int=4)::Dict{String,Any}
    if isempty(x)
        return Dict{String,Any}("n" => 0, "min" => NaN, "p05" => NaN, "p10" => NaN,
                                "p25" => NaN, "p50" => NaN, "p75" => NaN, "p90" => NaN,
                                "p95" => NaN, "max" => NaN, "mean" => NaN,
                                "std" => NaN, "iqr" => NaN)
    end
    s = sort(Float64.(x))
    p25 = q(s, 25)
    p75 = q(s, 75)
    return Dict{String,Any}(
        "n" => length(s),
        "min" => rnd(s[1], digits),
        "p05" => rnd(q(s, 5), digits),
        "p10" => rnd(q(s, 10), digits),
        "p25" => rnd(p25, digits),
        "p50" => rnd(q(s, 50), digits),
        "p75" => rnd(p75, digits),
        "p90" => rnd(q(s, 90), digits),
        "p95" => rnd(q(s, 95), digits),
        "max" => rnd(s[end], digits),
        "mean" => rnd(mean(s), digits),
        "std" => rnd(length(s) > 1 ? std(s) : 0.0, digits),
        "iqr" => rnd(p75 - p25, digits),
    )
end

"""
    CleanReport

Audit trail of the outlier filter, so the JSON states exactly how many samples
each rejection rule removed.
"""
struct CleanReport
    n_input::Int
    n_missing::Int       # blank / unparsable → NaN
    n_infinite::Int      # ±Inf
    n_nonpositive::Int   # ρ ≤ 0 (log10 undefined)
    n_out_of_range::Int  # outside `rho_range`
    n_outlier::Int       # rejected by the MAD rule in log10 space
    n_kept::Int
end

report_dict(r::CleanReport) = Dict{String,Any}(
    "n_input" => r.n_input,
    "n_missing" => r.n_missing,
    "n_infinite" => r.n_infinite,
    "n_nonpositive" => r.n_nonpositive,
    "n_out_of_range" => r.n_out_of_range,
    "n_mad_outlier" => r.n_outlier,
    "n_kept" => r.n_kept,
    "kept_fraction" => rnd(r.n_input == 0 ? NaN : r.n_kept / r.n_input, 4),
)

"""
    resistivity_mask(values, opts) -> (BitVector, CleanReport)

Build the keep-mask for a raw resistivity vector: drop `NaN`, `±Inf`, `ρ ≤ 0`
and samples outside `opts.rho_range`, then optionally reject `|log10 ρ − median|
> opts.mad_sigma · σ_MAD`. Returning a mask (rather than a filtered vector) lets
callers filter depth / coordinate arrays in lockstep.
"""
function resistivity_mask(values::AbstractVector{<:Real}, opts::StatsOptions)
    n = length(values)
    keep = falses(n)
    n_missing = 0; n_inf = 0; n_nonpos = 0; n_range = 0
    lo, hi = opts.rho_range
    @inbounds for i in 1:n
        v = Float64(values[i])
        if isnan(v)
            n_missing += 1
        elseif isinf(v)
            n_inf += 1
        elseif v <= 0
            n_nonpos += 1
        elseif v < lo || v > hi
            n_range += 1
        else
            keep[i] = true
        end
    end

    n_outlier = 0
    if opts.mad_sigma > 0 && count(keep) >= 32
        logs = [log10(Float64(values[i])) for i in 1:n if keep[i]]
        med = median(logs)
        σ = 1.4826 * median(abs.(logs .- med))
        if σ > 0
            cut = opts.mad_sigma * σ
            @inbounds for i in 1:n
                keep[i] || continue
                if abs(log10(Float64(values[i])) - med) > cut
                    keep[i] = false
                    n_outlier += 1
                end
            end
        end
    end

    return keep, CleanReport(n, n_missing, n_inf, n_nonpos, n_range, n_outlier, count(keep))
end

"""
    finite_mask(values) -> (BitVector, CleanReport)

Keep-mask for a signed quantity (e.g. Slingram `Re%`): finite values only, no
positivity requirement and no log transform.
"""
function finite_mask(values::AbstractVector{<:Real})
    n = length(values)
    keep = falses(n)
    n_missing = 0; n_inf = 0
    @inbounds for i in 1:n
        v = Float64(values[i])
        if isnan(v)
            n_missing += 1
        elseif isinf(v)
            n_inf += 1
        else
            keep[i] = true
        end
    end
    return keep, CleanReport(n, n_missing, n_inf, 0, 0, 0, count(keep))
end

"""
    decade_occupancy(rho, edges) -> (labels, counts, fractions)

Histogram of resistivity over log10 decades — the generator uses it to weight
which decade a random body is drawn from.
"""
function decade_occupancy(rho::AbstractVector{<:Real}, edges::Vector{Float64})
    nb = length(edges) - 1
    counts = zeros(Int, nb)
    @inbounds for v in rho
        isfinite(v) && v > 0 || continue
        for b in 1:nb
            upper_ok = b == nb ? v <= edges[b+1] : v < edges[b+1]
            if v >= edges[b] && upper_ok
                counts[b] += 1
                break
            end
        end
    end
    labels = [@sprintf("%g-%g", edges[b], edges[b+1]) for b in 1:nb]
    total = max(sum(counts), 1)
    return labels, counts, [round(c / total; digits=5) for c in counts]
end

# ─────────────────────────────────────────────────────────────────────────────
# 1-D Gaussian mixture — host rock vs conductive / resistive bodies
# ─────────────────────────────────────────────────────────────────────────────

"""
    GaussianComponent

One mixture component in `log10(ρ)` space.

# Fields
- `weight`: mixing proportion, Σ = 1
- `mean`, `std`: log10(ohm·m)
"""
struct GaussianComponent
    weight::Float64
    mean::Float64
    std::Float64
end

_logpdf(c::GaussianComponent, x::Float64) =
    -0.5 * ((x - c.mean) / c.std)^2 - log(c.std) - 0.5 * log(2π)

"""
    fit_gmm1d(x, k; iters=300, tol=1e-7, rng, std_floor=0.02) -> Vector{GaussianComponent}

Fit a `k`-component 1-D Gaussian mixture by expectation–maximisation and return
the components sorted by ascending mean, i.e. conductive → resistive.

Means are initialised on evenly spaced quantiles of `x`, which makes the fit
deterministic for a fixed sample; `rng` only breaks ties on degenerate input.
`std_floor` keeps a component from collapsing onto a spike of repeated values.
"""
function fit_gmm1d(x::Vector{Float64}, k::Int;
                   iters::Int=300, tol::Float64=1.0e-7,
                   rng::AbstractRNG=Random.default_rng(),
                   std_floor::Float64=0.02)::Vector{GaussianComponent}
    n = length(x)
    (k < 1 || n < 8k) && return GaussianComponent[]
    s = sort(x)
    μ = [q(s, 100 * (i - 0.5) / k) for i in 1:k]
    for i in 2:k
        μ[i] <= μ[i-1] && (μ[i] = μ[i-1] + 1.0e-3 * (1 + rand(rng)))
    end
    σ = fill(max(std(s) / k, std_floor), k)
    w = fill(1 / k, k)

    resp = zeros(Float64, n, k)
    ll_prev = -Inf
    for _ in 1:iters
        ll = 0.0
        @inbounds for i in 1:n
            m = -Inf
            for j in 1:k
                resp[i, j] = log(w[j]) + _logpdf(GaussianComponent(w[j], μ[j], σ[j]), x[i])
                m = max(m, resp[i, j])
            end
            acc = 0.0
            for j in 1:k
                resp[i, j] = exp(resp[i, j] - m)
                acc += resp[i, j]
            end
            for j in 1:k
                resp[i, j] /= acc
            end
            ll += m + log(acc)
        end
        @inbounds for j in 1:k
            nj = 0.0; sj = 0.0
            for i in 1:n
                nj += resp[i, j]
                sj += resp[i, j] * x[i]
            end
            nj <= 1.0e-9 && continue
            μj = sj / nj
            vj = 0.0
            for i in 1:n
                vj += resp[i, j] * (x[i] - μj)^2
            end
            w[j] = nj / n
            μ[j] = μj
            σ[j] = max(sqrt(vj / nj), std_floor)
        end
        abs(ll - ll_prev) <= tol * abs(ll) && break
        ll_prev = ll
    end

    order = sortperm(μ)
    return [GaussianComponent(w[j], μ[j], σ[j]) for j in order]
end

"""
    mixture_boundary(a, b) -> Float64

log10 ρ where two adjacent weighted components are equally likely, i.e. the
data-driven cutoff between the populations. Bisection on the log-likelihood
difference; falls back to the midpoint of the means when no crossing exists.
"""
function mixture_boundary(a::GaussianComponent, b::GaussianComponent)::Float64
    f(t) = (log(a.weight) + _logpdf(a, t)) - (log(b.weight) + _logpdf(b, t))
    lo, hi = a.mean, b.mean
    hi <= lo && return 0.5 * (lo + hi)
    flo, fhi = f(lo), f(hi)
    (isfinite(flo) && isfinite(fhi) && flo * fhi < 0) || return 0.5 * (lo + hi)
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0
            hi = mid
        else
            lo = mid; flo = fm
        end
    end
    return 0.5 * (lo + hi)
end

"""
    population_labels(k) -> Vector{String}

Geological names for `k` components ordered conductive → resistive.
"""
function population_labels(k::Int)::Vector{String}
    k == 1 && return ["host_rock"]
    k == 2 && return ["conductive_body", "host_rock"]
    k == 3 && return ["conductive_body", "host_rock", "resistive_body"]
    return vcat("conductive_body",
                ["host_rock_$(i)" for i in 1:(k-2)],
                "resistive_body")
end

"""
    assign_population(comps, x) -> Int

Index of the component with the highest posterior responsibility for `x`.
"""
function assign_population(comps::Vector{GaussianComponent}, x::Float64)::Int
    best = 1; bestv = -Inf
    @inbounds for (j, c) in enumerate(comps)
        v = log(c.weight) + _logpdf(c, x)
        v > bestv && (bestv = v; best = j)
    end
    return best
end

"""
    population_summary(comps, logrho; digits=3) -> Dict{String,Any}

Per-population report: mixture parameters, the mixture boundaries that separate
them, and the min/median/max/p05/p95 of the samples hard-assigned to each class,
in both log10 and ohm·m.
"""
function population_summary(comps::Vector{GaussianComponent},
                            logrho::Vector{Float64})::Dict{String,Any}
    isempty(comps) && return Dict{String,Any}("n_components" => 0,
                                              "note" => "too few samples for a mixture fit")
    labels = population_labels(length(comps))
    buckets = [Float64[] for _ in comps]
    for v in logrho
        push!(buckets[assign_population(comps, v)], v)
    end

    classes = Dict{String,Any}()
    for (j, c) in enumerate(comps)
        lg = buckets[j]
        classes[labels[j]] = Dict{String,Any}(
            "index" => j,
            "weight" => rnd(c.weight, 4),
            "log10_mean" => rnd(c.mean, 3),
            "log10_std" => rnd(c.std, 3),
            "mode_ohm_m" => rnd(10.0^c.mean, 2),
            "n_assigned" => length(lg),
            "log10_ohm_m" => describe(lg; digits=3),
            "ohm_m" => describe(isempty(lg) ? Float64[] : 10.0 .^ lg; digits=2),
        )
    end

    bounds_log = [mixture_boundary(comps[j], comps[j+1]) for j in 1:(length(comps)-1)]
    return Dict{String,Any}(
        "n_components" => length(comps),
        "space" => "log10(ohm.m)",
        "labels_conductive_to_resistive" => labels,
        "classes" => classes,
        "boundaries_log10" => [rnd(b, 3) for b in bounds_log],
        "boundaries_ohm_m" => [rnd(10.0^b, 2) for b in bounds_log],
        "method" => "1-D Gaussian mixture, EM, quantile initialisation",
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Continuity: autocorrelation and variogram
# ─────────────────────────────────────────────────────────────────────────────

"""
    acf1d(x; maxlag) -> (lags, ρ)

Biased autocorrelation of a mean-centred series at lags `0:maxlag` (units of
sample spacing).
"""
function acf1d(x::Vector{Float64}; maxlag::Int=80)
    n = length(x)
    n < 8 && return Int[], Float64[]
    y = x .- mean(x)
    var0 = sum(abs2, y)
    var0 <= 0 && return Int[], Float64[]
    maxl = min(maxlag, n - 2)
    lags = collect(0:maxl)
    ρ = Vector{Float64}(undef, maxl + 1)
    @inbounds for k in 0:maxl
        acc = 0.0
        for i in 1:(n-k)
            acc += y[i] * y[i+k]
        end
        ρ[k+1] = acc / var0
    end
    return lags, ρ
end

"""
    efolding_length(lags, ρ) -> Float64

First lag where the autocorrelation drops to `1/e`, linearly interpolated.
Returns the largest lag when it never does.
"""
function efolding_length(lags::Vector{Int}, ρ::Vector{Float64})::Float64
    length(ρ) < 2 && return NaN
    target = 1 / ℯ
    for i in 2:length(ρ)
        ρ[i] > target && continue
        r0, r1 = ρ[i-1], ρ[i]
        l0, l1 = Float64(lags[i-1]), Float64(lags[i])
        r0 == r1 && return l0
        return l0 + (target - r0) / (r1 - r0) * (l1 - l0)
    end
    return Float64(lags[end])
end

"""
    VariogramAccumulator

Pooled semivariance bins, so many boreholes / survey lines contribute to one
experimental variogram.
"""
struct VariogramAccumulator
    lag::Float64
    sums::Vector{Float64}
    counts::Vector{Int}
end

VariogramAccumulator(lag::Real, nlags::Integer) =
    VariogramAccumulator(Float64(lag), zeros(Float64, nlags), zeros(Int, nlags))

"""
    accumulate_variogram!(acc, coords, values)

Add all pairs of one 1-D profile (`coords` ascending) to the pooled bins.
Pairs beyond `lag · nlags` are skipped, which keeps the double loop near-linear.
"""
function accumulate_variogram!(acc::VariogramAccumulator,
                               coords::AbstractVector{Float64},
                               values::AbstractVector{Float64})
    n = length(coords)
    n < 2 && return acc
    nb = length(acc.counts)
    hmax = acc.lag * nb
    @inbounds for i in 1:(n-1)
        vi = values[i]
        ci = coords[i]
        for j in (i+1):n
            h = coords[j] - ci
            h > hmax && break
            h <= 0 && continue
            b = min(nb, max(1, ceil(Int, h / acc.lag)))
            d = vi - values[j]
            acc.sums[b] += d * d
            acc.counts[b] += 1
        end
    end
    return acc
end

"""
    experimental_variogram(acc) -> (h, γ, counts)

Bin centres `h` (m), semivariance `γ = Σ(Δv)²/2N` and pair counts for the
populated bins only.
"""
function experimental_variogram(acc::VariogramAccumulator)
    nb = length(acc.counts)
    idx = [b for b in 1:nb if acc.counts[b] > 0]
    h = [(b - 0.5) * acc.lag for b in idx]
    γ = [acc.sums[b] / (2 * acc.counts[b]) for b in idx]
    return h, γ, acc.counts[idx]
end

"""
    fit_exponential_variogram(h, γ, counts) -> NamedTuple

Least-squares fit of `γ(h) = c₀ + c₁ (1 − exp(−3h/a))`, weighted by pair counts.
`a` is the practical range (95 % of the sill), scanned on a log grid while the
nugget `c₀` and partial sill `c₁` are solved analytically and clamped to ≥ 0.

# Returns
`(range_m, nugget, sill, partial_sill, rmse, n_bins)`; `range_m` is `NaN` when
there is not enough data to fit.
"""
function fit_exponential_variogram(h::Vector{Float64}, γ::Vector{Float64},
                                   counts::Vector{Int})
    nb = length(h)
    nb < 4 && return (range_m=NaN, nugget=NaN, sill=NaN, partial_sill=NaN,
                      rmse=NaN, n_bins=nb)
    w = Float64.(counts)
    best = (range_m=NaN, nugget=NaN, sill=NaN, partial_sill=NaN, rmse=Inf, n_bins=nb)
    amin = max(0.5 * minimum(h), 1.0e-6)
    amax = 5.0 * maximum(h)
    for t in range(log(amin), log(amax); length=240)
        a = exp(t)
        f = @. 1 - exp(-3 * h / a)
        sw = sum(w)
        sf = sum(w .* f)
        sff = sum(w .* f .* f)
        sg = sum(w .* γ)
        sfg = sum(w .* f .* γ)
        det = sw * sff - sf * sf
        c0, c1 = if abs(det) < 1.0e-12
            (0.0, sff > 0 ? sfg / sff : 0.0)
        else
            ((sff * sg - sf * sfg) / det, (sw * sfg - sf * sg) / det)
        end
        if c1 < 0
            c1 = 0.0
            c0 = sg / sw
        end
        if c0 < 0
            c0 = 0.0
            c1 = sff > 0 ? sfg / sff : 0.0
        end
        resid = @. c0 + c1 * f - γ
        rmse = sqrt(sum(w .* resid .* resid) / sw)
        if rmse < best.rmse
            best = (range_m=a, nugget=c0, sill=c0 + c1, partial_sill=c1,
                    rmse=rmse, n_bins=nb)
        end
    end
    return best
end

"""
    variogram_dict(acc; digits=3) -> Dict{String,Any}

Serialisable experimental variogram plus its fitted exponential model.
"""
function variogram_dict(acc::VariogramAccumulator; digits::Int=3)::Dict{String,Any}
    h, γ, counts = experimental_variogram(acc)
    fit = fit_exponential_variogram(h, γ, counts)
    return Dict{String,Any}(
        "model" => "exponential: gamma(h) = nugget + partial_sill * (1 - exp(-3h/range))",
        "space" => "log10(ohm.m)",
        "lag_m" => acc.lag,
        "n_bins" => fit.n_bins,
        "n_pairs" => sum(counts),
        "range_m" => rnd(fit.range_m, 2),
        "nugget" => rnd(fit.nugget, digits),
        "sill" => rnd(fit.sill, digits),
        "partial_sill" => rnd(fit.partial_sill, digits),
        "rmse" => rnd(fit.rmse, digits),
        "nugget_to_sill" => rnd(fit.sill > 0 ? fit.nugget / fit.sill : NaN, digits),
        "lag_centers_m" => [rnd(v, 2) for v in h],
        "gamma" => [rnd(v, digits) for v in γ],
        "pair_counts" => counts,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Vertical structure along boreholes
# ─────────────────────────────────────────────────────────────────────────────

"""
    HoleLog

Cleaned resistivity log of one borehole, ascending in depth.

# Units
`depth` m along hole, `rho` ohm·m, `logrho` log10(ohm·m), `spacing_m` m.
"""
struct HoleLog
    id::String
    depth::Vector{Float64}
    rho::Vector{Float64}
    logrho::Vector{Float64}
    spacing_m::Float64
end

"""
    median_spacing(coords, max_gap) -> Float64

Median increment of a sorted coordinate vector, ignoring gaps above `max_gap`
(hole restarts / line jumps).
"""
function median_spacing(coords::AbstractVector{Float64}, max_gap::Float64)::Float64
    length(coords) < 2 && return NaN
    d = Float64[]
    @inbounds for i in 2:length(coords)
        δ = coords[i] - coords[i-1]
        0 < δ <= max_gap && push!(d, δ)
    end
    return isempty(d) ? NaN : median(d)
end

"""
    read_borehole_resistivity(path, opts; hole_col, depth_col, value_col)
        -> (Vector{HoleLog}, CleanReport, Dict)

Stream a GTK petrophysics TXT mirror, extract the hole / depth / resistivity
columns by header name, filter outliers and group the survivors per hole.

The file is decoded as ISO-8859-1 (Finnish characters) and split on `^`, so
trailing empty fields cannot shift the column indices.
"""
function read_borehole_resistivity(path::AbstractString, opts::StatsOptions;
                                   hole_col::Vector{String}=["Tunnus", "HOLE_ID"],
                                   depth_col::Vector{String}=["Syvyys", "DEPTH"],
                                   value_col::Vector{String}=["LUO_R"])
    text = _latin1_string(path)
    header = String[]
    ih = 0; id_ = 0; iv = 0
    n_rows = 0
    holes = String[]; depths = Float64[]; rhos = Float64[]
    n_no_depth = 0

    for (lineno, raw) in enumerate(eachline(IOBuffer(text)))
        if lineno == 1
            header = [String(strip(f)) for f in split(raw, '^')]
            ih = _find_column(header, hole_col)
            id_ = _find_column(header, depth_col)
            iv = _find_column(header, value_col)
            (ih == 0 || id_ == 0) &&
                error("$(basename(path)): missing hole/depth column $(hole_col)/$(depth_col)")
            iv == 0 && error("$(basename(path)): missing resistivity column $(value_col)")
            continue
        end
        lineno <= GTK_HEADER_ROWS && continue
        isempty(strip(raw)) && continue
        fields = split(raw, '^')
        length(fields) < max(ih, id_, iv) && continue
        n_rows += 1
        hid = String(strip(fields[ih]))
        depth = parse_f64(fields[id_])
        if isempty(hid) || !isfinite(depth)
            n_no_depth += 1
            continue
        end
        push!(holes, hid)
        push!(depths, depth)
        push!(rhos, parse_f64(fields[iv]))
    end

    keep, report = resistivity_mask(rhos, opts)
    grouped = Dict{String,Vector{Tuple{Float64,Float64}}}()
    @inbounds for i in eachindex(holes)
        keep[i] || continue
        push!(get!(Vector{Tuple{Float64,Float64}}, grouped, holes[i]), (depths[i], rhos[i]))
    end

    logs = HoleLog[]
    for (hid, pairs) in grouped
        sort!(pairs; by=first)
        d = [p[1] for p in pairs]
        r = [p[2] for p in pairs]
        sp = median_spacing(d, opts.max_depth_gap_m)
        push!(logs, HoleLog(hid, d, r, log10.(r), isfinite(sp) ? sp : 0.5))
    end
    sort!(logs; by=l -> l.id)

    meta = Dict{String,Any}(
        "path" => path,
        "n_data_rows" => n_rows,
        "n_missing_hole_or_depth" => n_no_depth,
        "n_holes" => length(logs),
        "columns" => Dict("hole" => header[ih], "depth" => header[id_], "value" => header[iv]),
    )
    return logs, report, meta
end

"""
    _latin1_string(path) -> String

Read a GTK TXT export as ISO-8859-1 and return valid UTF-8.
"""
function _latin1_string(path::AbstractString)::String
    io = IOBuffer()
    @inbounds for b in read(path)
        write(io, Char(b))
    end
    return String(take!(io))
end

"""
    _find_column(header, candidates) -> Int

Case/format-insensitive column index, `0` when absent.
"""
function _find_column(header::AbstractVector{<:AbstractString},
                      candidates::AbstractVector{<:AbstractString})::Int
    norm = _norm_token.(header)
    for c in candidates
        i = findfirst(==(_norm_token(c)), norm)
        i === nothing || return i
    end
    return 0
end

"""
    moving_median(depth, values, window_m) -> Vector{Float64}

Depth-window moving median — removes single-sample core spikes before layers are
segmented, so thicknesses reflect geology rather than measurement noise.
"""
function moving_median(depth::Vector{Float64}, values::Vector{Float64},
                       window_m::Float64)::Vector{Float64}
    n = length(values)
    n == 0 && return Float64[]
    window_m <= 0 && return copy(values)
    half = window_m / 2
    out = Vector{Float64}(undef, n)
    lo = 1; hi = 1
    @inbounds for i in 1:n
        while lo < n && depth[lo] < depth[i] - half
            lo += 1
        end
        hi = max(hi, i)
        while hi < n && depth[hi+1] <= depth[i] + half
            hi += 1
        end
        out[i] = median(view(values, lo:hi))
    end
    return out
end

"""
    segment_layers(depth, logrho; log_split, max_gap_m, support_m) -> Vector{Float64}

Split a log into quasi-homogeneous layers and return their thicknesses (m).
A new layer starts where `|Δlog10 ρ| > log_split` (a contrast of `10^log_split`)
or where the depth gap exceeds `max_gap_m`. Each thickness is padded by
`support_m` — the sample spacing — so a single-sample layer is one sample thick
rather than zero.
"""
function segment_layers(depth::Vector{Float64}, logrho::Vector{Float64};
                        log_split::Float64=0.5, max_gap_m::Float64=5.0,
                        support_m::Float64=0.5)::Vector{Float64}
    n = length(depth)
    n == 0 && return Float64[]
    thick = Float64[]
    start = 1
    @inbounds for i in 2:n
        if abs(logrho[i] - logrho[i-1]) > log_split || (depth[i] - depth[i-1]) > max_gap_m
            push!(thick, depth[i-1] - depth[start] + support_m)
            start = i
        end
    end
    push!(thick, depth[n] - depth[start] + support_m)
    return thick
end

"""
    low_runs(values, coords, thresh; support_m) -> Vector{Float64}

Lengths of runs of consecutive samples below `thresh`, measured along `coords`
and padded by `support_m`. Used for conductive intervals down a hole and for
low-resistivity segments along a survey line.
"""
function low_runs(values::AbstractVector{Float64}, coords::AbstractVector{Float64},
                  thresh::Float64; support_m::Float64=0.0)::Vector{Float64}
    lengths = Float64[]
    n = length(values)
    i = 1
    while i <= n
        if isfinite(values[i]) && values[i] < thresh
            j = i
            while j < n && isfinite(values[j+1]) && values[j+1] < thresh
                j += 1
            end
            push!(lengths, coords[j] - coords[i] + support_m)
            i = j + 1
        else
            i += 1
        end
    end
    return lengths
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface (XYZ) data
# ─────────────────────────────────────────────────────────────────────────────

"""
    SurfaceLine

One survey line of a ground geophysics `.xyz` file, sorted along its dominant
axis.

# Fields
- `x, y`: metres (KKJ easting / northing)
- `value`: channel value in its native unit
- `along`: monotone along-line coordinate, m
- `spacing_m`: median station spacing, m
"""
struct SurfaceLine
    id::String
    x::Vector{Float64}
    y::Vector{Float64}
    value::Vector{Float64}
    along::Vector{Float64}
    spacing_m::Float64
end

"""
    SurfaceData

All accepted stations of one channel plus its per-line decomposition.
"""
struct SurfaceData
    x::Vector{Float64}
    y::Vector{Float64}
    value::Vector{Float64}
    lines::Vector{SurfaceLine}
    column::String
end

"""
    read_xyz_channel(dir, files, column_candidates; positive_only, opts)
        -> (SurfaceData, CleanReport, Dict)

Read GTK ground-geophysics `.xyz` files and pull out one channel.

Column position comes from the `/ X/m Y/m Station ...` comment header when
present, otherwise from `default_index`. `Line NNN` markers delimit survey lines.
`positive_only = true` applies the resistivity filter (and enables log10 use);
`false` keeps signed values such as Slingram `Re%`.
"""
function read_xyz_channel(dir::AbstractString, files::Vector{String},
                          column_candidates::Vector{String}, opts::StatsOptions;
                          positive_only::Bool=true, default_index::Int=4)
    xs = Float64[]; ys = Float64[]; vs = Float64[]; lids = String[]
    column_name = isempty(column_candidates) ? "col$(default_index)" : column_candidates[1]
    n_rows = 0
    present = isdir(dir)

    if present
        available = sort(filter(f -> endswith(lowercase(f), ".xyz"), readdir(dir)))
        wanted = isempty(files) ? available : filter(f -> f in available, files)
        isempty(wanted) && (wanted = available)
        for fname in wanted
            path = joinpath(dir, fname)
            isfile(path) || continue
            idx = default_index
            current = "unknown@" * fname
            for raw in eachline(IOBuffer(_latin1_string(path)))
                line = strip(raw)
                isempty(line) && continue
                if startswith(line, "/")
                    names = [String(t) for t in split(strip(line[2:end]))]
                    if length(names) >= 2 && _norm_token(names[1]) == "x" &&
                       _norm_token(names[2]) == "y"
                        found = _find_column(names, column_candidates)
                        found > 0 && (idx = found)
                        idx <= length(names) && (column_name = names[idx])
                    end
                    continue
                end
                if startswith(lowercase(line), "line")
                    current = strip(line[5:end]) * "@" * fname
                    continue
                end
                c = line[1]
                (isdigit(c) || c == '+' || c == '-' || c == '.') || continue
                tok = split(line)
                length(tok) < max(3, idx) && continue
                x = parse_f64(tok[1]); y = parse_f64(tok[2])
                (isfinite(x) && isfinite(y)) || continue
                n_rows += 1
                push!(xs, x); push!(ys, y)
                push!(vs, parse_f64(tok[idx]))
                push!(lids, current)
            end
        end
    end

    keep, report = positive_only ? resistivity_mask(vs, opts) : finite_mask(vs)
    kx = xs[keep]; ky = ys[keep]; kv = vs[keep]; kl = lids[keep]

    grouped = Dict{String,Vector{Int}}()
    for i in eachindex(kl)
        push!(get!(Vector{Int}, grouped, kl[i]), i)
    end
    lines = SurfaceLine[]
    for (lid, idxs) in grouped
        lx = kx[idxs]; ly = ky[idxs]; lv = kv[idxs]
        length(lv) < 2 && continue
        along = (length(lv) > 1 && std(ly) >= std(lx)) ? copy(ly) : copy(lx)
        perm = sortperm(along)
        along = along[perm]
        sp = median_spacing(along, 200.0)
        push!(lines, SurfaceLine(lid, lx[perm], ly[perm], lv[perm], along,
                                 isfinite(sp) ? sp : NaN))
    end
    sort!(lines; by=l -> l.id)

    meta = Dict{String,Any}(
        "path" => dir,
        "present" => present,
        "n_data_rows" => n_rows,
        "n_lines" => length(lines),
        "column" => column_name,
    )
    return SurfaceData(kx, ky, kv, lines, column_name), report, meta
end

# ─────────────────────────────────────────────────────────────────────────────
# Plan-view anomaly geometry
# ─────────────────────────────────────────────────────────────────────────────

"""
    survey_spacing(lines) -> (station_m, line_m)

Median station spacing along a line and median spacing *between* lines, both in
metres.

Line spacing is measured on the lines sharing the dominant orientation, using
their constant cross-line coordinate. It sets the true plan-view resolution: a
raster finer than `line_m` slices anomalies into one-cell-wide strips and
fabricates elongation, so it is the floor for the raster cell.
"""
function survey_spacing(lines::Vector{SurfaceLine})
    station = [l.spacing_m for l in lines if isfinite(l.spacing_m)]
    ns = Float64[]; ew = Float64[]
    for l in lines
        length(l.value) < 2 && continue
        if std(l.y) >= std(l.x)
            push!(ns, median(l.x))     # N–S line: constant easting
        else
            push!(ew, median(l.y))     # E–W line: constant northing
        end
    end
    dominant = length(ns) >= length(ew) ? ns : ew
    u = sort(unique(round.(dominant; digits=1)))
    gaps = filter(>(1.0e-6), diff(u))
    return (isempty(station) ? NaN : median(station),
            isempty(gaps) ? NaN : median(gaps))
end

"""
    PlanAnomaly

Geometry of one connected low-resistivity patch in plan view.

# Fields
- `area_m2`, `eq_diameter_m`: area and equal-area circle diameter
- `extent_x_m`, `extent_y_m`: axis-aligned bounding box
- `major_axis_m`, `minor_axis_m`: principal axes (4σ of the member cells,
  including the `cell²/12` variance of a cell)
- `aspect_ratio`: `major / minor` ≥ 1
- `azimuth_deg`: strike of the major axis, clockwise from grid north, `[0, 180)`
"""
struct PlanAnomaly
    n_cells::Int
    area_m2::Float64
    eq_diameter_m::Float64
    extent_x_m::Float64
    extent_y_m::Float64
    major_axis_m::Float64
    minor_axis_m::Float64
    aspect_ratio::Float64
    azimuth_deg::Float64
end

"""
    rasterize_log(x, y, value, cell) -> (grid, mask, x0, y0)

Cell-mean of `log10(value)` on a regular `cell`-metre raster. `mask` marks cells
that received at least one station.
"""
function rasterize_log(x::Vector{Float64}, y::Vector{Float64}, value::Vector{Float64},
                       cell::Float64)
    isempty(x) && return zeros(Float64, 0, 0), falses(0, 0), 0.0, 0.0
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    nx = max(1, floor(Int, (xmax - xmin) / cell) + 1)
    ny = max(1, floor(Int, (ymax - ymin) / cell) + 1)
    acc = zeros(Float64, nx, ny)
    cnt = zeros(Int, nx, ny)
    @inbounds for k in eachindex(x)
        i = clamp(floor(Int, (x[k] - xmin) / cell) + 1, 1, nx)
        j = clamp(floor(Int, (y[k] - ymin) / cell) + 1, 1, ny)
        acc[i, j] += log10(value[k])
        cnt[i, j] += 1
    end
    grid = fill(NaN, nx, ny)
    filled = falses(nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        cnt[i, j] == 0 && continue
        grid[i, j] = acc[i, j] / cnt[i, j]
        filled[i, j] = true
    end
    return grid, filled, xmin, ymin
end

"""
    plan_anomalies(grid, filled, cell; log_thresh, min_cells) -> Vector{PlanAnomaly}

4-connected components of `grid < log_thresh` and their geometry. Components
smaller than `min_cells` are discarded as station-level noise.
"""
function plan_anomalies(grid::Matrix{Float64}, filled::BitMatrix, cell::Float64;
                        log_thresh::Float64, min_cells::Int=2)::Vector{PlanAnomaly}
    nx, ny = size(grid)
    (nx == 0 || ny == 0) && return PlanAnomaly[]
    mask = falses(nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        mask[i, j] = filled[i, j] && grid[i, j] < log_thresh
    end

    seen = falses(nx, ny)
    out = PlanAnomaly[]
    stack = Tuple{Int,Int}[]
    cells_i = Int[]; cells_j = Int[]
    for j0 in 1:ny, i0 in 1:nx
        (mask[i0, j0] && !seen[i0, j0]) || continue
        empty!(stack); empty!(cells_i); empty!(cells_j)
        push!(stack, (i0, j0)); seen[i0, j0] = true
        while !isempty(stack)
            i, j = pop!(stack)
            push!(cells_i, i); push!(cells_j, j)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ni, nj = i + di, j + dj
                (1 <= ni <= nx && 1 <= nj <= ny) || continue
                (mask[ni, nj] && !seen[ni, nj]) || continue
                seen[ni, nj] = true
                push!(stack, (ni, nj))
            end
        end
        length(cells_i) < min_cells && continue
        push!(out, _component_geometry(cells_i, cells_j, cell))
    end
    return out
end

"""
    _component_geometry(cells_i, cells_j, cell) -> PlanAnomaly

Area, bounding box and principal axes of a set of raster cells. The principal
axes come from the closed-form eigen-decomposition of the 2×2 cell-centre
covariance, inflated by `cell²/12` so a single-cell patch has the cell's size.
"""
function _component_geometry(cells_i::Vector{Int}, cells_j::Vector{Int},
                             cell::Float64)::PlanAnomaly
    n = length(cells_i)
    area = n * cell * cell
    imin, imax = extrema(cells_i)
    jmin, jmax = extrema(cells_j)
    cx = [(i - 0.5) * cell for i in cells_i]
    cy = [(j - 0.5) * cell for j in cells_j]
    mx = mean(cx); my = mean(cy)
    disc = cell * cell / 12
    sxx = mean((cx .- mx) .^ 2) + disc
    syy = mean((cy .- my) .^ 2) + disc
    sxy = mean((cx .- mx) .* (cy .- my))

    tr = sxx + syy
    root = sqrt(max((sxx - syy)^2 + 4 * sxy^2, 0.0))
    λ1 = 0.5 * (tr + root)
    λ2 = max(0.5 * (tr - root), 0.0)
    major = 4 * sqrt(λ1)
    minor = 4 * sqrt(λ2)

    vx, vy = if abs(sxy) > 1.0e-12
        (λ1 - syy, sxy)
    else
        sxx >= syy ? (1.0, 0.0) : (0.0, 1.0)
    end
    az = mod(atand(vx, vy), 180.0)

    return PlanAnomaly(n, area, 2 * sqrt(area / π),
                       (imax - imin + 1) * cell, (jmax - jmin + 1) * cell,
                       major, max(minor, cell),
                       major / max(minor, cell), az)
end

"""
    anomaly_geometry_dict(anoms; digits=2) -> Dict{String,Any}

Distribution summaries of every `PlanAnomaly` field.
"""
function anomaly_geometry_dict(anoms::Vector{PlanAnomaly})::Dict{String,Any}
    return Dict{String,Any}(
        "n_anomalies" => length(anoms),
        "area_m2" => describe([a.area_m2 for a in anoms]; digits=0),
        "eq_diameter_m" => describe([a.eq_diameter_m for a in anoms]; digits=1),
        "extent_x_m" => describe([a.extent_x_m for a in anoms]; digits=1),
        "extent_y_m" => describe([a.extent_y_m for a in anoms]; digits=1),
        "major_axis_m" => describe([a.major_axis_m for a in anoms]; digits=1),
        "minor_axis_m" => describe([a.minor_axis_m for a in anoms]; digits=1),
        "aspect_ratio" => describe([a.aspect_ratio for a in anoms]; digits=3),
        "strike_azimuth_deg" => describe([a.azimuth_deg for a in anoms]; digits=1),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    _lognormal_params(x) -> (μ, σ)

Mean and standard deviation of `log(x)` over the positive samples, i.e. the
parameters a generator feeds to `exp(μ + σ·randn())`.
"""
function _lognormal_params(x::AbstractVector{<:Real})
    v = [log(Float64(t)) for t in x if isfinite(t) && t > 0]
    length(v) < 2 && return (NaN, NaN)
    return (mean(v), std(v))
end

"""
    _thickness_prior(x; digits=2) -> Dict{String,Any}

Thickness / size prior: percentiles plus lognormal parameters for direct sampling.
"""
function _thickness_prior(x::AbstractVector{<:Real}; digits::Int=2)::Dict{String,Any}
    μ, σ = _lognormal_params(x)
    d = describe(x; digits=digits)
    return Dict{String,Any}(
        "n" => d["n"],
        "p25" => d["p25"], "p50" => d["p50"], "p75" => d["p75"],
        "p05" => d["p05"], "p95" => d["p95"],
        "lognormal_mu" => rnd(μ, 4),
        "lognormal_sigma" => rnd(σ, 4),
        "sample" => "exp(lognormal_mu + lognormal_sigma * randn())",
    )
end

"""
    compute_priors(opts=StatsOptions()) -> Dict{String,Any}

Run the whole analysis and return the prior dictionary (the exact content of
`keivitsa_priors.json`).

# Returns
A `Dict{String,Any}` with the blocks `schema`, `project`, `units`, `options`,
`sources`, `statistics` and `generator_priors`. Non-finite numbers survive as
`NaN` here and become `null` in the serialised file.
"""
function compute_priors(opts::StatsOptions=StatsOptions())::Dict{String,Any}
    root = project_root(@__DIR__)
    cfg_path = opts.config_path === nothing ? default_config_path(root) :
               abspath(opts.config_path)
    cfg = load_config(cfg_path)
    rng = MersenneTwister(opts.seed)

    # ── borehole LUO_R ────────────────────────────────────────────────────────
    petro_yaml = string(lookup(cfg, "borehole.petrophysics.path";
                               default="database/3_DRILLINGS/Downhole_soundings_and_core_measurements/petroph.shp"))
    petro_path = resolve_table_path(root, petro_yaml)
    petro_path === nothing && error("Petrophysics table not found for $(petro_yaml)")
    luo_col = string(lookup(cfg, "borehole.petrophysics.columns.resistivity.name";
                            default="LUO_R"))
    @info "Loading borehole resistivity" file=basename(petro_path) column=luo_col
    logs, bh_report, bh_meta = read_borehole_resistivity(petro_path, opts;
                                                         value_col=[luo_col, "LUO_R"])
    bh_meta["n_records_declared"] = lookup(cfg, "borehole.petrophysics.n_records")

    rho_bh = reduce(vcat, (l.rho for l in logs); init=Float64[])
    log_bh = reduce(vcat, (l.logrho for l in logs); init=Float64[])
    @info "Borehole resistivity accepted" n=length(rho_bh) holes=length(logs)

    # ── VLF apparent resistivity ─────────────────────────────────────────────
    vlf_dir = joinpath(root, string(lookup(cfg, "ground_geophysics.vlf.path";
                                           default="database/5_GROUND_GEOPHYSICS/xyz/vlf/")))
    vlf_files = String[string(f) for f in
                       something(lookup(cfg, "ground_geophysics.vlf.files"), String[])]
    vlf, vlf_report, vlf_meta = read_xyz_channel(vlf_dir, vlf_files, ["ov"], opts;
                                                 positive_only=true, default_index=4)
    @info "VLF resistivity accepted" n=length(vlf.value) lines=length(vlf.lines)

    # ── Slingram in-phase (geometry only) ────────────────────────────────────
    sl_dir = joinpath(root, string(lookup(cfg, "ground_geophysics.slingram.path";
                                          default="database/5_GROUND_GEOPHYSICS/xyz/slingram/")))
    sl_files = String[string(f) for f in
                      something(lookup(cfg, "ground_geophysics.slingram.files"), String[])]
    slingram, sl_report, sl_meta = read_xyz_channel(sl_dir, sl_files, ["Re_pct", "Re"],
                                                    opts; positive_only=false,
                                                    default_index=4)
    sl_meta["unit"] = "percent"
    sl_meta["note"] = "In-phase EM response, not ohm.m: used for anomaly geometry only, never log10-scaled."
    if !sl_meta["present"]
        @warn "slingram_real is absent; horizontal geometry rests on VLF alone" path=sl_dir
    end

    # ── population separation ────────────────────────────────────────────────
    comps = fit_gmm1d(log_bh, opts.n_populations; rng=rng)
    pops = population_summary(comps, log_bh)
    # Data-driven conductor cutoff: where the conductive population stops winning.
    cond_boundary_ohm = length(comps) < 2 ? opts.conductor_ohm_m :
                        10.0^mixture_boundary(comps[1], comps[2])

    # ── vertical structure ───────────────────────────────────────────────────
    layers = Float64[]
    layers_raw = Float64[]
    cond_thick = Float64[]
    res_thick = Float64[]
    efold_m = Float64[]
    layers_per_hole = Float64[]
    cond_metres = 0.0
    total_metres = 0.0
    vacc = VariogramAccumulator(opts.variogram_lag_m, opts.variogram_nlags)

    for l in logs
        sm = moving_median(l.depth, l.logrho, opts.smooth_window_m)
        raw_t = segment_layers(l.depth, l.logrho; log_split=opts.log_split,
                               max_gap_m=opts.max_depth_gap_m, support_m=l.spacing_m)
        sm_t = segment_layers(l.depth, sm; log_split=opts.log_split,
                              max_gap_m=opts.max_depth_gap_m, support_m=l.spacing_m)
        append!(layers_raw, raw_t)
        geo = filter(>=(opts.min_layer_m), sm_t)
        append!(layers, geo)
        push!(layers_per_hole, Float64(length(geo)))

        rho_sm = 10.0 .^ sm
        append!(cond_thick, filter(>=(opts.min_layer_m),
                                   low_runs(rho_sm, l.depth, cond_boundary_ohm;
                                            support_m=l.spacing_m)))
        # Negating both series turns the "below threshold" scan into "above threshold".
        append!(res_thick, filter(>=(opts.min_layer_m),
                                  low_runs(-rho_sm, l.depth, -opts.resistor_ohm_m;
                                           support_m=l.spacing_m)))

        total_metres += l.depth[end] - l.depth[1]
        @inbounds for i in 1:(length(l.depth)-1)
            gap = l.depth[i+1] - l.depth[i]
            0 < gap <= opts.max_depth_gap_m || continue
            rho_sm[i] < cond_boundary_ohm && (cond_metres += gap)
        end

        accumulate_variogram!(vacc, l.depth, l.logrho)
        if length(l.logrho) >= 16
            lags, ρ = acf1d(l.logrho; maxlag=opts.acf_maxlag)
            lag = isempty(lags) ? NaN : efolding_length(lags, ρ)
            isfinite(lag) && isfinite(l.spacing_m) && push!(efold_m, lag * l.spacing_m)
        end
    end

    # ── horizontal continuity from VLF lines ─────────────────────────────────
    hacc = VariogramAccumulator(opts.vlf_lag_m, opts.vlf_nlags)
    vlf_efold = Float64[]
    vlf_spacings = Float64[]
    vlf_runs = Float64[]
    vlf_sorted = sort(vlf.value)
    vlf_cutoff = isempty(vlf_sorted) ? opts.conductor_ohm_m :
                 min(opts.conductor_ohm_m, q(vlf_sorted, 25))
    for l in vlf.lines
        isfinite(l.spacing_m) && push!(vlf_spacings, l.spacing_m)
        accumulate_variogram!(hacc, l.along, log10.(l.value))
        append!(vlf_runs, low_runs(l.value, l.along, vlf_cutoff;
                                   support_m=isfinite(l.spacing_m) ? l.spacing_m : 0.0))
        if length(l.value) >= 24 && isfinite(l.spacing_m)
            lags, ρ = acf1d(log10.(l.value); maxlag=opts.acf_maxlag)
            lag = isempty(lags) ? NaN : efolding_length(lags, ρ)
            isfinite(lag) && push!(vlf_efold, lag * l.spacing_m)
        end
    end

    # ── plan-view geometry ───────────────────────────────────────────────────
    station_m, line_m = survey_spacing(vlf.lines)
    cell = if opts.raster_cell_m !== nothing
        opts.raster_cell_m
    else
        maximum(filter(isfinite, [station_m, line_m, 10.0]))
    end
    if isfinite(line_m) && cell < line_m
        @warn "Raster cell is finer than the VLF line spacing; plan-view aspect ratios are aliased" cell_m=cell line_spacing_m=line_m
    end
    grid, filled, _, _ = rasterize_log(vlf.x, vlf.y, vlf.value, cell)
    anoms = plan_anomalies(grid, filled, cell;
                           log_thresh=log10(vlf_cutoff), min_cells=opts.min_anomaly_cells)
    @info "Plan-view anomalies" n=length(anoms) cell_m=cell cutoff_ohm_m=round(vlf_cutoff; digits=1)

    sl_geom = Dict{String,Any}("present" => sl_meta["present"])
    if sl_meta["present"] && !isempty(slingram.value)
        t90 = q(sort(abs.(slingram.value)), 90)
        sl_geom["threshold_abs_Re_p90_percent"] = rnd(t90, 3)
        sl_geom["values_percent"] = describe(slingram.value; digits=3)
        if t90 > 0
            # |Re| above p90 marks a conductor: invert so the low-value machinery applies.
            inv = [1.0 / max(abs(v), 1.0e-6) for v in slingram.value]
            sgrid, sfilled, _, _ = rasterize_log(slingram.x, slingram.y, inv, cell)
            sanoms = plan_anomalies(sgrid, sfilled, cell;
                                    log_thresh=log10(1 / t90),
                                    min_cells=opts.min_anomaly_cells)
            sl_geom["geometry"] = anomaly_geometry_dict(sanoms)
        end
    end

    # ── generator priors ─────────────────────────────────────────────────────
    cond_thick_p = _thickness_prior(cond_thick)
    eqd = [a.eq_diameter_m for a in anoms]
    vert_med = isempty(cond_thick) ? NaN : median(cond_thick)
    width_med = isempty(vlf_runs) ? NaN : median(vlf_runs)
    eqd_med = isempty(eqd) ? NaN : median(eqd)
    corr_z = isempty(efold_m) ? NaN : median(efold_m)
    corr_xy = isempty(vlf_efold) ? NaN : median(vlf_efold)
    vfit = fit_exponential_variogram(experimental_variogram(vacc)...)
    hfit = fit_exponential_variogram(experimental_variogram(hacc)...)

    buckets = [Float64[] for _ in comps]
    for v in log_bh
        isempty(comps) && break
        push!(buckets[assign_population(comps, v)], v)
    end
    bucket_of(j) = (isempty(comps) || j < 1 || j > length(buckets)) ?
                   Float64[] : sort(buckets[j])

    """Prior block for one population: mixture parameters + an empirical range."""
    function class_prior(c::Union{Nothing,GaussianComponent}, lo_p::Real, hi_p::Real,
                         sorted_bucket::Vector{Float64})
        lo = isempty(sorted_bucket) ? NaN : q(sorted_bucket, lo_p)
        hi = isempty(sorted_bucket) ? NaN : q(sorted_bucket, hi_p)
        return Dict{String,Any}(
            "n_assigned" => length(sorted_bucket),
            "log10_mean" => c === nothing ? NaN : rnd(c.mean, 3),
            "log10_std" => c === nothing ? NaN : rnd(c.std, 3),
            "weight" => c === nothing ? NaN : rnd(c.weight, 4),
            "log10_range" => [rnd(lo, 3), rnd(hi, 3)],
            "ohm_m_range" => [rnd(10.0^lo, 2), rnd(10.0^hi, 2)],
            "percentiles" => [lo_p, hi_p],
        )
    end

    log_sorted = sort(log_bh)
    generator = Dict{String,Any}(
        "note" => "Default sampling envelope for synthetic 3-D resistivity models of a Keivitsa-like mafic/ultramafic host with disseminated-to-massive sulfide bodies. Sample in log10(ohm.m), then exponentiate.",
        "space" => "log10(ohm.m)",
        "clip_ohm_m" => [rnd(max(opts.rho_range[1], 10.0^q(log_sorted, 0.1)), 3),
                         rnd(min(opts.rho_range[2], 10.0^q(log_sorted, 99.9)), 1)],
        "envelope_p05_p95_ohm_m" => [rnd(10.0^q(log_sorted, 5), 2),
                                     rnd(10.0^q(log_sorted, 95), 1)],
        "conductor_cutoff_ohm_m" => rnd(cond_boundary_ohm, 2),
        "layer_thickness_m" => _thickness_prior(layers),
        "layers_per_hole" => describe(layers_per_hole; digits=1),
        "anomaly" => Dict{String,Any}(
            "vertical_thickness_m" => cond_thick_p,
            "across_strike_width_m" => _thickness_prior(vlf_runs; digits=1),
            "plan_eq_diameter_m" => _thickness_prior(eqd; digits=1),
            "plan_major_axis_m" => describe([a.major_axis_m for a in anoms]; digits=1),
            "plan_minor_axis_m" => describe([a.minor_axis_m for a in anoms]; digits=1),
            "plan_aspect_ratio" => describe([a.aspect_ratio for a in anoms]; digits=3),
            "strike_azimuth_deg" => describe([a.azimuth_deg for a in anoms]; digits=1),
            "thickness_to_width_ratio" =>
                rnd(isfinite(vert_med) && isfinite(width_med) && width_med > 0 ?
                    vert_med / width_med : NaN, 4),
            "thickness_to_eq_diameter_ratio" =>
                rnd(isfinite(vert_med) && isfinite(eqd_med) && eqd_med > 0 ?
                    vert_med / eqd_med : NaN, 4),
            "volume_fraction_conductive" =>
                rnd(total_metres > 0 ? cond_metres / total_metres : NaN, 5),
            "resolution" => Dict{String,Any}(
                "vlf_station_spacing_m" => rnd(station_m, 1),
                "vlf_line_spacing_m" => rnd(line_m, 1),
                "raster_cell_m" => cell,
            ),
            "note" => join([
                "vertical_thickness_m: conductive runs down inclined holes, so an upper bound on true thickness.",
                "across_strike_width_m: low-resistivity runs along VLF lines, resolved at the station spacing — the tightest horizontal constraint.",
                "plan_*: VLF raster connected components, resolution-limited by the line spacing, so they overestimate small bodies.",
            ], " "),
        ),
        "correlation_length_m" => Dict{String,Any}(
            "vertical_acf_median" => rnd(corr_z, 2),
            "horizontal_acf_median" => rnd(corr_xy, 1),
            "vertical_variogram_range" => rnd(vfit.range_m, 2),
            "horizontal_variogram_range" => rnd(hfit.range_m, 1),
            "anisotropy_ratio" =>
                rnd(isfinite(vfit.range_m) && isfinite(hfit.range_m) && vfit.range_m > 0 ?
                    hfit.range_m / vfit.range_m : NaN, 3),
            "definition" => "vertical = along-hole, horizontal = along VLF line; ACF = e-folding lag, variogram = exponential practical range",
        ),
    )

    # One prior block per fitted population, named conductive → resistive. A host
    # window is deliberately tighter (p16–p84) than a body window (p05–p95).
    class_labels = population_labels(length(comps))
    class_windows = Dict("conductive_body" => (5, 95), "resistive_body" => (5, 95))
    for (j, label) in enumerate(class_labels)
        lo, hi = get(class_windows, label, (16, 84))
        generator[label] = class_prior(comps[j], lo, hi, bucket_of(j))
    end
    generator["population_order_conductive_to_resistive"] = class_labels

    labels, dec_counts, dec_frac = decade_occupancy(rho_bh,
        [1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
    vlf_labels, vlf_counts, vlf_frac = decade_occupancy(vlf.value,
        [1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
    generator["decade_weights"] = Dict{String,Any}(
        "bins_ohm_m" => labels,
        "borehole_fraction" => dec_frac,
        "vlf_fraction" => vlf_frac,
    )

    return Dict{String,Any}(
        "schema" => SCHEMA_VERSION,
        "generated_by" => Dict{String,Any}(
            "module" => "src/analysis/keivitsa_stats.jl",
            "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"),
            "julia_version" => string(VERSION),
            "config" => relpath(cfg_path, root),
        ),
        "project" => Dict{String,Any}(
            "name" => string(lookup(cfg, "project.name"; default="Keivitsa")),
            "crs_epsg" => something(lookup(cfg, "project.crs.epsg"), 2391),
        ),
        "units" => Dict{String,Any}(
            "resistivity" => "ohm.m",
            "log_resistivity" => "log10(ohm.m)",
            "length" => "m",
            "azimuth" => "deg clockwise from grid north",
        ),
        "options" => options_dict(opts),
        "sources" => Dict{String,Any}(
            "LUO_R" => merge(bh_meta, Dict{String,Any}(
                "path" => relpath(petro_path, root),
                "unit" => "ohm.m",
                "filter" => report_dict(bh_report))),
            "vlf_resistivity" => merge(vlf_meta, Dict{String,Any}(
                "path" => relpath(vlf_dir, root),
                "unit" => "ohm.m",
                "filter" => report_dict(vlf_report))),
            "slingram_real" => merge(sl_meta, Dict{String,Any}(
                "path" => relpath(sl_dir, root),
                "filter" => report_dict(sl_report))),
        ),
        "statistics" => Dict{String,Any}(
            "resistivity" => Dict{String,Any}(
                "borehole_LUO_R_ohm_m" => describe(rho_bh; digits=2),
                "borehole_LUO_R_log10" => describe(log_bh; digits=3),
                "vlf_ohm_m" => describe(vlf.value; digits=2),
                "vlf_log10" => describe(isempty(vlf.value) ? Float64[] :
                                        log10.(vlf.value); digits=3),
                "decades" => Dict{String,Any}(
                    "bins_ohm_m" => labels,
                    "borehole_counts" => dec_counts,
                    "borehole_fraction" => dec_frac,
                    "vlf_counts" => vlf_counts,
                    "vlf_fraction" => vlf_frac,
                ),
            ),
            "populations_borehole" => pops,
            "vertical_structure" => Dict{String,Any}(
                "layer_definition" => "|dlog10(rho)| > $(opts.log_split) after a $(opts.smooth_window_m) m moving median, or a depth gap > $(opts.max_depth_gap_m) m",
                "layer_thickness_m" => describe(layers; digits=2),
                "layer_thickness_unfiltered_m" => describe(layers_raw; digits=2),
                "layers_per_hole" => describe(layers_per_hole; digits=1),
                "conductive_interval_m" => describe(cond_thick; digits=2),
                "resistive_interval_m" => describe(res_thick; digits=2),
                "conductive_metre_fraction" =>
                    rnd(total_metres > 0 ? cond_metres / total_metres : NaN, 5),
                "logged_metres" => rnd(total_metres, 1),
                "acf_efolding_m" => describe(efold_m; digits=2),
                "variogram" => variogram_dict(vacc),
            ),
            "horizontal_structure" => Dict{String,Any}(
                "vlf_station_spacing_m" => describe(vlf_spacings; digits=2),
                "vlf_conductor_cutoff_ohm_m" => rnd(vlf_cutoff, 2),
                "vlf_low_run_length_m" => describe(vlf_runs; digits=1),
                "acf_efolding_m" => describe(vlf_efold; digits=1),
                "variogram" => variogram_dict(hacc),
            ),
            "plan_geometry" => Dict{String,Any}(
                "raster_cell_m" => cell,
                "raster_cell_source" => opts.raster_cell_m === nothing ?
                                        "auto: max(station spacing, line spacing)" : "explicit",
                "vlf_station_spacing_m" => rnd(station_m, 2),
                "vlf_line_spacing_m" => rnd(line_m, 2),
                "cutoff_ohm_m" => rnd(vlf_cutoff, 2),
                "source" => "VLF ov, cell-mean log10 raster, 4-connected components",
                "resolution_note" => "Anomaly dimensions below the line spacing are unresolved; aspect ratio and strike azimuth are only meaningful for patches several cells across.",
                "vlf" => anomaly_geometry_dict(anoms),
                "slingram" => sl_geom,
            ),
        ),
        "generator_priors" => generator,
    )
end

"""
    options_dict(opts) -> Dict{String,Any}

Echo the run configuration so a prior file documents its own provenance.
"""
function options_dict(opts::StatsOptions)::Dict{String,Any}
    d = Dict{String,Any}()
    for f in fieldnames(StatsOptions)
        v = getfield(opts, f)
        d[String(f)] = v isa Tuple ? collect(v) : (v === nothing ? nothing : v)
    end
    return d
end

# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

"""
    _sanitize(x) -> Any

Recursively replace non-finite floats with `nothing` (JSON `null`) and coerce
tuples to arrays, so the artefact is valid strict JSON.
"""
function _sanitize(x)
    if x isa AbstractDict
        return Dict{String,Any}(string(k) => _sanitize(v) for (k, v) in x)
    elseif x isa Tuple
        return Any[_sanitize(v) for v in x]
    elseif x isa AbstractVector
        return Any[_sanitize(v) for v in x]
    elseif x isa AbstractFloat
        return isfinite(x) ? Float64(x) : nothing
    else
        return x
    end
end

"""
    write_priors(priors, path; yaml_path=nothing) -> String

Write the prior dictionary as pretty-printed JSON (and optionally as a YAML
mirror). Non-finite numbers are serialised as `null`. Returns the JSON path.
"""
function write_priors(priors::AbstractDict, path::AbstractString;
                      yaml_path::Union{Nothing,AbstractString}=nothing)::String
    clean = _sanitize(priors)
    mkpath(dirname(abspath(path)))
    open(abspath(path), "w") do io
        JSON3.pretty(io, JSON3.write(clean))
        println(io)
    end
    if yaml_path !== nothing
        mkpath(dirname(abspath(yaml_path)))
        open(abspath(yaml_path), "w") do io
            println(io, "# Keivitsa synthetic-model priors — generated by src/analysis/keivitsa_stats.jl")
            println(io, "# Do not edit by hand; regenerate with scripts/build_keivitsa_priors.jl")
            YAML.write(io, clean)
        end
    end
    return abspath(path)
end

"""
    load_priors(path) -> Dict{String,Any}

Read `keivitsa_priors.json` back into a string-keyed dictionary — the entry point
for the synthetic model generator.
"""
function load_priors(path::AbstractString)::Dict{String,Any}
    return JSON3.read(read(String(path), String), Dict{String,Any})
end

"""
    run_stats(; kwargs...) -> Dict{String,Any}

Compute the priors, write `config/keivitsa_priors.json` (plus a YAML mirror when
`output_yaml` is set) and log a short console report.

Keyword arguments are `StatsOptions` fields, e.g.
`run_stats(n_populations=2, raster_cell_m=50.0)`.
"""
function run_stats(; kwargs...)::Dict{String,Any}
    opts = StatsOptions(; kwargs...)
    root = project_root(@__DIR__)
    priors = compute_priors(opts)
    json_path = opts.output_json === nothing ?
                joinpath(root, "config", "keivitsa_priors.json") : opts.output_json
    written = write_priors(priors, json_path; yaml_path=opts.output_yaml)
    print_report(priors)
    @info "Priors written" path=relpath(written, root)
    return priors
end

"""
    print_report(priors)

Human-readable digest of the numbers a modeller checks first.
"""
function print_report(priors::AbstractDict)
    g = priors["generator_priors"]
    st = priors["statistics"]
    res = st["resistivity"]["borehole_LUO_R_ohm_m"]
    println("\n═══ Keivitsa realism parameters ═══")
    @printf("  LUO_R  n=%d   min=%.3g  p05=%.3g  median=%.3g  p95=%.3g  max=%.3g ohm.m\n",
            res["n"], res["min"], res["p05"], res["p50"], res["p95"], res["max"])
    for cls in ("conductive_body", "host_rock", "resistive_body")
        c = get(g, cls, nothing)
        (c isa AbstractDict && isfinite(c["log10_mean"])) || continue
        @printf("  %-16s log10 mu=%.2f sigma=%.2f  w=%.3f  n=%d  range=%s ohm.m\n",
                cls, c["log10_mean"], c["log10_std"], c["weight"],
                c["n_assigned"], c["ohm_m_range"])
    end
    lt = g["layer_thickness_m"]
    @printf("  layer thickness   p25/p50/p75 = %.2f / %.2f / %.2f m (n=%d)\n",
            lt["p25"], lt["p50"], lt["p75"], lt["n"])
    an = g["anomaly"]
    vt = an["vertical_thickness_m"]
    @printf("  conductor dz      p25/p50/p75 = %.2f / %.2f / %.2f m (n=%d)\n",
            vt["p25"], vt["p50"], vt["p75"], vt["n"])
    for (label, key) in (("across-strike width", "across_strike_width_m"),
                         ("plan eq. diameter", "plan_eq_diameter_m"))
        d = an[key]
        d["n"] > 0 || continue
        @printf("  %-17s p25/p50/p75 = %.1f / %.1f / %.1f m (n=%d)\n",
                label, d["p25"], d["p50"], d["p75"], d["n"])
    end
    ar = an["plan_aspect_ratio"]
    ar["n"] > 0 && @printf("  plan aspect ratio p50=%.2f   strike p50=%.0f deg\n",
                           ar["p50"], an["strike_azimuth_deg"]["p50"])
    cl = g["correlation_length_m"]
    @printf("  correlation length  vertical=%s m   horizontal=%s m\n",
            cl["vertical_variogram_range"], cl["horizontal_variogram_range"])
    println("  conductive metre fraction: ", an["volume_fraction_conductive"])
end

end # module KeivitsaStats
