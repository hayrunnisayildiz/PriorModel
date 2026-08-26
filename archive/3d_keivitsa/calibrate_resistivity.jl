#!/usr/bin/env julia
#=
Keivitsa resistivity "realism" calibration.

Sources:
  - LUO_R borehole resistivity (petro.txt, ohm·m)
  - VLF-R apparent resistivity ov (xyz, ohm·m)
  - slingram_real: expected under database/5_GROUND_GEOPHYSICS/xyz/slingram/
    (percent, used for anomaly *size* only — not ohm·m)

Writes:
  config/keivitsa_resistivity_calibration.yaml
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics
using Printf
using YAML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PETRO = joinpath(ROOT, "database/3_DRILLINGS/Downhole_soundings_and_core_measurements/petro.txt")
const VLF_DIR = joinpath(ROOT, "database/5_GROUND_GEOPHYSICS/xyz/vlf")
const SLINGRAM_DIR = joinpath(ROOT, "database/5_GROUND_GEOPHYSICS/xyz/slingram")
const YAML_OUT = joinpath(ROOT, "config/keivitsa_resistivity_calibration.yaml")

const LOG_SPLIT = 0.30          # raw sample-to-sample split (noise-scale)
const LOG_SPLIT_SMOOTH = 0.50   # after 3 m median smooth ≈ factor 3
const SMOOTH_M = 3.0            # borehole moving-median window (m)
const MIN_LAYER_M = 2.0         # drop sub-resolution layers
const MIN_COND_M = 2.0          # drop sub-resolution conductive hits
const MIN_ANOM_XY_M = 20.0      # drop VLF runs shorter than ~1 station group
const COND_OHM = 100.0          # geological conductor cutoff (ohm·m)
const HOST_OHM = 1000.0         # fresh-host cutoff (ohm·m)
const GRID_M = 25.0             # VLF 2-D component raster (m)
const MAX_LAG = 80              # autocorrelation lags

# ── helpers ──────────────────────────────────────────────────────────────────

function parse_f64(s::AbstractString)::Float64
    t = strip(s)
    (isempty(t) || t == "-" || t == "*") && return NaN
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

function pct(xs::AbstractVector{<:Real}, p::Real)::Float64
    isempty(xs) && return NaN
    return Float64(quantile(xs, p / 100))
end

function summarize(xs::AbstractVector{<:Real})::Dict{String,Any}
    isempty(xs) && return Dict{String,Any}("n" => 0)
    s = sort(Float64.(xs))
    return Dict{String,Any}(
        "n" => length(s),
        "min" => s[1],
        "p05" => pct(s, 5),
        "p10" => pct(s, 10),
        "p16" => pct(s, 16),
        "p25" => pct(s, 25),
        "p50" => pct(s, 50),
        "p75" => pct(s, 75),
        "p84" => pct(s, 84),
        "p90" => pct(s, 90),
        "p95" => pct(s, 95),
        "p99" => pct(s, 99),
        "max" => s[end],
        "mean" => mean(s),
        "std" => std(s),
        "geometric_mean" => all(>(0), s) ? exp(mean(log.(s))) : NaN,
    )
end

function log_hist_counts(xs::AbstractVector{<:Real}, edges::Vector{Float64})
    counts = zeros(Int, length(edges) - 1)
    for v in xs
        for i in 1:(length(edges) - 1)
            if v >= edges[i] && (i == length(edges) - 1 ? v <= edges[i + 1] : v < edges[i + 1])
                counts[i] += 1
                break
            end
        end
    end
    return counts
end

"""Autocorrelation of a 1-D series (mean-centred), lags 0:maxlag. Returns (lags, ρ)."""
function acf1d(x::Vector{Float64}; maxlag::Int=MAX_LAG)
    n = length(x)
    n < 8 && return Int[], Float64[]
    μ = mean(x)
    y = x .- μ
    var0 = sum(abs2, y)
    var0 <= 0 && return Int[], Float64[]
    maxl = min(maxlag, n - 2)
    lags = collect(0:maxl)
    ρ = Vector{Float64}(undef, maxl + 1)
    @inbounds for k in 0:maxl
        s = 0.0
        m = n - k
        for i in 1:m
            s += y[i] * y[i + k]
        end
        ρ[k + 1] = s / var0
    end
    return lags, ρ
end

"""First lag where ρ drops below 1/e (linear interpolate). Units = sample spacings."""
function efolding_lag(lags::Vector{Int}, ρ::Vector{Float64})::Float64
    length(ρ) < 2 && return NaN
    target = 1 / ℯ
    for i in 2:length(ρ)
        if ρ[i] <= target
            r0, r1 = ρ[i - 1], ρ[i]
            l0, l1 = Float64(lags[i - 1]), Float64(lags[i])
            (r0 == r1) && return l0
            return l0 + (target - r0) / (r1 - r0) * (l1 - l0)
        end
    end
    return Float64(lags[end])
end

"""Runs of consecutive samples below `thresh` (already sorted along the axis)."""
function low_runs(values::Vector{Float64}, coords::Vector{Float64}, thresh::Float64)
    lengths = Float64[]
    n = length(values)
    i = 1
    while i <= n
        if values[i] < thresh && isfinite(values[i])
            j = i
            while j <= n && isfinite(values[j]) && values[j] < thresh
                j += 1
            end
            # length along coordinate; single-station hits use median spacing
            if j - 1 > i
                push!(lengths, abs(coords[j - 1] - coords[i]))
            else
                # isolated hit — use local station spacing if available
                sp = if i < n
                    abs(coords[i + 1] - coords[i])
                elseif i > 1
                    abs(coords[i] - coords[i - 1])
                else
                    0.0
                end
                push!(lengths, sp)
            end
            i = j
        else
            i += 1
        end
    end
    return lengths
end

# ── LUO_R ────────────────────────────────────────────────────────────────────

struct PetroSample
    hole::String
    depth::Float64
    rho::Float64
end

function load_luo_r(path::AbstractString)::Vector{PetroSample}
    samples = PetroSample[]
    open(path, "r") do io
        for (lineno, raw) in enumerate(eachline(io))
            lineno <= 4 && continue
            line = strip(raw)
            isempty(line) && continue
            parts = split(line, " ^ ")
            length(parts) < 17 && continue
            hole = strip(parts[3])
            depth = parse_f64(parts[4])
            rho = parse_f64(parts[17])
            (isempty(hole) || !isfinite(depth) || !isfinite(rho) || rho <= 0) && continue
            push!(samples, PetroSample(hole, depth, rho))
        end
    end
    return samples
end

"""Segment a hole into layers where |Δlog10 ρ| stays ≤ `split`."""
function layer_thicknesses(hole_samples::Vector{PetroSample}; split::Float64=LOG_SPLIT)
    isempty(hole_samples) && return Float64[]
    s = sort(hole_samples; by=x -> x.depth)
    thick = Float64[]
    start = 1
    for i in 2:length(s)
        dlog = abs(log10(s[i].rho) - log10(s[i - 1].rho))
        gap = s[i].depth - s[i - 1].depth
        if dlog > split || gap > 5.0
            th = s[i - 1].depth - s[start].depth
            th > 0 && push!(thick, th)
            start = i
        end
    end
    th = s[end].depth - s[start].depth
    th > 0 && push!(thick, th)
    return thick
end

function conductive_interval_thicknesses(hole_samples::Vector{PetroSample}, thresh::Float64)
    isempty(hole_samples) && return Float64[]
    s = sort(hole_samples; by=x -> x.depth)
    return low_runs([x.rho for x in s], [x.depth for x in s], thresh)
end

"""Depth-window median of log10 ρ (window in metres)."""
function smooth_hole(hs::Vector{PetroSample}; window_m::Float64=SMOOTH_M)
    s = sort(hs; by=x -> x.depth)
    n = length(s)
    n == 0 && return s
    half = window_m / 2
    out = Vector{PetroSample}(undef, n)
    @inbounds for i in 1:n
        logs = Float64[]
        di = s[i].depth
        for j in 1:n
            abs(s[j].depth - di) <= half && push!(logs, log10(s[j].rho))
        end
        ρs = 10.0 ^ median(logs)
        out[i] = PetroSample(s[i].hole, di, ρs)
    end
    return out
end

keep_min(xs::Vector{Float64}, lo::Float64) = filter(>=(lo), xs)

# ── VLF / slingram XYZ ───────────────────────────────────────────────────────

struct XyzPoint
    x::Float64
    y::Float64
    v::Float64
    line::String
end

function load_vlf_files(dir::AbstractString)::Vector{XyzPoint}
    pts = XyzPoint[]
    isdir(dir) || return pts
    current_line = "unknown"
    for fname in sort(readdir(dir))
        endswith(lowercase(fname), ".xyz") || continue
        path = joinpath(dir, fname)
        open(path, "r") do io
            for raw in eachline(io)
                line = strip(raw)
                isempty(line) && continue
                if startswith(line, "Line")
                    current_line = strip(line[5:end]) * "@" * fname
                    continue
                end
                startswith(line, "/") && continue
                c = line[1]
                (c == '+' || c == '-' || c == '.' || isdigit(c)) || continue
                tok = split(line)
                length(tok) < 5 && continue
                x = parse_f64(tok[1])
                y = parse_f64(tok[2])
                ov = parse_f64(tok[4])   # ov/ohmi-m
                (!isfinite(x) || !isfinite(y) || !isfinite(ov) || ov <= 0) && continue
                push!(pts, XyzPoint(x, y, ov, current_line))
            end
        end
        current_line = "unknown"
    end
    return pts
end

function load_slingram_files(dir::AbstractString)
    pts = XyzPoint[]
    isdir(dir) || return pts, false
    files = filter(f -> endswith(lowercase(f), ".xyz"), readdir(dir))
    isempty(files) && return pts, false
    current_line = "unknown"
    for fname in sort(files)
        path = joinpath(dir, fname)
        open(path, "r") do io
            header = String[]
            for raw in eachline(io)
                line = strip(raw)
                isempty(line) && continue
                if startswith(line, "/")
                    s = strip(line[2:end])
                    tokens = split(s)
                    if length(tokens) >= 2 && lowercase(split(tokens[1], '/'; limit=2)[1]) == "x"
                        header = [split(t, '/'; limit=2)[1] for t in tokens]
                    end
                    continue
                end
                if startswith(line, "Line")
                    current_line = strip(line[5:end]) * "@" * fname
                    continue
                end
                c = line[1]
                (c == '+' || c == '-' || c == '.' || isdigit(c)) || continue
                tok = split(line)
                length(tok) < 4 && continue
                x = parse_f64(tok[1])
                y = parse_f64(tok[2])
                # Re is first extra after Station when header is X Y Station Re Im
                re_idx = 4
                if !isempty(header)
                    for (i, h) in enumerate(header)
                        hl = lowercase(h)
                        if startswith(hl, "re")
                            re_idx = i
                            break
                        end
                    end
                end
                re_idx > length(tok) && continue
                re = parse_f64(tok[re_idx])
                (!isfinite(x) || !isfinite(y) || !isfinite(re)) && continue
                push!(pts, XyzPoint(x, y, re, current_line))
            end
        end
        current_line = "unknown"
    end
    return pts, true
end

function group_by_line(pts::Vector{XyzPoint})
    g = Dict{String,Vector{XyzPoint}}()
    for p in pts
        push!(get!(Vector{XyzPoint}, g, p.line), p)
    end
    return g
end

"""Dominant along-line coordinate (N-S lines vary in Y)."""
function sort_along_line!(pts::Vector{XyzPoint})
    isempty(pts) && return pts
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    if std(ys) >= std(xs)
        sort!(pts; by=p -> p.y)
    else
        sort!(pts; by=p -> p.x)
    end
    return pts
end

function along_line_coords(pts::Vector{XyzPoint})
    isempty(pts) && return Float64[]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    if std(ys) >= std(xs)
        return ys
    else
        return xs
    end
end

function station_spacing(pts::Vector{XyzPoint})::Float64
    length(pts) < 2 && return NaN
    coords = along_line_coords(pts)
    d = Float64[]
    for i in 2:length(coords)
        δ = abs(coords[i] - coords[i - 1])
        0 < δ < 100 && push!(d, δ)   # ignore line jumps
    end
    return isempty(d) ? NaN : median(d)
end

# 4-connected components on a binary grid
function connected_sizes(mask::BitMatrix, dx::Float64, dy::Float64)
    nx, ny = size(mask)
    seen = falses(nx, ny)
    areas = Float64[]
    eq_diam = Float64[]
    lx = Float64[]
    ly = Float64[]
    stack = Tuple{Int,Int}[]
    for j in 1:ny, i in 1:nx
        (mask[i, j] && !seen[i, j]) || continue
        empty!(stack)
        push!(stack, (i, j))
        seen[i, j] = true
        imin = i; imax = i; jmin = j; jmax = j; ncell = 0
        while !isempty(stack)
            ci, cj = pop!(stack)
            ncell += 1
            imin = min(imin, ci); imax = max(imax, ci)
            jmin = min(jmin, cj); jmax = max(jmax, cj)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ni, nj = ci + di, cj + dj
                (1 <= ni <= nx && 1 <= nj <= ny) || continue
                (mask[ni, nj] && !seen[ni, nj]) || continue
                seen[ni, nj] = true
                push!(stack, (ni, nj))
            end
        end
        area = ncell * dx * dy
        push!(areas, area)
        push!(eq_diam, 2 * sqrt(area / π))
        push!(lx, (imax - imin + 1) * dx)
        push!(ly, (jmax - jmin + 1) * dy)
    end
    return areas, eq_diam, lx, ly
end

function raster_low_components(pts::Vector{XyzPoint}, thresh::Float64; cell::Float64=GRID_M)
    isempty(pts) && return Float64[], Float64[], Float64[], Float64[]
    xmin, xmax = extrema(p.x for p in pts)
    ymin, ymax = extrema(p.y for p in pts)
    nx = max(1, Int(floor((xmax - xmin) / cell)) + 1)
    ny = max(1, Int(floor((ymax - ymin) / cell)) + 1)
    acc = zeros(Float64, nx, ny)
    cnt = zeros(Int, nx, ny)
    for p in pts
        i = clamp(Int(floor((p.x - xmin) / cell)) + 1, 1, nx)
        j = clamp(Int(floor((p.y - ymin) / cell)) + 1, 1, ny)
        acc[i, j] += log10(p.v)
        cnt[i, j] += 1
    end
    mask = falses(nx, ny)
    logt = log10(thresh)
    for j in 1:ny, i in 1:nx
        cnt[i, j] == 0 && continue
        mask[i, j] = (acc[i, j] / cnt[i, j]) < logt
    end
    return connected_sizes(mask, cell, cell)
end

function round_num(x; digits=3)
    isnan(x) && return nothing
    return round(Float64(x); digits=digits)
end

function round_summary(s::Dict{String,Any}; digits=3)
    out = Dict{String,Any}()
    for (k, v) in s
        if v isa Number
            out[k] = k == "n" ? Int(v) : round_num(v; digits=digits)
        else
            out[k] = v
        end
    end
    return out
end

# ── main ─────────────────────────────────────────────────────────────────────

function main()
    println("Loading LUO_R from ", PETRO)
    petro = load_luo_r(PETRO)
    rho_bh = [s.rho for s in petro]
    log_bh = log10.(rho_bh)
    println("  LUO_R finite > 0: ", length(rho_bh), " / expected ~62976 records")

    holes = Dict{String,Vector{PetroSample}}()
    for s in petro
        push!(get!(Vector{PetroSample}, holes, s.hole), s)
    end

    layers_raw = Float64[]
    layers_sm = Float64[]
    cond_raw = Float64[]
    cond_sm = Float64[]
    efold_m = Float64[]
    cond_metre = 0.0
    total_metre = 0.0
    for (_, hs) in holes
        append!(layers_raw, layer_thicknesses(hs; split=LOG_SPLIT))
        append!(cond_raw, conductive_interval_thicknesses(hs, COND_OHM))
        sm = smooth_hole(hs)
        append!(layers_sm, layer_thicknesses(sm; split=LOG_SPLIT_SMOOTH))
        append!(cond_sm, conductive_interval_thicknesses(sm, COND_OHM))
        hs_s = sort(hs; by=x -> x.depth)
        if length(hs_s) >= 2
            total_metre += hs_s[end].depth - hs_s[1].depth
            for i in 1:(length(hs_s) - 1)
                gap = hs_s[i + 1].depth - hs_s[i].depth
                0 < gap <= 5 || continue
                (hs_s[i].rho < COND_OHM) && (cond_metre += gap)
            end
        end
        if length(hs_s) >= 16
            lags, ρacf = acf1d(log10.([x.rho for x in hs_s]))
            lag = efolding_lag(lags, ρacf)
            if isfinite(lag)
                dd = diff([x.depth for x in hs_s])
                med_dz = median(filter(z -> 0 < z < 5, dd))
                isfinite(med_dz) && push!(efold_m, lag * med_dz)
            end
        end
    end
    layers_geo = keep_min(layers_sm, MIN_LAYER_M)
    cond_geo = keep_min(cond_sm, MIN_COND_M)

    println("Loading VLF from ", VLF_DIR)
    vlf = load_vlf_files(VLF_DIR)
    rho_vlf = [p.v for p in vlf]
    println("  VLF ov finite > 0: ", length(rho_vlf))

    slingram, slingram_present = load_slingram_files(SLINGRAM_DIR)
    println("  slingram present: ", slingram_present, "  n=", length(slingram))

    p25_vlf = pct(rho_vlf, 25)
    p10_vlf = pct(rho_vlf, 10)
    cond_vlf = min(COND_OHM, p25_vlf)

    vlf_line_len = Float64[]
    vlf_line_len_p10 = Float64[]
    vlf_spacings = Float64[]
    vlf_efold = Float64[]
    for (_, lp) in group_by_line(vlf)
        length(lp) < 8 && continue
        sort_along_line!(lp)
        sp = station_spacing(lp)
        isfinite(sp) && push!(vlf_spacings, sp)
        coords = along_line_coords(lp)
        vals = [p.v for p in lp]
        append!(vlf_line_len, low_runs(vals, coords, cond_vlf))
        append!(vlf_line_len_p10, low_runs(vals, coords, p10_vlf))
        if length(lp) >= 24
            lags, ρ = acf1d(log10.(vals))
            lag = efolding_lag(lags, ρ)
            if isfinite(lag) && isfinite(sp)
                push!(vlf_efold, lag * sp)
            end
        end
    end
    vlf_line_geo = keep_min(vlf_line_len, MIN_ANOM_XY_M)
    vlf_eqd_geo = keep_min(eqd, MIN_ANOM_XY_M)

    areas, eqd, lx, ly = raster_low_components(vlf, cond_vlf)
    areas10, eqd10, lx10, ly10 = raster_low_components(vlf, p10_vlf)

    slingram_line_len = Float64[]
    slingram_eqd = Float64[]
    slingram_stats = Dict{String,Any}("present" => slingram_present, "n" => length(slingram),
                                      "unit" => "percent", "note" =>
                                      "Slingram Re is an in-phase EM response (%), not ohm·m. Use for anomaly size only.")
    if slingram_present && !isempty(slingram)
        re = [p.v for p in slingram]
        slingram_stats["values"] = round_summary(summarize(re))
        # strong in-phase |Re| anomalies: |Re| above p90 of |Re|
        absre = abs.(re)
        t90 = pct(absre, 90)
        for (_, lp) in group_by_line(slingram)
            length(lp) < 8 && continue
            sort_along_line!(lp)
            coords = along_line_coords(lp)
            # treat |Re| as "anomaly strength"; invert so low_runs can be reused
            # by mapping high |Re| → low dummy
            dummy = [t90 / max(abs(p.v), 1e-6) for p in lp]
            append!(slingram_line_len, low_runs(dummy, coords, 1.0))
        end
        # raster |Re| > p90
        hi = [XyzPoint(p.x, p.y, 1.0 / max(abs(p.v), 1e-6), p.line) for p in slingram]
        _, seqd, _, _ = raster_low_components(hi, 1.0 / t90)
        slingram_eqd = seqd
        slingram_stats["anomaly_along_line_m"] = round_summary(summarize(slingram_line_len))
        slingram_stats["anomaly_eq_diameter_m"] = round_summary(summarize(slingram_eqd))
        slingram_stats["anomaly_threshold_abs_Re_p90"] = round_num(t90)
    else
        slingram_stats["missing_path"] = SLINGRAM_DIR
        slingram_stats["files_found"] = false
    end

    # log-decade occupancy for generator bins
    decade_edges = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    decade_labels = ["0.1–1", "1–10", "10–100", "100–1e3", "1e3–1e4", "1e4–1e5", "1e5–1e6"]
    bh_dec = log_hist_counts(rho_bh, decade_edges)
    vlf_dec = log_hist_counts(rho_vlf, decade_edges)
    bh_n = length(rho_bh)
    vlf_n = length(rho_vlf)

    # recommended generator window: borehole p10–p90 clipped to geological sanity,
    # host = p50–p84, conductor = p05–p25 ∩ < 100 ohm·m
    rec_rho_min = max(1.0, pct(rho_bh, 5))
    rec_rho_max = min(1.0e5, pct(rho_bh, 95))
    rec_host = (pct(rho_bh, 50), pct(rho_bh, 84))
    rec_cond = (max(1.0, pct(rho_bh, 5)), min(COND_OHM, pct(rho_bh, 25)))
    rec_layer = (pct(layers, 25), pct(layers, 50), pct(layers, 75))
    rec_anom_z = (pct(cond_int, 25), pct(cond_int, 50), pct(cond_int, 75))
    rec_anom_xy = (pct(eqd, 25), pct(eqd, 50), pct(eqd, 75))
    rec_corr_z = isempty(efold_m) ? NaN : median(efold_m)
    rec_corr_xy = isempty(vlf_efold) ? NaN : median(vlf_efold)

    calib = Dict{String,Any}(
        "project" => "Keivitsa",
        "crs" => "EPSG:2391",
        "units" => Dict("resistivity" => "ohm.m", "length" => "m"),
        "sources" => Dict(
            "LUO_R" => Dict("path" => relpath(PETRO, ROOT), "n_records_file" => 62976,
                            "n_finite_positive" => length(rho_bh), "n_holes" => length(holes),
                            "unit" => "ohm.m"),
            "vlf_resistivity" => Dict("path" => relpath(VLF_DIR, ROOT),
                                      "n_finite_positive" => length(rho_vlf),
                                      "unit" => "ohm.m", "column" => "ov"),
            "slingram_real" => slingram_stats,
        ),
        "resistivity" => Dict(
            "borehole_LUO_R" => round_summary(summarize(rho_bh)),
            "vlf_ov" => round_summary(summarize(rho_vlf)),
            "log10_borehole" => round_summary(summarize(log_bh); digits=3),
            "log10_vlf" => round_summary(summarize(log10.(rho_vlf)); digits=3),
            "decade_bins_ohm_m" => decade_labels,
            "decade_fraction_borehole" => round.(bh_dec ./ max(bh_n, 1); digits=4),
            "decade_fraction_vlf" => round.(vlf_dec ./ max(vlf_n, 1); digits=4),
            "decade_counts_borehole" => bh_dec,
            "decade_counts_vlf" => vlf_dec,
        ),
        "layer_thickness_m" => Dict(
            "definition" => "|Δlog10 ρ| > $(LOG_SPLIT) or depth gap > 5 m splits a layer",
            "all_layers" => round_summary(summarize(layers); digits=2),
            "conductive_intervals_rho_lt_$(Int(COND_OHM))" => round_summary(summarize(cond_int); digits=2),
            "conductive_intervals_per_hole_p25" => round_summary(summarize(cond_int_p25); digits=2),
            "vertical_correlation_length_m" => round_summary(summarize(efold_m); digits=2),
        ),
        "anomaly_size_m" => Dict(
            "vlf_station_spacing_m" => round_summary(summarize(vlf_spacings); digits=2),
            "vlf_along_line_lowR_lt_$(round(cond_vlf; digits=1))" => round_summary(summarize(vlf_line_len); digits=1),
            "vlf_along_line_lowR_p10" => round_summary(summarize(vlf_line_len_p10); digits=1),
            "vlf_2d_eq_diameter_lt_$(round(cond_vlf; digits=1))" => round_summary(summarize(eqd); digits=1),
            "vlf_2d_eq_diameter_p10" => round_summary(summarize(eqd10); digits=1),
            "vlf_2d_area_m2" => round_summary(summarize(areas); digits=0),
            "vlf_2d_extent_x_m" => round_summary(summarize(lx); digits=1),
            "vlf_2d_extent_y_m" => round_summary(summarize(ly); digits=1),
            "vlf_horizontal_correlation_length_m" => round_summary(summarize(vlf_efold); digits=1),
            "raster_cell_m" => GRID_M,
        ),
        "generator_priors" => Dict(
            "note" => "Use these as the default sampling envelope for synthetic 3-D resistivity models of Keivitsa-like mafic/ultramafic + sulfide hosts.",
            "rho_ohm_m" => Dict(
                "background_host" => Dict("p50" => round_num(rec_host[1]; digits=1),
                                          "p84" => round_num(rec_host[2]; digits=1),
                                          "recommended_range" => [round_num(rec_host[1]; digits=1),
                                                                  round_num(rec_host[2]; digits=1)]),
                "conductor" => Dict("p05" => round_num(rec_cond[1]; digits=1),
                                    "p25_or_100" => round_num(rec_cond[2]; digits=1),
                                    "recommended_range" => [round_num(rec_cond[1]; digits=1),
                                                            round_num(rec_cond[2]; digits=1)],
                                    "cutoff_ohm_m" => COND_OHM),
                "full_envelope_p05_p95" => [round_num(rec_rho_min; digits=1),
                                            round_num(rec_rho_max; digits=1)],
                "clip" => [1.0, 1.0e5],
            ),
            "layer_thickness_m" => Dict(
                "p25" => round_num(rec_layer[1]; digits=1),
                "p50" => round_num(rec_layer[2]; digits=1),
                "p75" => round_num(rec_layer[3]; digits=1),
                "sample_lognormal" => true,
            ),
            "conductor_vertical_thickness_m" => Dict(
                "p25" => round_num(rec_anom_z[1]; digits=1),
                "p50" => round_num(rec_anom_z[2]; digits=1),
                "p75" => round_num(rec_anom_z[3]; digits=1),
            ),
            "conductor_horizontal_eq_diameter_m" => Dict(
                "p25" => round_num(rec_anom_xy[1]; digits=1),
                "p50" => round_num(rec_anom_xy[2]; digits=1),
                "p75" => round_num(rec_anom_xy[3]; digits=1),
                "source" => "VLF ov 25 m raster, 4-connected, ρ < min(100, VLF p25)",
            ),
            "correlation_length_m" => Dict(
                "vertical_median" => round_num(rec_corr_z; digits=1),
                "horizontal_median" => round_num(rec_corr_xy; digits=1),
                "definition" => "e-folding lag of log10(ρ) autocorrelation",
            ),
        ),
    )

    mkpath(dirname(YAML_OUT))
    open(YAML_OUT, "w") do io
        println(io, "# Keivitsa resistivity realism calibration")
        println(io, "# Generated by scripts/calibrate_resistivity.jl — do not edit by hand.")
        YAML.print(io, calib)
    end

    println("\n═══ Generator priors ═══")
    println("  host ρ (p50–p84): ", rec_host, " ohm·m")
    println("  conductor ρ:      ", rec_cond, " ohm·m")
    println("  envelope p05–p95: ", (rec_rho_min, rec_rho_max), " ohm·m")
    println("  layer thickness p25/p50/p75: ", rec_layer, " m")
    println("  conductor Δz p25/p50/p75:    ", rec_anom_z, " m")
    println("  conductor eqØ p25/p50/p75:   ", rec_anom_xy, " m")
    println("  ξz median: ", rec_corr_z, " m   ξxy median: ", rec_corr_xy, " m")
    println("Wrote ", YAML_OUT)
end

main()
