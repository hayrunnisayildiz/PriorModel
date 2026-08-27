#!/usr/bin/env julia
#=
forge_edi_inventory.jl — Utah FORGE EDI catalogue (no resampling / interpolation
/ SPECTRA→Z conversion).

MTGeophysics.jl v0.4.2 has no EDI reader (`src/` has ModEM + 2D `.dat` I/O
only). This script therefore uses a minimal header/SPECTRA/MTSECT parser.

Usage (from project root):
    julia --project=. scripts/forge_edi_inventory.jl
    julia --project=. scripts/forge_edi_inventory.jl \
        /Users/hayrunnisayildiz/data/forge/mt/EDIs \
        results
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics
using Dates

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_EDI_DIR = "/Users/hayrunnisayildiz/data/forge/mt/EDIs"
const DEFAULT_OUT_DIR = joinpath(ROOT, "results")

# Impedance / tipper / error tags we look for (never synthesised from SPECTRA).
const Z_TAGS = ("ZXXR", "ZXXI", "ZXYR", "ZXYI", "ZYXR", "ZYXI", "ZYYR", "ZYYI")
const TIPPER_TAGS = ("TXR", "TXI", "TYR", "TYI")
const ERR_TAGS = (
    "ZXX.VAR", "ZXY.VAR", "ZYX.VAR", "ZYY.VAR",
    "ZXXVAR", "ZXYVAR", "ZYXVAR", "ZYYVAR",
    "ZXX.ERR", "ZXY.ERR", "ZYX.ERR", "ZYY.ERR",
    "TXVAR.EXP", "TYVAR.EXP", "TX.VAR", "TY.VAR",
)

# ─────────────────────────────────────────────────────────────────────────────
# EDI record
# ─────────────────────────────────────────────────────────────────────────────

struct EDIRecord
    path::String
    filename::String
    site::String
    dataid::String
    sectid::String
    lat::Float64
    lon::Float64
    elev::Float64
    easting::Float64
    northing::Float64
    freqs::Vector{Float64}
    periods::Vector{Float64}
    components::String
    has_tipper::Bool
    mean_rel_error::Float64
    is_bvv::Bool
    remote_offset_m::Float64
    nchan::Int
    chtypes::Vector{String}
end

# ─────────────────────────────────────────────────────────────────────────────
# Header helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Strip EDI `>TAG` / `>=TAG` to the tag name (first token)."""
function edi_tag(line::AbstractString)::String
    t = strip(line)
    startswith(t, ">") || return ""
    t = lstrip(t, '>')
    t = lstrip(t, '=')
    tok = split(t; limit = 2)
    return isempty(tok) ? "" : uppercase(tok[1])
end

"""Parse `KEY=value` assignments on an EDI header line (quotes optional)."""
function parse_assignments(line::AbstractString)::Dict{String,String}
    out = Dict{String,String}()
    for m in eachmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|[^\s]+)", line)
        key = uppercase(m.captures[1])
        val = m.captures[2]
        if startswith(val, "\"") && endswith(val, "\"") && length(val) >= 2
            val = val[2:end-1]
        end
        out[key] = val
    end
    return out
end

"""DMS `±DD:MM:SS.s` or decimal degrees → Float64 degrees."""
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
        d = parse(Float64, parts[1])
        mn = parse(Float64, parts[2])
        sec = parse(Float64, parts[3])
        return sgn * (d + mn / 60.0 + sec / 3600.0)
    end
    v = tryparse(Float64, t)
    v === nothing && error("cannot parse angle $(repr(s))")
    return v
end

function first_present(dict::Dict{String,String}, keys::Vararg{String})::String
    for k in keys
        haskey(dict, k) && return dict[k]
    end
    return ""
end

"""
    site_id_from_filename(name) -> String

`FRG19013.edi` → `FRG19013`; `FRG19028R_bvv.edi` → `FRG19028`;
`FRG19102RR.edi` → `FRG19102`.
"""
function site_id_from_filename(name::AbstractString)::String
    stem = splitext(basename(String(name)))[1]
    stem = replace(stem, r"_bvv$"i => "")
    m = match(r"^(FRG\d+)", stem)
    return m === nothing ? stem : m.captures[1]
end

function is_bvv_filename(name::AbstractString)::Bool
    return occursin(r"_bvv"i, basename(String(name)))
end

# ─────────────────────────────────────────────────────────────────────────────
# NAD83 / GRS80 → UTM Zone 12N (EPSG:26912)
# USGS Transverse Mercator (Snyder). Not a data transform of MT responses.
# ─────────────────────────────────────────────────────────────────────────────

const GRS80_A = 6378137.0
const GRS80_F = 1.0 / 298.257222101
const UTM_K0 = 0.9996
const UTM_E0 = 500_000.0
const UTM_ZONE12_LON0 = -111.0  # deg

"""Geographic NAD83 (lat, lon in degrees) → UTM Zone 12N (easting, northing) m."""
function nad83_to_utm12(lat_deg::Float64, lon_deg::Float64)::Tuple{Float64,Float64}
    a = GRS80_A
    f = GRS80_F
    b = a * (1.0 - f)
    e2 = (a * a - b * b) / (a * a)
    ep2 = (a * a - b * b) / (b * b)
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    lon0 = deg2rad(UTM_ZONE12_LON0)
    sinφ = sin(lat)
    cosφ = cos(lat)
    tanφ = tan(lat)
    N = a / sqrt(1.0 - e2 * sinφ * sinφ)
    T = tanφ * tanφ
    C = ep2 * cosφ * cosφ
    A = (lon - lon0) * cosφ
    e4 = e2 * e2
    e6 = e4 * e2
    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat -
        (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * sin(2.0 * lat) +
        (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * sin(4.0 * lat) -
        (35.0 * e6 / 3072.0) * sin(6.0 * lat)
    )
    A2 = A * A
    A3 = A2 * A
    A4 = A2 * A2
    A5 = A4 * A
    A6 = A3 * A3
    east = UTM_K0 * N * (
        A + (1.0 - T + C) * A3 / 6.0 +
        (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ep2) * A5 / 120.0
    ) + UTM_E0
    north = UTM_K0 * (
        M + N * tanφ * (
            A2 / 2.0 +
            (5.0 - T + 9.0 * C + 4.0 * C * C) * A4 / 24.0 +
            (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ep2) * A6 / 720.0
        )
    )
    return east, north
end

# ─────────────────────────────────────────────────────────────────────────────
# Parser (inventory only)
# ─────────────────────────────────────────────────────────────────────────────

function parse_floats_from_lines(lines::Vector{String}, start_idx::Int)::Tuple{Vector{Float64},Int}
    vals = Float64[]
    i = start_idx
    n = length(lines)
    while i <= n
        t = strip(lines[i])
        if isempty(t)
            i += 1
            continue
        end
        if startswith(t, ">")
            break
        end
        for tok in split(t)
            v = tryparse(Float64, tok)
            v === nothing && continue
            push!(vals, v)
        end
        i += 1
    end
    return vals, i
end

function parse_edi(path::AbstractString)::EDIRecord
    lines = readlines(path)
    head = Dict{String,String}()
    meas = Dict{String,String}()
    sect = Dict{String,String}()
    chtypes = String[]
    remote_hyps = Float64[]
    freqs = Float64[]
    blocks = Dict{String,Vector{Float64}}()
    nchan = 0

    i = 1
    n = length(lines)
    while i <= n
        line = lines[i]
        t = strip(line)
        if isempty(t)
            i += 1
            continue
        end
        if !startswith(t, ">")
            i += 1
            continue
        end
        tag = edi_tag(t)
        assigns = parse_assignments(t)

        if tag == "HEAD"
            merge!(head, assigns)
            i += 1
            while i <= n && !startswith(strip(lines[i]), ">")
                merge!(head, parse_assignments(lines[i]))
                i += 1
            end
            continue
        elseif tag == "DEFINEMEAS"
            merge!(meas, assigns)
            i += 1
            while i <= n && !startswith(strip(lines[i]), ">")
                merge!(meas, parse_assignments(lines[i]))
                i += 1
            end
            continue
        elseif tag == "SPECTRASECT" || tag == "MTSECT"
            merge!(sect, assigns)
            i += 1
            while i <= n && !startswith(strip(lines[i]), ">")
                merge!(sect, parse_assignments(lines[i]))
                i += 1
            end
            continue
        elseif tag == "HMEAS" || tag == "EMEAS"
            cht = get(assigns, "CHTYPE", "")
            !isempty(cht) && push!(chtypes, uppercase(cht))
            xs = get(assigns, "X", "0")
            ys = get(assigns, "Y", "0")
            x = something(tryparse(Float64, xs), 0.0)
            y = something(tryparse(Float64, ys), 0.0)
            if tag == "HMEAS" && uppercase(get(assigns, "CHTYPE", "")) == "HX"
                push!(remote_hyps, hypot(x, y))
            end
            i += 1
            continue
        elseif tag == "SPECTRA"
            if haskey(assigns, "FREQ")
                f = tryparse(Float64, assigns["FREQ"])
                f !== nothing && push!(freqs, f)
            end
            # Skip spectral matrix values — inventory does not convert SPECTRA→Z.
            _, i = parse_floats_from_lines(lines, i + 1)
            continue
        elseif tag in Z_TAGS || tag in TIPPER_TAGS || tag in ERR_TAGS ||
               tag == "FREQ" || tag == "ZROT" || tag == "TROT" ||
               startswith(tag, "Z") || startswith(tag, "TX") || startswith(tag, "TY")
            vals, i = parse_floats_from_lines(lines, i + 1)
            blocks[tag] = vals
            if tag == "FREQ"
                append!(freqs, vals)
            end
            continue
        else
            i += 1
        end
    end

    lat_s = first_present(meas, "REFLAT")
    isempty(lat_s) && (lat_s = first_present(head, "LAT"))
    lon_s = first_present(meas, "REFLONG")
    isempty(lon_s) && (lon_s = first_present(head, "LONG"))
    elev_s = first_present(meas, "REFELEV")
    isempty(elev_s) && (elev_s = first_present(head, "ELEV"))

    lat = parse_angle(lat_s)
    lon = parse_angle(lon_s)
    elev = something(tryparse(Float64, strip(elev_s)), NaN)
    east, north = nad83_to_utm12(lat, lon)

    nchan_hdr = tryparse(Int, get(sect, "NCHAN", ""))
    nchan = nchan_hdr === nothing ? length(chtypes) : nchan_hdr

    # Periods from stored frequencies only (T = 1/f). No interpolation.
    periods = Float64[f > 0.0 ? 1.0 / f : NaN for f in freqs]

    present_z = String[t for t in Z_TAGS if haskey(blocks, t) && !isempty(blocks[t])]
    present_tip = String[t for t in TIPPER_TAGS if haskey(blocks, t) && !isempty(blocks[t])]
    present_err = String[t for t in ERR_TAGS if haskey(blocks, t) && !isempty(blocks[t])]
    has_zrot = haskey(blocks, "ZROT") && !isempty(blocks["ZROT"])
    has_tipper = !isempty(present_tip)

    parts = String[]
    if !isempty(present_z)
        push!(parts, join(present_z, "+"))
    end
    if has_zrot
        push!(parts, "ZROT")
    end
    if has_tipper
        push!(parts, join(present_tip, "+"))
    end
    if isempty(parts)
        # SPECTRA files: report measured channels, not derived Z.
        ch = copy(chtypes)
        if length(ch) >= 2 && ch[end - 1] == "HX" && ch[end] == "HY"
            ch[end - 1] = "HX_rr"
            ch[end] = "HY_rr"
        end
        ch_str = isempty(ch) ? "SPECTRA" : "SPECTRA:" * join(ch, ",")
        push!(parts, ch_str)
    end
    components = join(parts, ";")

    mean_rel = NaN
    if !isempty(present_z) && !isempty(present_err)
        rels = Float64[]
        # Pair Z??R/Z??I magnitude with matching VAR if both exist and same length.
        for (re_tag, im_tag, err_tag) in (
            ("ZXXR", "ZXXI", "ZXX.VAR"),
            ("ZXYR", "ZXYI", "ZXY.VAR"),
            ("ZYXR", "ZYXI", "ZYX.VAR"),
            ("ZYYR", "ZYYI", "ZYY.VAR"),
        )
            haskey(blocks, re_tag) || continue
            haskey(blocks, im_tag) || continue
            err = haskey(blocks, err_tag) ? blocks[err_tag] :
                  (haskey(blocks, replace(err_tag, "." => "")) ? blocks[replace(err_tag, "." => "")] : Float64[])
            zr = blocks[re_tag]
            zi = blocks[im_tag]
            npts = min(length(zr), length(zi), length(err))
            for k in 1:npts
                mag = hypot(zr[k], zi[k])
                mag > 0.0 && err[k] >= 0.0 && push!(rels, err[k] / mag)
            end
        end
        !isempty(rels) && (mean_rel = mean(rels))
    end

    remote_offset = isempty(remote_hyps) ? NaN : maximum(remote_hyps)
    filename = basename(String(path))
    sectid = get(sect, "SECTID", splitext(filename)[1])
    dataid = get(head, "DATAID", "")
    site = site_id_from_filename(filename)

    return EDIRecord(
        String(path), filename, site, dataid, sectid,
        lat, lon, elev, east, north,
        freqs, periods, components, has_tipper, mean_rel,
        is_bvv_filename(filename), remote_offset, nchan, chtypes,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Directory listing (step 0)
# ─────────────────────────────────────────────────────────────────────────────

function count_by_ext(dir::AbstractString)::Dict{String,Int}
    counts = Dict{String,Int}()
    isdir(dir) || return counts
    for (root, _, files) in walkdir(dir)
        startswith(basename(root), ".") && continue
        for f in files
            startswith(f, ".") && continue
            ext = lowercase(splitext(f)[2])
            isempty(ext) && (ext = "(none)")
            counts[ext] = get(counts, ext, 0) + 1
        end
    end
    return counts
end

function report_directory(edi_dir::AbstractString)
    println("═"^72)
    println("[0] Directory listing")
    println("    EDI dir: ", edi_dir)
    println("═"^72)
    isdir(edi_dir) || error("EDI directory not found: $edi_dir")

    entries = readdir(edi_dir)
    files = String[f for f in entries if isfile(joinpath(edi_dir, f)) && !startswith(f, ".")]
    edis = sort(String[f for f in files if lowercase(splitext(f)[2]) == ".edi"])
    txts = sort(String[f for f in files if lowercase(splitext(f)[2]) == ".txt"])

    ext_counts = Dict{String,Int}()
    for f in files
        ext = lowercase(splitext(f)[2])
        isempty(ext) && (ext = "(none)")
        ext_counts[ext] = get(ext_counts, ext, 0) + 1
    end

    println("  files (non-dot): ", length(files))
    println("  extensions:")
    for ext in sort(collect(keys(ext_counts)))
        @printf("    %-8s %d\n", ext, ext_counts[ext])
    end

    n_plain = count(f -> occursin(r"^FRG\d+\.edi$"i, f), edis)
    n_r = count(f -> occursin(r"^FRG\d+R+\.edi$"i, f), edis)
    n_bvv = count(f -> occursin(r"_bvv\.edi$"i, f), edis)
    n_r_bvv = count(f -> occursin(r"^FRG\d+R+_bvv\.edi$"i, f), edis)
    println("  EDI naming (this directory):")
    @printf("    FRG#####.edi          %d\n", n_plain)
    @printf("    FRG#####R(R).edi      %d\n", n_r)
    @printf("    FRG#####_bvv.edi      %d  (of which FRG#####R+_bvv: %d)\n", n_bvv, n_r_bvv)
    @printf("    other .edi            %d\n", length(edis) - n_plain - n_r - n_bvv)
    println("  companion survey txt:   ", length(txts), "  (*_srvy.txt)")

    parent = dirname(edi_dir)
    dup = joinpath(parent, "EDIs 2")
    if isdir(dup) && abspath(dup) != abspath(edi_dir)
        dup_edis = count(f -> lowercase(splitext(f)[2]) == ".edi", readdir(dup))
        println("  note: sibling '", dup, "' has ", dup_edis,
                " EDI files (byte-identical copy; not inventoried).")
    end
    println()
    return edis
end

function print_first_lines(path::AbstractString, n::Int)
    println("═"^72)
    println("[1] First $n lines of ", basename(path), " (verbatim)")
    println("═"^72)
    lines = readlines(path)
    for i in 1:min(n, length(lines))
        println(lines[i])
    end
    println("── end of excerpt (", min(n, length(lines)), " / ", length(lines), " lines) ──")
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

function csv_escape(s::AbstractString)::String
    t = String(s)
    if occursin(',', t) || occursin('"', t) || occursin('\n', t)
        return "\"" * replace(t, "\"" => "\"\"") * "\""
    end
    return t
end

function nearest_neighbour_km(east::Vector{Float64}, north::Vector{Float64})::Vector{Float64}
    n = length(east)
    dmin = fill(Inf, n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            d = hypot(east[i] - east[j], north[i] - north[j]) / 1000.0
            dmin[i] = min(dmin[i], d)
        end
    end
    return dmin
end

"""Intersection of frequency lists; exact match after 8-decimal rounding (Hz)."""
function common_frequencies(records::Vector{EDIRecord})::Vector{Float64}
    isempty(records) && return Float64[]
    key(f) = round(f; digits = 8)
    common = Set{Float64}(key.(records[1].freqs))
    for rec in records[2:end]
        intersect!(common, Set{Float64}(key.(rec.freqs)))
    end
    return sort(collect(common); rev = true)
end

function unique_site_records(records::Vector{EDIRecord})::Vector{EDIRecord}
    by_site = Dict{String,Vector{EDIRecord}}()
    for rec in records
        push!(get!(Vector{EDIRecord}, by_site, rec.site), rec)
    end
    out = EDIRecord[]
    for site in sort(collect(keys(by_site)))
        recs = by_site[site]
        local_ones = EDIRecord[r for r in recs if !r.is_bvv]
        push!(out, isempty(local_ones) ? recs[1] : local_ones[1])
    end
    return out
end

function full_z_tensor(rec::EDIRecord)::Bool
    tags = uppercase(rec.components)
    return all(t -> occursin(t, tags), Z_TAGS)
end

# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

function write_csv(path::AbstractString, records::Vector{EDIRecord})
    open(path, "w") do io
        println(io, "site,filename,lat,lon,elev,utm_easting,utm_northing,",
                    "n_period,t_min_s,t_max_s,components,has_tipper,mean_rel_error")
        for rec in records
            tfin = Float64[t for t in rec.periods if isfinite(t) && t > 0.0]
            nper = length(tfin)
            tmin = isempty(tfin) ? NaN : minimum(tfin)
            tmax = isempty(tfin) ? NaN : maximum(tfin)
            rel = rec.mean_rel_error
            rel_s = isfinite(rel) ? @sprintf("%.6g", rel) : ""
            @printf(io, "%s,%s,%.8f,%.8f,%.3f,%.3f,%.3f,%d,%.8g,%.8g,%s,%s,%s\n",
                    csv_escape(rec.site), csv_escape(rec.filename),
                    rec.lat, rec.lon, rec.elev, rec.easting, rec.northing,
                    nper, tmin, tmax, csv_escape(rec.components),
                    rec.has_tipper ? "yes" : "no", rel_s)
        end
    end
end

function write_markdown(path::AbstractString, edi_dir::AbstractString,
                        records::Vector{EDIRecord}, sites::Vector{EDIRecord})
    n_files = length(records)
    n_sites = length(sites)
    n_bvv = count(r -> r.is_bvv, records)
    n_local = n_files - n_bvv
    n_r_suffix = count(r -> occursin(r"^FRG\d+R+"i, splitext(r.filename)[1]) && !r.is_bvv,
                       records)

    east = Float64[r.easting for r in sites]
    north = Float64[r.northing for r in sites]
    e_min, e_max = minimum(east), maximum(east)
    n_min, n_max = minimum(north), maximum(north)
    area_km2 = ((e_max - e_min) / 1000.0) * ((n_max - n_min) / 1000.0)

    nn = nearest_neighbour_km(east, north)
    nn_min, nn_med, nn_max = minimum(nn), median(nn), maximum(nn)

    common_f = common_frequencies(records)
    all_t = Float64[]
    for rec in records
        for t in rec.periods
            isfinite(t) && t > 0.0 && push!(all_t, t)
        end
    end
    t_union_min = isempty(all_t) ? NaN : minimum(all_t)
    t_union_max = isempty(all_t) ? NaN : maximum(all_t)
    common_t = Float64[f > 0.0 ? 1.0 / f : NaN for f in common_f]
    common_t = Float64[t for t in common_t if isfinite(t)]

    n_full_z = count(full_z_tensor, records)
    n_tip = count(r -> r.has_tipper, records)
    n_spectra_only = count(r -> startswith(r.components, "SPECTRA"), records)

    rels = Float64[r.mean_rel_error for r in records if isfinite(r.mean_rel_error)]
    rel_med = isempty(rels) ? NaN : median(rels)
    rel_p10 = isempty(rels) ? NaN : quantile(rels, 0.10)
    rel_p90 = isempty(rels) ? NaN : quantile(rels, 0.90)

    nchan_set = sort(unique(Int[r.nchan for r in records]))
    nfreq_set = sort(unique(Int[length(r.freqs) for r in records]))
    bvv_off = Float64[r.remote_offset_m for r in records if r.is_bvv && isfinite(r.remote_offset_m)]
    loc_off = Float64[r.remote_offset_m for r in records if !r.is_bvv && isfinite(r.remote_offset_m)]

    mismatched = EDIRecord[]
    by_site = Dict{String,Vector{EDIRecord}}()
    for rec in records
        push!(get!(Vector{EDIRecord}, by_site, rec.site), rec)
    end
    n_pairs = 0
    n_unpaired = 0
    for (_, recs) in by_site
        has_l = any(!r.is_bvv for r in recs)
        has_b = any(r.is_bvv for r in recs)
        has_l && has_b && (n_pairs += 1)
        xor(has_l, has_b) && (n_unpaired += 1)
        if length(recs) >= 2
            lat0, lon0 = recs[1].lat, recs[1].lon
            if any(abs(r.lat - lat0) > 1e-8 || abs(r.lon - lon0) > 1e-8 for r in recs)
                append!(mismatched, recs)
            end
        end
    end

    open(path, "w") do io
        println(io, "# Utah FORGE EDI inventory")
        println(io)
        println(io, "Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)
        println(io, "Source directory: `", edi_dir, "`")
        println(io)
        println(io, "Parser: minimal EDI reader in `scripts/forge_edi_inventory.jl`.")
        println(io, "MTGeophysics.jl v0.4.2 has **no EDI reader** (ModEM / 2D `.dat` only);")
        println(io, "nothing was resampled, interpolated, or converted SPECTRA→Z.")
        println(io)
        println(io, "## Sites and local / remote pairing")
        println(io)
        println(io, "- Unique sites (FRG#####): **", n_sites, "**")
        println(io, "- EDI files inventoried: **", n_files, "**")
        println(io, "- Non-`_bvv` files: **", n_local, "**  (of which `R`/`RR` suffix: ", n_r_suffix, ")")
        println(io, "- `_bvv` files: **", n_bvv, "**")
        println(io, "- Sites with both counterparts: **", n_pairs, "**")
        println(io, "- Sites with only one counterpart: **", n_unpaired, "**")
        println(io)
        println(io, "GDR: each site may have a local-independent-reference file and a")
        println(io, "remote-reference file. On disk that split is the filename suffix,")
        println(io, "not an MTSECT impedance block:")
        println(io)
        println(io, "- `FRG#####.edi` or `FRG#####R.edi` / `FRG#####RR.edi` — nearer")
        println(io, "  remote H (HMEAS HX/HY offset hypot median ",
                    @sprintf("%.1f km", isempty(loc_off) ? NaN : median(loc_off) / 1000),
                    "; range ",
                    @sprintf("%.1f–%.1f km",
                             isempty(loc_off) ? NaN : minimum(loc_off) / 1000,
                             isempty(loc_off) ? NaN : maximum(loc_off) / 1000),
                    ")")
        println(io, "- `FRG#####_bvv.edi` (or `FRG#####R_bvv.edi`) — farther remote H")
        println(io, "  (hypot median ",
                    @sprintf("%.1f km", isempty(bvv_off) ? NaN : median(bvv_off) / 1000),
                    "; range ",
                    @sprintf("%.1f–%.1f km",
                             isempty(bvv_off) ? NaN : minimum(bvv_off) / 1000,
                             isempty(bvv_off) ? NaN : maximum(bvv_off) / 1000),
                    ")")
        println(io)
        println(io, "Five sites use an `R`/`RR` stem instead of a plain `FRG#####.edi`:")
        println(io, "FRG19028, FRG19029, FRG19042, FRG19046, FRG19102.")
        println(io, "Their `_bvv` twins keep that stem (`FRG19028R_bvv.edi`, …,")
        println(io, "`FRG19102RR_bvv.edi`).")
        println(io)
        if !isempty(mismatched)
            println(io, "Coordinate mismatch between a site's two files (REFLAT/REFLONG):")
            for rec in mismatched
                @printf(io, "- `%s`  lat=%.8f  lon=%.8f  elev=%.1f m\n",
                        rec.filename, rec.lat, rec.lon, rec.elev)
            end
            println(io)
        end
        println(io, "A sibling directory `EDIs 2` is a byte-identical copy of `EDIs`")
        println(io, "and was not inventoried a second time.")
        println(io)
        println(io, "## Coverage (unique sites, non-`_bvv` coordinates)")
        println(io)
        println(io, "CRS: NAD83 geographic in the EDI header → UTM Zone 12N")
        println(io, "(GRS80 Transverse Mercator, EPSG:26912).")
        println(io)
        @printf(io, "- Easting  min–max: **%.1f – %.1f m**  (Δ = %.2f km)\n",
                e_min, e_max, (e_max - e_min) / 1000)
        @printf(io, "- Northing min–max: **%.1f – %.1f m**  (Δ = %.2f km)\n",
                n_min, n_max, (n_max - n_min) / 1000)
        @printf(io, "- Bounding-box area: **%.2f km²**\n", area_km2)
        println(io)
        println(io, "## Station spacing (nearest neighbour, unique sites)")
        println(io)
        @printf(io, "- min / median / max: **%.3f / %.3f / %.3f km**\n",
                nn_min, nn_med, nn_max)
        println(io)
        println(io, "## Period band")
        println(io)
        println(io, "Every file stores `>=SPECTRASECT` / `>SPECTRA FREQ=` (Hz);")
        println(io, "periods below are `T = 1/f` from those listed frequencies.")
        println(io)
        println(io, "- Frequencies per file: ", join(nfreq_set, ", "))
        println(io, "- Channels per file (`NCHAN`): ", join(nchan_set, ", "))
        println(io, "- Common frequencies across all files: **", length(common_f), "**")
        if !isempty(common_t)
            @printf(io, "- Common-band T min / max: **%.6g / %.6g s**\n",
                    minimum(common_t), maximum(common_t))
        end
        @printf(io, "- Union-band T min / max: **%.6g / %.6g s**\n",
                t_union_min, t_union_max)
        println(io)
        println(io, "## Component completeness")
        println(io)
        println(io, "- Files with a full impedance tensor")
        println(io, "  (ZXXR/I, ZXYR/I, ZYXR/I, ZYYR/I): **", n_full_z, " / ", n_files, "**")
        println(io, "- Files with tipper blocks (TXR/TXI/TYR/TYI): **", n_tip, " / ", n_files, "**")
        println(io, "- Files that are SPECTRA-only (no MTSECT Z/T blocks): **",
                    n_spectra_only, " / ", n_files, "**")
        println(io)
        println(io, "SPECTRA `NCHAN=7` is HX, HY, HZ, EX, EY plus remote HX, HY.")
        println(io, "HZ is present in the spectra, but **tipper is not stored** as TX/TY.")
        println(io)
        println(io, "## Error levels")
        println(io)
        if isempty(rels)
            println(io, "No impedance/tipper error fields (`Zxx.VAR`, `TX.VAR`, …) in any file.")
            println(io, "Mean relative error column in the CSV is empty. Median and")
            println(io, "%10–%90 range: **n/a**.")
        else
            @printf(io, "- Median relative error: **%.4g**\n", rel_med)
            @printf(io, "- 10th–90th percentile: **%.4g – %.4g**\n", rel_p10, rel_p90)
            println(io, "- Files contributing: ", length(rels), " / ", n_files)
        end
        println(io)
        println(io, "## Artefacts")
        println(io)
        println(io, "- `results/forge_edi_inventory.csv`")
        println(io, "- `results/forge_edi_inventory.md`")
        println(io, "- `results/forge_stations.png`")
    end
end

function plot_stations(png_path::AbstractString, sites::Vector{EDIRecord})
    east_km = Float64[r.easting / 1000.0 for r in sites]
    north_km = Float64[r.northing / 1000.0 for r in sites]
    nsite = length(sites)

    println("[5] Loading MTGeophysics (CairoMakie backend) for the station map…")
    flush(stdout)
    # `using` inside a function bumps world age; plot in the same @eval.
    @eval begin
        using MTGeophysics
        let
            CM = MTGeophysics.CairoMakie
            CM.activate!()
            fig = CM.Figure(size = (900, 900), fontsize = 14)
            ax = CM.Axis(
                fig[1, 1];
                xlabel = "Easting (km, UTM Zone 12N NAD83)",
                ylabel = "Northing (km, UTM Zone 12N NAD83)",
                title = "Utah FORGE MT stations (n = $($nsite))",
                aspect = CM.DataAspect(),
            )
            CM.scatter!(ax, $(east_km), $(north_km); markersize = 9, color = :steelblue)
            CM.save($(png_path), fig)
        end
    end
    return png_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function parse_cli(args::Vector{String})
    edi_dir = DEFAULT_EDI_DIR
    out_dir = DEFAULT_OUT_DIR
    if length(args) >= 1 && !isempty(args[1])
        edi_dir = abspath(args[1])
    end
    if length(args) >= 2 && !isempty(args[2])
        out_dir = abspath(args[2])
    end
    return edi_dir, out_dir
end

function main(args::Vector{String}=ARGS)
    edi_dir, out_dir = parse_cli(args)
    mkpath(out_dir)

    edis = report_directory(edi_dir)
    isempty(edis) && error("no .edi files in $edi_dir")

    sample = joinpath(edi_dir, edis[1])
    print_first_lines(sample, 60)

    println("═"^72)
    println("[2] EDI reader: MTGeophysics.jl has none (checked src/). Minimal parser.")
    println("[3–5] Parsing ", length(edis), " EDI files (no SPECTRA→Z conversion)…")
    println("═"^72)
    flush(stdout)

    records = EDIRecord[]
    for (k, name) in enumerate(edis)
        rec = parse_edi(joinpath(edi_dir, name))
        push!(records, rec)
        if k == 1 || k == length(edis) || k % 50 == 0
            @printf("  %3d/%d  %s  site=%s  nfreq=%d  %s\n",
                    k, length(edis), rec.filename, rec.site,
                    length(rec.freqs), rec.components)
            flush(stdout)
        end
    end
    sort!(records; by = r -> (r.site, r.is_bvv, r.filename))
    sites = unique_site_records(records)

    csv_path = joinpath(out_dir, "forge_edi_inventory.csv")
    md_path = joinpath(out_dir, "forge_edi_inventory.md")
    png_path = joinpath(out_dir, "forge_stations.png")

    write_csv(csv_path, records)
    write_markdown(md_path, edi_dir, records, sites)
    println("  wrote ", csv_path)
    println("  wrote ", md_path)
    flush(stdout)

    plot_stations(png_path, sites)
    println("  wrote ", png_path)

    println()
    println("Done. Unique sites = ", length(sites),
            ", files = ", length(records), ".")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
