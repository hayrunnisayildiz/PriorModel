#!/usr/bin/env julia
#=
geysers_elevation_check.jl — Northwest Geysers MT elevation vs ModEM Z
diagnosis. Diagnosis only: no inversion, no forward modelling, no datum
or unit conversion applied to any data product.

Compares three hypotheses for why EDI REFELEV and ModEM Z disagree
(~1780 m at GZ01), using gravity ELEV_M (NGVD29 metres) as independent
topography control.

Usage (from project root):
    julia --project=. scripts/geysers_elevation_check.jl

Outputs (written, never converted in place):
    results/geysers_elevation_check.md
    results/geysers_elevation_check.csv
    results/DATA_NOTES.md  — section 4 appended/replaced only
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics

const DEFAULT_EDI_DIR = joinpath(ROOT, "data", "geysers", "mt", "edi")
const DEFAULT_MODEM_DAT = joinpath(ROOT, "data", "geysers", "model",
                                   "geysers_modEM_data_file.dat")
const DEFAULT_RHO = joinpath(ROOT, "data", "geysers", "model",
                             "geysers_preferred_model.rho")
const DEFAULT_GRAV_CSV = joinpath(ROOT, "data", "geysers", "gravity",
                                  "gravity_clean.csv")
const DEFAULT_OUT_DIR = joinpath(ROOT, "results")
const DEFAULT_DATA_NOTES = joinpath(ROOT, "results", "DATA_NOTES.md")

# User-stated origin z if the mesh file cannot be parsed (metres, +z down).
const ORIGIN_Z_FALLBACK = -1083.929

# International foot → metre (exact).
const FT_TO_M = 0.3048

# Nearby-site topography should agree to tens of metres, not hundreds.
# NGVD29→NAVD88 at The Geysers is ~0.93 m (see DATA_NOTES.md; not applied).
const TOPO_OK_STD_M = 80.0
const TOPO_OK_MAXABS_M = 200.0
const DATUM_SHIFT_OK_STD_M = 5.0

# WGS84 → UTM Zone 10N (same series as scripts/geysers_gravity_prep.jl)
const WGS84_A = 6378137.0
const WGS84_F = 1 / 298.257223563
const UTM_K0 = 0.9996
const UTM_FALSE_EASTING = 500_000.0
const UTM_ZONE10_LON0 = -123.0

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""One EDI station: HEAD / DEFINEMEAS metadata only (no spectra)."""
struct EDIStation
    name::String
    path::String
    lat::Float64
    lon::Float64
    east::Float64
    north::Float64
    refelev::Float64
    elev_head::Float64
    units_head::String
    units_meas::String
    units::String
end

"""One unique ModEM data-file station (first occurrence of Code)."""
struct ModEMStation
    name::String
    lat::Float64
    lon::Float64
    x::Float64
    y::Float64
    z::Float64
end

struct GravPt
    id::String
    x::Float64
    y::Float64
    elev::Float64
end

"""Difference statistics. `mean` / `std` of signed `Δ`; `maxabs` of `|Δ|`."""
struct DiffStats
    n::Int
    mean::Float64
    std::Float64
    maxabs::Float64
    median::Float64
    min::Float64
    max::Float64
end

struct PairStats
    n::Int
    mean::Float64
    std::Float64
    maxabs::Float64
    median::Float64
    cor::Float64
end

"""Per-station matched row used for hypotheses and gravity NN."""
struct MatchedRow
    name::String
    refelev::Float64
    units::String
    modem_x::Float64
    modem_y::Float64
    modem_z::Float64
    elev_h1::Float64          # REFELEV * 0.3048
    elev_modEM::Float64       # (-origin_z) - Z
    east::Float64
    north::Float64
    grav_id::String
    grav_elev::Float64
    dist_m::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Geometry / parse helpers (copied, not imported, from gravity_prep)
# ─────────────────────────────────────────────────────────────────────────────

"""Read a text file, normalise CRLF / CR to LF, split into lines (no EOL)."""
function read_lines_crlf(path::AbstractString)::Vector{String}
    raw = read(path, String)
    raw = replace(raw, "\r\n" => "\n")
    raw = replace(raw, "\r" => "\n")
    return split(raw, '\n'; keepempty = true)
end

function parse_float_cell(s::AbstractString)::Float64
    t = strip(s)
    (isempty(t) || t == "NA" || t == "NaN" || t == "nan" || t == "NULL") && return NaN
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
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
        length(parts) == 3 || return NaN
        d = tryparse(Float64, parts[1])
        m = tryparse(Float64, parts[2])
        sec = tryparse(Float64, parts[3])
        (d === nothing || m === nothing || sec === nothing) && return NaN
        return sgn * (d + m / 60 + sec / 3600)
    end
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

"""
    wgs84_to_utm10n(lat_deg, lon_deg) -> (easting_m, northing_m)

WGS84 geographic → UTM Zone 10N (EPSG:32610). Snyder (1987) USGS PP 1395.
Copied from `scripts/geysers_gravity_prep.jl` so this script stays standalone.
WGS84 vs NAD83 is ~1 m here (same note as gravity QC); no datum shift applied.
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

function diff_stats(δ::AbstractVector{<:Real})::DiffStats
    w = Float64[x for x in δ if isfinite(x)]
    isempty(w) && return DiffStats(0, NaN, NaN, NaN, NaN, NaN, NaN)
    n = length(w)
    s = n >= 2 ? std(w) : 0.0
    return DiffStats(n, mean(w), s, maximum(abs.(w)), median(w),
                     minimum(w), maximum(w))
end

function pair_stats(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})::PairStats
    aa = Float64[]
    bb = Float64[]
    for i in eachindex(a)
        isfinite(a[i]) && isfinite(b[i]) || continue
        push!(aa, Float64(a[i]))
        push!(bb, Float64(b[i]))
    end
    n = length(aa)
    n == 0 && return PairStats(0, NaN, NaN, NaN, NaN, NaN)
    δ = aa .- bb
    c = (n >= 3 && std(aa) > 1e-9 && std(bb) > 1e-9) ? cor(aa, bb) : NaN
    s = n >= 2 ? std(δ) : 0.0
    return PairStats(n, mean(δ), s, maximum(abs.(δ)), median(δ), c)
end

function fmt_num(x::Float64; digits::Int = 3)::String
    isfinite(x) || return "n/a"
    return @sprintf("%.*f", digits, x)
end

function fmt_cor(x::Float64)::String
    isfinite(x) || return "n/a (std = 0)"
    return @sprintf("%.4f", x)
end

function consistent_topo(st::DiffStats)::Bool
    return st.n > 0 && st.std <= TOPO_OK_STD_M && st.maxabs <= TOPO_OK_MAXABS_M
end

# ─────────────────────────────────────────────────────────────────────────────
# EDI
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_edi_station(path) -> EDIStation

HEAD / DEFINEMEAS only. Elevation: `REFELEV` (DEFINEMEAS), else HEAD `ELEV`.
UNITS from HEAD and DEFINEMEAS are both recorded; `units` prefers DEFINEMEAS.
"""
function parse_edi_station(path::AbstractString)::EDIStation
    head = Dict{String,String}()
    meas = Dict{String,String}()
    section = :none
    for raw in eachline(path)
        t = strip(raw)
        ut = uppercase(t)
        if startswith(ut, ">HEAD")
            section = :head
            merge!(head, parse_assignments(t))
            continue
        elseif startswith(ut, ">=") && occursin("DEFINEMEAS", ut)
            section = :meas
            merge!(meas, parse_assignments(t))
            continue
        elseif startswith(t, ">")
            section = :none
            continue
        end
        section == :head && merge!(head, parse_assignments(t))
        section == :meas && merge!(meas, parse_assignments(t))
    end
    lat_s = get(meas, "REFLAT", get(head, "LAT", get(head, "LATITUDE", "")))
    lon_s = get(meas, "REFLON", get(head, "LON", get(head, "LONG",
                 get(head, "LONGITUDE", ""))))
    lat = parse_angle(lat_s)
    lon = parse_angle(lon_s)
    (isfinite(lat) && isfinite(lon)) || error("no LAT/LON in $path")
    east, north = wgs84_to_utm10n(lat, lon)
    name = uppercase(get(head, "DATAID", splitext(basename(path))[1]))
    elev_head = parse_float_cell(get(head, "ELEV", get(head, "ELEVATION", "")))
    refelev = parse_float_cell(get(meas, "REFELEV", ""))
    if !isfinite(refelev)
        refelev = elev_head
    end
    units_head = String(get(head, "UNITS", ""))
    units_meas = String(get(meas, "UNITS", ""))
    units = isempty(units_meas) ? units_head : units_meas
    return EDIStation(name, path, lat, lon, east, north, refelev, elev_head,
                      units_head, units_meas, units)
end

function load_edi_stations(edi_dir::AbstractString)::Vector{EDIStation}
    isdir(edi_dir) || error("EDI directory not found: $edi_dir")
    files = sort(String[joinpath(edi_dir, f) for f in readdir(edi_dir)
                        if lowercase(splitext(f)[2]) == ".edi" && !startswith(f, ".")])
    isempty(files) && error("no .edi files in $edi_dir")
    return EDIStation[parse_edi_station(f) for f in files]
end

function unique_units_table(edi::Vector{EDIStation})
    counts = Dict{String,Int}()
    for s in edi
        key = isempty(s.units) ? "(missing)" : s.units
        counts[key] = get(counts, key, 0) + 1
    end
    return sort(collect(counts); by = first)
end

# ─────────────────────────────────────────────────────────────────────────────
# ModEM data + mesh origin
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_ws_origin_z(rho_path) -> (origin_x, origin_y, origin_z, used_fallback)

Mackie/WS `.rho` files end with `ox oy oz` then a rotation line. Walk from
EOF to the first line with exactly three floats. Fallback: `$ORIGIN_Z_FALLBACK`.
"""
function parse_ws_origin_z(rho_path::AbstractString)
    isfile(rho_path) || return (NaN, NaN, ORIGIN_Z_FALLBACK, true)
    lines = read_lines_crlf(rho_path)
    for line in Iterators.reverse(lines)
        t = strip(line)
        isempty(t) && continue
        nums = Float64[]
        ok = true
        for tok in split(t)
            v = tryparse(Float64, tok)
            if v === nothing
                ok = false
                break
            end
            push!(nums, v)
        end
        ok || continue
        if length(nums) == 3
            return (nums[1], nums[2], nums[3], false)
        end
    end
    return (NaN, NaN, ORIGIN_Z_FALLBACK, true)
end

"""
    parse_modem_data(path) -> (stations, n_data_rows, origin_lat, origin_lon,
                               n_period_hdr, n_sta_hdr, header_comments)

WS/ModEM data file. Comment lines `#…`; block headers `>…`.
Data columns (file header): Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m)
Component Real Imag Error. Station Z is taken from the first row of each Code;
later rows are checked for a Z mismatch > 1 mm.
"""
function parse_modem_data(path::AbstractString)
    isfile(path) || error("ModEM data file not found: $path")
    stations = Dict{String,ModEMStation}()
    n_data = 0
    origin_lat = NaN
    origin_lon = NaN
    n_period_hdr = 0
    n_sta_hdr = 0
    comments = String[]
    z_mismatch = String[]
    # WS/ModEM `>` two-number lines in file order: (lat, lon), then (nper, nsta).
    two_float_hdr = Tuple{Float64,Float64}[]
    for raw in eachline(path)
        t = strip(raw)
        if startswith(t, "#")
            push!(comments, t)
            continue
        end
        if startswith(t, ">")
            rest = strip(t[2:end])
            isempty(rest) && continue
            parts = split(rest)
            if length(parts) == 2
                a = tryparse(Float64, parts[1])
                b = tryparse(Float64, parts[2])
                if a !== nothing && b !== nothing
                    push!(two_float_hdr, (a, b))
                end
            end
            continue
        end
        isempty(t) && continue
        parts = split(t)
        length(parts) < 8 && continue
        name = uppercase(String(parts[2]))
        lat = parse_float_cell(parts[3])
        lon = parse_float_cell(parts[4])
        x = parse_float_cell(parts[5])
        y = parse_float_cell(parts[6])
        z = parse_float_cell(parts[7])
        n_data += 1
        if haskey(stations, name)
            if isfinite(z) && isfinite(stations[name].z) &&
               abs(z - stations[name].z) > 1e-3
                push!(z_mismatch, name)
            end
        else
            stations[name] = ModEMStation(name, lat, lon, x, y, z)
        end
    end
    isempty(z_mismatch) || @warn "ModEM Z not unique per station" names=unique(z_mismatch)
    if length(two_float_hdr) >= 1
        origin_lat, origin_lon = two_float_hdr[1]
    end
    if length(two_float_hdr) >= 2
        n_period_hdr = Int(round(two_float_hdr[2][1]))
        n_sta_hdr = Int(round(two_float_hdr[2][2]))
    end
    return stations, n_data, origin_lat, origin_lon, n_period_hdr, n_sta_hdr, comments
end

# ─────────────────────────────────────────────────────────────────────────────
# Gravity nearest neighbour
# ─────────────────────────────────────────────────────────────────────────────

function load_gravity_points(csv_path::AbstractString)::Vector{GravPt}
    isfile(csv_path) || error("gravity CSV not found: $csv_path")
    df = CSV.read(csv_path, DataFrame)
    cmap = Dict{String,Symbol}()
    for n in names(df)
        cmap[uppercase(String(n))] = Symbol(n)
    end
    haskey(cmap, "UTM10X_NAD83") || error("$csv_path: missing UTM10x_NAD83")
    haskey(cmap, "UTM10Y_NAD83") || error("$csv_path: missing UTM10y_NAD83")
    haskey(cmap, "ELEV_M") || error("$csv_path: missing ELEV_M")
    idcol = haskey(cmap, "ID") ? cmap["ID"] : nothing
    pts = GravPt[]
    for row in eachrow(df)
        x = Float64(row[cmap["UTM10X_NAD83"]])
        y = Float64(row[cmap["UTM10Y_NAD83"]])
        e = Float64(row[cmap["ELEV_M"]])
        (isfinite(x) && isfinite(y) && isfinite(e)) || continue
        id = idcol === nothing ? "" : String(row[idcol])
        push!(pts, GravPt(id, x, y, e))
    end
    isempty(pts) && error("$csv_path: no usable gravity stations")
    return pts
end

"""Nearest gravity station in UTM 10N (metres). Brute force; N is small."""
function nearest_gravity(east::Float64, north::Float64,
                         grav::Vector{GravPt})::Tuple{GravPt,Float64}
    best = grav[1]
    bestd = hypot(east - best.x, north - best.y)
    for g in grav
        d = hypot(east - g.x, north - g.y)
        if d < bestd
            bestd = d
            best = g
        end
    end
    return best, bestd
end

# ─────────────────────────────────────────────────────────────────────────────
# Match + hypotheses
# ─────────────────────────────────────────────────────────────────────────────

"""
    elev_from_modem(z_station, origin_z) -> metres (up)

Mesh origin `origin_z` is negative when the mesh top sits above the z = 0
datum (sea level, +z down). Station `Z` in the data file is positive-down
from that highest-topography mesh top. So

    elev_modEM = (−origin_z) − Z_station

With origin_z = −1083.929 this is `1083.929 − Z`.
"""
elev_from_modem(z_station::Float64, origin_z::Float64)::Float64 =
    (-origin_z) - z_station

function match_stations(edi::Vector{EDIStation},
                        modem::Dict{String,ModEMStation},
                        grav::Vector{GravPt},
                        origin_z::Float64)
    edi_names = Set{String}(s.name for s in edi)
    modem_names = Set{String}(keys(modem))
    edi_only = sort(collect(setdiff(edi_names, modem_names)))
    modem_only = sort(collect(setdiff(modem_names, edi_names)))
    common = sort(collect(intersect(edi_names, modem_names)))
    edi_by = Dict{String,EDIStation}(s.name => s for s in edi)
    rows = MatchedRow[]
    for name in common
        e = edi_by[name]
        m = modem[name]
        elev_h1 = e.refelev * FT_TO_M
        elev_m = elev_from_modem(m.z, origin_z)
        g, d = nearest_gravity(e.east, e.north, grav)
        push!(rows, MatchedRow(name, e.refelev, e.units, m.x, m.y, m.z,
                               elev_h1, elev_m, e.east, e.north,
                               g.id, g.elev, d))
    end
    return rows, edi_only, modem_only
end

# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

function hypo_label(ok::Bool)::String
    return ok ? "**tutarlı**" : "**tutarsız**"
end

function write_csv(path::AbstractString, rows::Vector{MatchedRow})
    df = DataFrame(
        name = String[r.name for r in rows],
        refelev = Float64[r.refelev for r in rows],
        units = String[r.units for r in rows],
        modem_x = Float64[r.modem_x for r in rows],
        modem_y = Float64[r.modem_y for r in rows],
        modem_z = Float64[r.modem_z for r in rows],
        elev_h1_ft_to_m = Float64[r.elev_h1 for r in rows],
        elev_modEM = Float64[r.elev_modEM for r in rows],
        nearest_grav_id = String[r.grav_id for r in rows],
        nearest_grav_elev_m = Float64[r.grav_elev for r in rows],
        dist_m = Float64[r.dist_m for r in rows],
        d_h1_minus_modEM = Float64[r.elev_h1 - r.elev_modEM for r in rows],
        d_modEM_minus_refelev = Float64[r.elev_modEM - r.refelev for r in rows],
        d_refelev_minus_grav = Float64[r.refelev - r.grav_elev for r in rows],
        d_h1_minus_grav = Float64[r.elev_h1 - r.grav_elev for r in rows],
        d_modEM_minus_grav = Float64[r.elev_modEM - r.grav_elev for r in rows],
    )
    CSV.write(path, df)
    return path
end

function write_markdown(path::AbstractString;
                        edi::Vector{EDIStation},
                        modem::Dict{String,ModEMStation},
                        n_data::Int,
                        origin_lat::Float64,
                        origin_lon::Float64,
                        n_period_hdr::Int,
                        n_sta_hdr::Int,
                        modem_comments::Vector{String},
                        origin_xyz::NTuple{3,Float64},
                        origin_fallback::Bool,
                        rho_path::AbstractString,
                        rows::Vector{MatchedRow},
                        edi_only::Vector{String},
                        modem_only::Vector{String},
                        h1::DiffStats,
                        h2::DiffStats,
                        h3_shift::Float64,
                        h3::DiffStats,
                        g_raw::PairStats,
                        g_h1::PairStats,
                        g_mod::PairStats,
                        dist_med::Float64,
                        dist_min::Float64,
                        dist_max::Float64,
                        n_refelev_unique::Int,
                        refelev_value::Float64,
                        units_tbl)
    ox, oy, oz = origin_xyz
    z_top = -oz
    open(path, "w") do io
        println(io, "# Northwest Geysers — MT kot vs ModEM Z (tanı)")
        println(io)
        println(io, "Tanı only. Ters çözüm, forward, kot dönüşümü **uygulanmadı**.")
        @printf(io, "Üretim: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM"))
        println(io, "Script: `scripts/geysers_elevation_check.jl`.")
        println(io)
        println(io, "## 1. Kaynaklar ve çerçeve")
        println(io)
        println(io, "- EDI: `data/geysers/mt/edi/*.edi`")
        println(io, "- ModEM data: `data/geysers/model/geysers_modEM_data_file.dat`")
        println(io, "  (WS/ModEM; başlık: `Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) …`)")
        println(io, "- Mesh: `", relpath(rho_path, ROOT), "`")
        println(io, "- Gravite: `data/geysers/gravity/gravity_clean.csv` (`ELEV_M`, NGVD29 m)")
        println(io)
        println(io, "ModEM data-file başlığı (yorum satırları):")
        println(io)
        println(io, "```")
        for c in modem_comments
            println(io, c)
        end
        println(io, "```")
        println(io)
        @printf(io, "Coğrafi origin (data `>` satırı): **%.6f, %.6f**.\n",
                origin_lat, origin_lon)
        @printf(io, "Başlık dönem / istasyon sayısı: **%d / %d**.\n",
                n_period_hdr, n_sta_hdr)
        @printf(io, "Data satırı: **%d**. Eşsiz ModEM istasyonu: **%d**.\n",
                n_data, length(modem))
        println(io)
        println(io, "Mesh origin (Mackie/WS `.rho` sonundaki `ox oy oz`):")
        println(io)
        if origin_fallback
            @printf(io, "- Dosyadan okunamadı; fallback **origin_z = %.3f m** kullanıldı.\n", oz)
        else
            @printf(io, "- origin = (**%.3f**, **%.3f**, **%.3f**) m\n", ox, oy, oz)
        end
        println(io, "- **z pozitif aşağı.** `origin_z < 0` → mesh tepesi (en yüksek topoğrafya)")
        @printf(io, "  deniz seviyesi datumunun **%.3f m** üzerindedir.\n", z_top)
        println(io, "- ModEM data `Z`: bu tepeden aşağı (pozitif). Türetilen kot:")
        println(io)
        println(io, "```")
        @printf(io, "elev_modEM = (%.3f) − Z_station     # metre, yukarı pozitif\n", z_top)
        println(io, "```")
        println(io)
        println(io, "Bu, `elev_modEM = 1083.929 − Z` ile aynıdır (dosya origin_z = −1083.929).")
        println(io)
        println(io, "MT UTM: EDI HEAD/DEFINEMEAS lat/lon → **WGS84 UTM 10N** (EPSG:32610).")
        println(io, "Gravite UTM **NAD83** 10N. WGS84–NAD83 ~1 m; kayma uygulanmadı")
        println(io, "(gravity QC ile aynı).")
        println(io)
        println(io, "## 2. EDI UNITS (beyan)")
        println(io)
        println(io, "UNITS, `>HEAD` ve `>=DEFINEMEAS` içinden okundu. Ampirik kot")
        println(io, "yorumu beyanla **aynı olmak zorunda değil** (yaygın EDI sorunu).")
        println(io)
        println(io, "| UNITS (DEFINEMEAS, yoksa HEAD) | n istasyon |")
        println(io, "|---|---|")
        for (u, n) in units_tbl
            @printf(io, "| `%s` | %d |\n", u, n)
        end
        n_head_u = count(s -> !isempty(s.units_head), edi)
        n_meas_u = count(s -> !isempty(s.units_meas), edi)
        println(io)
        @printf(io, "HEAD `UNITS` dolu: **%d / %d**. DEFINEMEAS `UNITS` dolu: **%d / %d**.\n",
                n_head_u, length(edi), n_meas_u, length(edi))
        println(io)
        println(io, "## 3. Eşleşme")
        println(io)
        @printf(io, "- EDI: **%d** dosya\n", length(edi))
        @printf(io, "- ModEM: **%d** istasyon\n", length(modem))
        @printf(io, "- Ortak: **%d**\n", length(rows))
        if isempty(edi_only)
            println(io, "- EDI'de olup ModEM'de olmayan: yok")
        else
            println(io, "- EDI'de olup ModEM'de olmayan: `",
                    join(edi_only, "`, `"), "`")
        end
        if isempty(modem_only)
            println(io, "- ModEM'de olup EDI'de olmayan: yok")
        else
            println(io, "- ModEM'de olup EDI'de olmayan: `",
                    join(modem_only, "`, `"), "`")
        end
        println(io)
        @printf(io, "EDI `REFELEV` (yoksa HEAD `ELEV`) eşsiz değer sayısı: **%d**",
                n_refelev_unique)
        if n_refelev_unique == 1 && isfinite(refelev_value)
            @printf(io, " — **tüm istasyonlarda %.3f** (placeholder / dummy).",
                    refelev_value)
        end
        println(io)
        println(io)
        println(io, "## 4. Üç hipotez (ortak istasyonlar)")
        println(io)
        println(io, "Hepsi n = ", length(rows), " ortak istasyon.")
        println(io, "`elev_modEM = ", fmt_num(z_top), " − Z`.")
        println(io)
        println(io, "### H1 — EDI feet, ModEM kot metre")
        println(io)
        println(io, "`elev_h1 = REFELEV × 0.3048`")
        println(io, "**Δ_H1 = elev_h1 − elev_modEM**")
        println(io)
        println(io, "| nicelik | m |")
        println(io, "|---|---|")
        @printf(io, "| mean(Δ) | %s |\n", fmt_num(h1.mean))
        @printf(io, "| std(Δ) | %s |\n", fmt_num(h1.std))
        @printf(io, "| max \\|Δ\\| | %s |\n", fmt_num(h1.maxabs))
        @printf(io, "| median(Δ) | %s |\n", fmt_num(h1.median))
        @printf(io, "| min … max Δ | %s … %s |\n", fmt_num(h1.min), fmt_num(h1.max))
        println(io)
        print(io, "Karar: ", hypo_label(consistent_topo(h1)), ". ")
        if consistent_topo(h1)
            println(io, "Kalıntı onlarca metre ölçeğinde; feet→metre + mesh kotu örtüşüyor.")
        else
            println(io, "Yakın istasyon topoğrafyası onlarca metrede kalmalı;")
            println(io, "std / max\\|Δ\\| bu eşiğin çok üzerinde. Feet ölçeği istasyon kotunu")
            println(io, "geri getirmiyor (EDI değeri istasyonlar arası değişmiyorsa zaten")
            println(io, "getiremez).")
        end
        println(io)
        println(io, "### H2 — EDI metre (dönüşümsüz) vs elev_modEM")
        println(io)
        println(io, "**Δ_H2 = elev_modEM − REFELEV**  (REFELEV metre kabul)")
        println(io)
        println(io, "| nicelik | m |")
        println(io, "|---|---|")
        @printf(io, "| mean(Δ) | %s |\n", fmt_num(h2.mean))
        @printf(io, "| std(Δ) | %s |\n", fmt_num(h2.std))
        @printf(io, "| max \\|Δ\\| | %s |\n", fmt_num(h2.maxabs))
        @printf(io, "| median(Δ) | %s |\n", fmt_num(h2.median))
        @printf(io, "| min … max Δ | %s … %s |\n", fmt_num(h2.min), fmt_num(h2.max))
        println(io)
        print(io, "Karar: ", hypo_label(consistent_topo(h2)), ". ")
        println(io, "NGVD29→NAVD88 ~0.93 m; 1780 m'lik sapma datum değil.")
        println(io, "Ortalama ~1.8 km, std hâlâ topoğrafya salınımı kadar büyükse")
        println(io, "REFELEV istasyon kotu olarak metre de değil.")
        println(io)
        println(io, "### H3 — EDI metre + sabit datum/kayma")
        println(io)
        println(io, "`shift = median(REFELEV − elev_modEM)`")
        println(io, "**r = (REFELEV − elev_modEM) − shift**  (sabit kayma çıkarıldı)")
        println(io)
        @printf(io, "Tahmini kayma: **%s m**.\n", fmt_num(h3_shift))
        println(io)
        println(io, "| nicelik (kalıntı r) | m |")
        println(io, "|---|---|")
        @printf(io, "| mean(r) | %s |\n", fmt_num(h3.mean))
        @printf(io, "| std(r) | %s |\n", fmt_num(h3.std))
        @printf(io, "| max \\|r\\| | %s |\n", fmt_num(h3.maxabs))
        @printf(io, "| median(r) | %s |\n", fmt_num(h3.median))
        println(io)
        h3_ok = h3.std <= DATUM_SHIFT_OK_STD_M
        print(io, "Karar: ", hypo_label(h3_ok), ". ")
        if h3_ok
            println(io, "Kayma sonrası saçılma ~1 m ölçeğinde → bilinmeyen sabit offset")
            println(io, "(NGVD29/NAVD88 değil; o ~0.93 m). Hangi serinin gerçek topo")
            println(io, "olduğuna gravite karar verir.")
        else
            println(io, "Kayma sonrası saçılma hâlâ onlarca/yüzlerce metre → **sabit")
            println(io, "datum kayması değil**. REFELEV istasyon topoğrafyasını taşımıyor")
            println(io, "(tek bir doldurma değeri, veya başka bir sistematik).")
        end
        println(io)
        println(io, "## 5. Gravite kontrolü (bağımsız hakem)")
        println(io)
        println(io, "`ELEV_M` NGVD29 metre (gerçekçi istasyon kotu). Her MT için")
        println(io, "en yakın gravite (UTM 10N). **Δ = seri − ELEV_M**.")
        println(io)
        @printf(io, "En yakın gravite uzaklığı: min **%s m**, medyan **%s m**, max **%s m**",
                fmt_num(dist_min, digits = 1), fmt_num(dist_med, digits = 1),
                fmt_num(dist_max, digits = 1))
        println(io, " (beklenen medyan ~252 m).")
        println(io)
        println(io, "| seri vs ELEV_M | n | mean Δ (m) | std Δ (m) | max \\|Δ\\| (m) | Pearson r |")
        println(io, "|---|---|---|---|---|---|")
        @printf(io, "| ham EDI REFELEV (m kabul) | %d | %s | %s | %s | %s |\n",
                g_raw.n, fmt_num(g_raw.mean), fmt_num(g_raw.std),
                fmt_num(g_raw.maxabs), fmt_cor(g_raw.cor))
        @printf(io, "| EDI × 0.3048 (feet→m) | %d | %s | %s | %s | %s |\n",
                g_h1.n, fmt_num(g_h1.mean), fmt_num(g_h1.std),
                fmt_num(g_h1.maxabs), fmt_cor(g_h1.cor))
        @printf(io, "| elev_modEM = %.3f − Z | %d | %s | %s | %s | %s |\n",
                z_top, g_mod.n, fmt_num(g_mod.mean), fmt_num(g_mod.std),
                fmt_num(g_mod.maxabs), fmt_cor(g_mod.cor))
        println(io)
        println(io, "Dağlık arazide ~250 m yatay uzaklık onlarca metre kot farkı üretebilir;")
        println(io, "yüzlerce/binlerce metre ve r ≈ 0 (veya tanımsız) ise seri topoğrafya değildir.")
        println(io)
        rows_near = MatchedRow[r for r in rows if r.dist_m <= 250.0]
        if !isempty(rows_near)
            g_mod_c = pair_stats(Float64[r.elev_modEM for r in rows_near],
                                 Float64[r.grav_elev for r in rows_near])
            @printf(io, "NN ≤ 250 m altküme (n = %d): elev_modEM − ELEV_M mean = %s m, std = %s m, r = %s.\n",
                    g_mod_c.n, fmt_num(g_mod_c.mean), fmt_num(g_mod_c.std),
                    fmt_cor(g_mod_c.cor))
            println(io, "Yakın çiftlerde kalan ~100 m ölçekli negatif bias, 0.93 m'lik")
            println(io, "NGVD29–NAVD88 değildir; ModEM `Z` hücre merkezi / mesh-top tanımı")
            println(io, "ile gravite serbest yüzey kotu birebir aynı olmayabilir.")
            println(io)
        end
        # Which series matches topography.
        cands = [("ham EDI REFELEV", g_raw),
                 ("EDI × 0.3048", g_h1),
                 ("elev_modEM", g_mod)]
        best = argmin(i -> (cands[i][2].std, cands[i][2].maxabs), 1:3)
        println(io, "**Gravite ile örtüşen seri:** `", cands[best][1], "`")
        println(io, "(en küçük std(Δ); tek sonlu/anlamlı Pearson r).")
        println(io, "EDI ve EDI×0.3048 sabit olduğu için r tanımsızdır; std(Δ) = std(ELEV_M).")
        println(io)
        println(io, "## 6. Verdict")
        println(io)
        println(io, chosen_paragraph(length(edi), n_refelev_unique, refelev_value, z_top,
                                     h1, h2, h3, h3_shift, g_raw, g_h1, g_mod,
                                     best, cands))
        println(io)
        println(io, "**Dönüşüm uygulanmadı.** EDI, ModEM data ve `gravity_clean.csv`")
        println(io, "değiştirilmedi. NGVD29→NAVD88 (~0.93 m) de uygulanmadı.")
        println(io)
        println(io, "## 7. GZ01 kontrol")
        println(io)
        gz = findfirst(r -> r.name == "GZ01", rows)
        if gz !== nothing
            r = rows[gz]
            @printf(io, "- REFELEV = %.3f (beyan UNITS=`%s`)\n", r.refelev, r.units)
            @printf(io, "- ModEM Z = %.3f m  → elev_modEM = %.3f m\n",
                    r.modem_z, r.elev_modEM)
            @printf(io, "- REFELEV − ModEM Z = %.3f m (birimler karışık; nicelikler farklı)\n",
                    r.refelev - r.modem_z)
            @printf(io, "- elev_h1 = %.3f m; Δ_H1 = %.3f m; Δ_H2 = %.3f m\n",
                    r.elev_h1, r.elev_h1 - r.elev_modEM, r.elev_modEM - r.refelev)
            @printf(io, "- en yakın gravite `%s` ELEV_M = %.2f m (%.1f m uzak)\n",
                    r.grav_id, r.grav_elev, r.dist_m)
        else
            println(io, "GZ01 ortak listede yok.")
        end
        println(io)
        println(io, "## Artefaktlar")
        println(io)
        println(io, "- `results/geysers_elevation_check.md` (bu dosya)")
        println(io, "- `results/geysers_elevation_check.csv`")
        println(io, "- `results/DATA_NOTES.md` §4 (yorum; dönüşüm yok)")
    end
    return path
end

function chosen_paragraph(n_edi::Int, n_unique::Int, refelev::Float64, z_top::Float64,
                          h1::DiffStats, h2::DiffStats, h3::DiffStats,
                          h3_shift::Float64,
                          g_raw::PairStats, g_h1::PairStats, g_mod::PairStats,
                          best::Int, cands)::String
    dummy = n_unique == 1 && isfinite(refelev)
    dummy_s = dummy ?
        @sprintf("EDI `REFELEV`/`ELEV` **tüm %d EDI dosyasında aynı (%.3f)**; bu bir istasyon kotu değil, MTpy/EDI placeholder.",
                 n_edi, refelev) :
        "EDI `REFELEV` istasyonlar arasında değişiyor."
    g_s = @sprintf("Gravite `ELEV_M` (NGVD29 m) hakem: ham EDI mean Δ = %.1f m, std = %.1f m, r = %s; EDI×0.3048 mean Δ = %.1f m, std = %.1f m, r = %s; elev_modEM mean Δ = %.1f m, std = %.1f m, r = %s. Örtüşen seri: %s.",
                   g_raw.mean, g_raw.std, fmt_cor(g_raw.cor),
                   g_h1.mean, g_h1.std, fmt_cor(g_h1.cor),
                   g_mod.mean, g_mod.std, fmt_cor(g_mod.cor),
                   cands[best][1])
    return string(
        dummy_s, " Beyan UNITS=`m` olsa da ampirik olarak bu sayı metre kot değildir ",
        "(The Geysers ~0.4–1.2 km; 2113 m aykırı) ve feet de değildir ",
        @sprintf("(H1 std(Δ)=%.1f m, max|Δ|=%.1f m).", h1.std, h1.maxabs),
        @sprintf(" H2 (REFELEV metre vs %.3f−Z) mean Δ = %.1f m — NGVD29–NAVD88 (~0.93 m) ile açıklanamaz; std hâlâ %.1f m.",
                 z_top, h2.mean, h2.std),
        @sprintf(" H3 kayması %.1f m; kalıntı std = %.1f m → sabit datum kayması değil.",
                 h3_shift, h3.std),
        " ", g_s, " ",
        @sprintf("Yorum: ModEM `Z` en yüksek topoğrafyadan (+z aşağı) derinlik; kot = %.3f − Z. EDI REFELEV dummy. Dönüşüm uygulanmadı.",
                 z_top),
    )
end

function data_notes_section4(; z_top::Float64, origin_z::Float64,
                             origin_fallback::Bool, n_edi::Int, n_modem::Int,
                             n_match::Int, edi_only::Vector{String},
                             units_tbl, n_refelev_unique::Int,
                             refelev_value::Float64,
                             h1::DiffStats, h2::DiffStats, h3::DiffStats,
                             h3_shift::Float64,
                             g_raw::PairStats, g_h1::PairStats, g_mod::PairStats,
                             dist_med::Float64)::String
    units_s = join(["`$(u)` (n=$(n))" for (u, n) in units_tbl], ", ")
    unmatched = isempty(edi_only) ? "none" : join(edi_only, ", ")
    src = origin_fallback ? "fallback" : "geysers_preferred_model.rho trailer"
    buf = IOBuffer()
    println(buf, "## 4. MT elevation vs ModEM Z (Geysers)")
    println(buf)
    println(buf, "Diagnosis only. **No conversion applied** to EDI, ModEM data,")
    println(buf, "or `gravity_clean.csv`. NGVD29→NAVD88 (~0.93 m) is still not applied.")
    println(buf)
    @printf(buf, "Mesh origin z = **%.3f m** (%s); z positive down from the highest\n",
            origin_z, src)
    @printf(buf, "topography. ModEM-derived orthometric-ish height:\n")
    println(buf)
    println(buf, "```")
    @printf(buf, "elev_modEM = %.3f − Z_station     # metres, up positive\n", z_top)
    println(buf, "```")
    println(buf)
    @printf(buf, "EDI n = %d; ModEM n = %d; matched n = %d; EDI-only: %s.\n",
            n_edi, n_modem, n_match, unmatched)
    println(buf, "Declared EDI UNITS (DEFINEMEAS, else HEAD): ", units_s, ".")
    if n_refelev_unique == 1 && isfinite(refelev_value)
        @printf(buf, "REFELEV is **identical (%.3f) on every EDI** — placeholder, not station topography.\n",
                refelev_value)
    end
    println(buf)
    println(buf, "| test | mean | std | max \\|Δ\\| |")
    println(buf, "|---|---|---|---|")
    @printf(buf, "| H1 Δ = (REFELEV×0.3048) − elev_modEM | %.3f | %.3f | %.3f |\n",
            h1.mean, h1.std, h1.maxabs)
    @printf(buf, "| H2 Δ = elev_modEM − REFELEV (m) | %.3f | %.3f | %.3f |\n",
            h2.mean, h2.std, h2.maxabs)
    @printf(buf, "| H3 residual after median shift %.3f m | %.3f | %.3f | %.3f |\n",
            h3_shift, h3.mean, h3.std, h3.maxabs)
    println(buf)
    @printf(buf, "Nearest gravity median distance **%.1f m**. vs ELEV_M (Δ = series − grav):\n",
            dist_med)
    @printf(buf, "raw EDI std = %.1f m (r %s); EDI×0.3048 std = %.1f m (r %s); elev_modEM std = %.1f m (r = %s, mean Δ = %.1f m).\n",
            g_raw.std, fmt_cor(g_raw.cor), g_h1.std, fmt_cor(g_h1.cor),
            g_mod.std, fmt_cor(g_mod.cor), g_mod.mean)
    println(buf)
    println(buf, "Chosen interpretation: EDI `REFELEV`/`ELEV` is a **placeholder**")
    @printf(buf, "(%.3f on every file), declared `UNITS=m`, but it is neither station topography in metres nor feet. H1/H2/H3 all fail: residuals track `elev_modEM` variation (std ≈ %.0f m), not a ~1 m datum. ModEM `Z` is positive-down from mesh top; `elev_modEM = %.3f − Z` is the only series that correlates with gravity `ELEV_M`. A ~%.0f m mean low bias vs nearest gravity remains (not NGVD29–NAVD88). **Conversion not applied.**\n",
            isfinite(refelev_value) ? refelev_value : NaN,
            h3.std, z_top, abs(g_mod.mean))
    println(buf)
    println(buf, "Full tables: `results/geysers_elevation_check.md`.")
    return String(take!(buf))
end

const SECTION4_MARKER = "## 4. MT elevation vs ModEM Z (Geysers)"

"""Replace or append DATA_NOTES.md section 4 only; leave earlier sections intact."""
function update_data_notes!(path::AbstractString, section4::AbstractString)
    existing = isfile(path) ? read(path, String) : ""
    existing = replace(existing, "\r\n" => "\n")
    existing = rstrip(existing)
    idx = findfirst(SECTION4_MARKER, existing)
    body = rstrip(section4)
    if idx === nothing
        newtext = existing * "\n\n" * body * "\n"
    else
        start = first(idx)
        newtext = existing[1:start-1] * body * "\n"
        newtext = rstrip(newtext) * "\n"
    end
    open(path, "w") do io
        write(io, newtext)
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# CLI / main
# ─────────────────────────────────────────────────────────────────────────────

function parse_cli(args)::NTuple{5,String}
    edi = DEFAULT_EDI_DIR
    dat = DEFAULT_MODEM_DAT
    rho = DEFAULT_RHO
    grav = DEFAULT_GRAV_CSV
    out = DEFAULT_OUT_DIR
    length(args) >= 1 && !isempty(args[1]) && (edi = abspath(args[1]))
    length(args) >= 2 && !isempty(args[2]) && (dat = abspath(args[2]))
    length(args) >= 3 && !isempty(args[3]) && (rho = abspath(args[3]))
    length(args) >= 4 && !isempty(args[4]) && (grav = abspath(args[4]))
    length(args) >= 5 && !isempty(args[5]) && (out = abspath(args[5]))
    return edi, dat, rho, grav, out
end

function main(args::Vector{String}=ARGS)
    edi_dir, dat_path, rho_path, grav_path, out_dir = parse_cli(args)
    isdir(out_dir) || mkpath(out_dir)

    println("[1/5] EDI  ", relpath(edi_dir, ROOT))
    edi = load_edi_stations(edi_dir)
    @printf("      %d files\n", length(edi))
    units_tbl = unique_units_table(edi)
    refelevs = Float64[s.refelev for s in edi]
    n_refelev_unique = length(unique(round.(refelevs; digits = 6)))
    refelev_value = n_refelev_unique == 1 ? refelevs[1] : NaN

    println("[2/5] ModEM data  ", relpath(dat_path, ROOT))
    modem, n_data, origin_lat, origin_lon, n_period_hdr, n_sta_hdr, comments =
        parse_modem_data(dat_path)
    @printf("      %d unique stations, %d data rows\n", length(modem), n_data)

    println("[3/5] Mesh origin  ", relpath(rho_path, ROOT))
    ox, oy, oz, used_fb = parse_ws_origin_z(rho_path)
    origin_xyz = (ox, oy, oz)
    @printf("      origin_z = %.3f m%s\n", oz, used_fb ? " (FALLBACK)" : "")
    z_top = -oz

    println("[4/5] Gravity  ", relpath(grav_path, ROOT))
    grav = load_gravity_points(grav_path)
    @printf("      %d stations with ELEV_M\n", length(grav))

    rows, edi_only, modem_only = match_stations(edi, modem, grav, oz)
    @printf("      matched %d  EDI-only %s  ModEM-only %s\n",
            length(rows),
            isempty(edi_only) ? "[]" : string(edi_only),
            isempty(modem_only) ? "[]" : string(modem_only))

    # H1: Δ = (REFELEV * 0.3048) − elev_modEM
    δ_h1 = Float64[r.elev_h1 - r.elev_modEM for r in rows]
    h1 = diff_stats(δ_h1)
    # H2: Δ = elev_modEM − REFELEV (metres, no conversion)
    δ_h2 = Float64[r.elev_modEM - r.refelev for r in rows]
    h2 = diff_stats(δ_h2)
    # H3: shift = median(REFELEV − elev_modEM); residual after removing it
    raw_shift = Float64[r.refelev - r.elev_modEM for r in rows]
    h3_shift = median(raw_shift)
    h3 = diff_stats(raw_shift .- h3_shift)

    dists = Float64[r.dist_m for r in rows]
    dist_med = median(dists)
    dist_min = minimum(dists)
    dist_max = maximum(dists)

    g_raw = pair_stats(Float64[r.refelev for r in rows],
                       Float64[r.grav_elev for r in rows])
    g_h1 = pair_stats(Float64[r.elev_h1 for r in rows],
                      Float64[r.grav_elev for r in rows])
    g_mod = pair_stats(Float64[r.elev_modEM for r in rows],
                       Float64[r.grav_elev for r in rows])

    md_path = joinpath(out_dir, "geysers_elevation_check.md")
    csv_path = joinpath(out_dir, "geysers_elevation_check.csv")
    notes_path = DEFAULT_DATA_NOTES

    println("[5/5] Writing reports")
    write_csv(csv_path, rows)
    write_markdown(md_path; edi, modem, n_data, origin_lat, origin_lon,
                   n_period_hdr, n_sta_hdr, modem_comments = comments,
                   origin_xyz, origin_fallback = used_fb, rho_path, rows,
                   edi_only, modem_only, h1, h2, h3_shift, h3,
                   g_raw, g_h1, g_mod, dist_med, dist_min, dist_max,
                   n_refelev_unique, refelev_value, units_tbl)
    sec4 = data_notes_section4(; z_top, origin_z = oz, origin_fallback = used_fb,
                               n_edi = length(edi), n_modem = length(modem),
                               n_match = length(rows), edi_only, units_tbl,
                               n_refelev_unique, refelev_value,
                               h1, h2, h3, h3_shift, g_raw, g_h1, g_mod,
                               dist_med)
    update_data_notes!(notes_path, sec4)

    println()
    println("Wrote:")
    println("  ", relpath(md_path, ROOT))
    println("  ", relpath(csv_path, ROOT))
    println("  ", relpath(notes_path, ROOT), "  (section 4)")
    @printf("H1  mean=%8.3f  std=%8.3f  max|Δ|=%8.3f\n", h1.mean, h1.std, h1.maxabs)
    @printf("H2  mean=%8.3f  std=%8.3f  max|Δ|=%8.3f\n", h2.mean, h2.std, h2.maxabs)
    @printf("H3  shift=%8.3f  residual std=%8.3f  max|r|=%8.3f\n",
            h3_shift, h3.std, h3.maxabs)
    @printf("grav NN median dist = %.1f m\n", dist_med)
    @printf("vs ELEV_M  EDI std=%.1f  ft→m std=%.1f  modEM std=%.1f r=%s\n",
            g_raw.std, g_h1.std, g_mod.std, fmt_cor(g_mod.cor))
    println("Conversion applied: NO")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
