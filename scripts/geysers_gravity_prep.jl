#!/usr/bin/env julia
#=
geysers_gravity_prep.jl — Northwest Geysers gravity parse + QC + regional
residual. Preparation only: no inversion, no forward modelling, no U-Net.

Source: USGS GY_grav_all_GRX_190726_USGS.txt (Peacock et al. 2020,
DOI 10.5066/P94D21UL), 534 gravity stations, UTM Zone 10N NAD83.

Usage (from project root):
    julia --project=. scripts/geysers_gravity_prep.jl
    julia --project=. scripts/geysers_gravity_prep.jl \
        data/geysers/gravity/GY_grav_all_GRX_190726_USGS.txt \
        data/geysers/mt/edi \
        results \
        data/geysers/gravity/gravity_clean.csv

Outputs:
    data/geysers/gravity/gravity_clean.csv
    results/geysers_gravity_qc.md
    results/geysers_gravity_map.png

Vertical datum (NGVD29 → NAVD88 → model z) is documented, not applied.
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics

const DEFAULT_GRAV_PATH = joinpath(ROOT, "data", "geysers", "gravity",
                                   "GY_grav_all_GRX_190726_USGS.txt")
const DEFAULT_EDI_DIR = joinpath(ROOT, "data", "geysers", "mt", "edi")
const DEFAULT_OUT_DIR = joinpath(ROOT, "results")
const DEFAULT_CLEAN_CSV = joinpath(ROOT, "data", "geysers", "gravity",
                                   "gravity_clean.csv")

# Duplicate if hypot(Δx, Δy) ≤ this (metres). Documented in the QC report.
const DUP_TOL_M = 0.5

# Outlier: |value − median| > OUTLIER_NSTD × std (not MAD).
const OUTLIER_NSTD = 4.0

const REQUIRED_COLS = ("UTM10x_NAD83", "UTM10y_NAD83", "ELEV_M", "CBA", "ISO")
const OPTIONAL_COLS = ("ID", "LAT", "LONG", "OG", "FTC", "TTC")

# WGS84 → UTM Zone 10N (same series as scripts/geysers_dimensionality.jl)
const WGS84_A = 6378137.0
const WGS84_F = 1 / 298.257223563
const UTM_K0 = 0.9996
const UTM_FALSE_EASTING = 500_000.0
const UTM_ZONE10_LON0 = -123.0

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""One gravity station after numeric parse. `usable` requires finite x, y, CBA."""
struct GravStation
    id::String
    lat::Float64
    lon::Float64
    x::Float64
    y::Float64
    elev::Float64
    cba::Float64
    iso::Float64
    og::Float64
    ftc::Float64
    ttc::Float64
    line_no::Int
    usable::Bool
    parse_ok::Bool
    missing_fields::Vector{String}
end

struct MTSite
    site::String
    lat::Float64
    lon::Float64
    east::Float64
    north::Float64
end

struct DistStats
    n::Int
    min::Float64
    median::Float64
    max::Float64
    std::Float64
end

struct PolyFit
    degree::Int
    coef::Vector{Float64}          # scaled-coordinate coefficients
    x_mean::Float64
    y_mean::Float64
    scale::Float64
    pred::Vector{Float64}
    resid::Vector{Float64}
    rmse::Float64
end

struct GravQC
    comments::Vector{String}
    n_data_rows::Int
    n_parse_ok::Int
    n_corrupt::Int
    n_usable::Int
    n_dropped::Int
    stations::Vector{GravStation}  # usable only, in file order
    dropped_ids::Vector{String}
    cba::DistStats
    iso::DistStats
    ftc::DistStats
    ttc::DistStats
    ftc_plus_ttc::DistStats
    outlier_cba::BitVector
    outlier_iso::BitVector
    duplicate::BitVector
    n_outlier_cba::Int
    n_outlier_iso::Int
    n_duplicate::Int
    dup_groups::Vector{Vector{String}}
    fit1::PolyFit
    fit2::PolyFit
    cor_cba_iso::Float64
    cor_r1_iso::Float64
    cor_r2_iso::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
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

WGS84 geographic → UTM Zone 10N (EPSG:32610). Snyder (1987) USGS PP 1395
eqs. (8-9)–(8-11) and meridional arc (3-21). Copied (not imported) from
`scripts/geysers_dimensionality.jl` so this script stays standalone.
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

function colmap(header::AbstractVector{<:AbstractString})::Dict{String,Int}
    m = Dict{String,Int}()
    for (i, h) in enumerate(header)
        m[uppercase(strip(String(h)))] = i
    end
    return m
end

function cell(fields::Vector{String}, cmap::Dict{String,Int}, name::String)::String
    k = uppercase(name)
    haskey(cmap, k) || return ""
    i = cmap[k]
    return (1 <= i <= length(fields)) ? fields[i] : ""
end

# ─────────────────────────────────────────────────────────────────────────────
# Gravity parser
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_gravity_file(path) -> (comments, header, stations_all)

Skip `#` comment lines, tolerate CRLF, parse the CSV header and every data
row. Rows that fail numeric conversion of required fields are kept in
`stations_all` with `parse_ok = false` so they can be counted, not silently
dropped until the usable filter.
"""
function parse_gravity_file(path::AbstractString)
    isfile(path) || error("gravity file not found: $path")
    lines = read_lines_crlf(path)
    comments = String[]
    header = String[]
    cmap = Dict{String,Int}()
    stations = GravStation[]
    header_line = 0

    for (lineno, line) in enumerate(lines)
        t = strip(line)
        if startswith(t, "#")
            push!(comments, t)
            continue
        end
        isempty(t) && continue
        if isempty(header)
            header = String[strip(x) for x in split(t, ',')]
            cmap = colmap(header)
            header_line = lineno
            continue
        end
        fields = String[strip(x) for x in split(t, ',')]
        id = cell(fields, cmap, "ID")
        isempty(id) && (id = @sprintf("row_%d", lineno))
        lat = parse_float_cell(cell(fields, cmap, "LAT"))
        lon = parse_float_cell(cell(fields, cmap, "LONG"))
        x = parse_float_cell(cell(fields, cmap, "UTM10x_NAD83"))
        y = parse_float_cell(cell(fields, cmap, "UTM10y_NAD83"))
        elev = parse_float_cell(cell(fields, cmap, "ELEV_M"))
        cba = parse_float_cell(cell(fields, cmap, "CBA"))
        iso = parse_float_cell(cell(fields, cmap, "ISO"))
        og = parse_float_cell(cell(fields, cmap, "OG"))
        ftc = parse_float_cell(cell(fields, cmap, "FTC"))
        ttc = parse_float_cell(cell(fields, cmap, "TTC"))

        missing = String[]
        for (name, v) in (("UTM10x_NAD83", x), ("UTM10y_NAD83", y),
                          ("ELEV_M", elev), ("CBA", cba), ("ISO", iso))
            isfinite(v) || push!(missing, name)
        end
        # Optional numeric fields: count as corrupt only if the cell is
        # present and non-empty but non-numeric.
        for (name, v) in (("OG", og), ("FTC", ftc), ("TTC", ttc))
            raw = cell(fields, cmap, name)
            if !isempty(strip(raw)) && !isfinite(v)
                push!(missing, name)
            end
        end
        parse_ok = isempty(missing)
        usable = isfinite(x) && isfinite(y) && isfinite(cba)
        push!(stations, GravStation(id, lat, lon, x, y, elev, cba, iso, og,
                                    ftc, ttc, lineno, usable, parse_ok, missing))
    end
    isempty(header) && error("$path: no CSV header after # comments")
    @info "Parsed gravity CSV" path=path header_line=header_line n_header=length(header)
    return comments, header, stations
end

# ─────────────────────────────────────────────────────────────────────────────
# MT stations (EDI HEAD / DEFINEMEAS only — coordinates, not responses)
# ─────────────────────────────────────────────────────────────────────────────

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

"""Parse LAT/LON from an EDI `>HEAD` / `>=DEFINEMEAS` block. No Z / spectra."""
function parse_edi_coords(path::AbstractString)::Union{MTSite,Nothing}
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
    (isfinite(lat) && isfinite(lon)) || return nothing
    east, north = wgs84_to_utm10n(lat, lon)
    site = uppercase(get(head, "DATAID", splitext(basename(path))[1]))
    return MTSite(site, lat, lon, east, north)
end

function load_mt_sites(edi_dir::AbstractString)::Vector{MTSite}
    isdir(edi_dir) || error("EDI directory not found: $edi_dir")
    files = sort(String[joinpath(edi_dir, f) for f in readdir(edi_dir)
                        if lowercase(splitext(f)[2]) == ".edi" && !startswith(f, ".")])
    isempty(files) && error("no .edi files in $edi_dir")
    sites = MTSite[]
    for f in files
        s = parse_edi_coords(f)
        s === nothing && error("no LAT/LON in $f")
        push!(sites, s)
    end
    return sites
end

# ─────────────────────────────────────────────────────────────────────────────
# QC + regional fit
# ─────────────────────────────────────────────────────────────────────────────

function dist_stats(v::AbstractVector{<:Real})::DistStats
    w = Float64[x for x in v if isfinite(x)]
    isempty(w) && return DistStats(0, NaN, NaN, NaN, NaN)
    return DistStats(length(w), minimum(w), median(w), maximum(w), std(w))
end

"""Flag |x − median(x)| > nstd × std(x). Non-finite values are not flagged."""
function outlier_flags(v::AbstractVector{<:Real}; nstd::Float64 = OUTLIER_NSTD)::BitVector
    w = Float64[x for x in v if isfinite(x)]
    flags = falses(length(v))
    length(w) < 2 && return flags
    med = median(w)
    s = std(w)
    (!isfinite(s) || s == 0) && return flags
    thresh = nstd * s
    for i in eachindex(v)
        isfinite(v[i]) || continue
        flags[i] = abs(v[i] - med) > thresh
    end
    return flags
end

"""
Duplicate groups: stations whose UTM (x, y) lie within `tol_m` metres
(Euclidean). Both members of a pair are flagged. Criterion: hypot(Δx,Δy) ≤ tol.
"""
function duplicate_flags(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                         tol_m::Float64 = DUP_TOL_M)
    n = length(x)
    flags = falses(n)
    groups = Vector{Vector{Int}}()
    parent = collect(1:n)
    function findroot(i::Int)::Int
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end
    function unite!(i::Int, j::Int)
        a, b = findroot(i), findroot(j)
        a != b && (parent[b] = a)
        return nothing
    end
    for i in 1:n-1
        for j in (i + 1):n
            hypot(x[i] - x[j], y[i] - y[j]) <= tol_m && unite!(i, j)
        end
    end
    buckets = Dict{Int,Vector{Int}}()
    for i in 1:n
        r = findroot(i)
        push!(get!(buckets, r, Int[]), i)
    end
    for idxs in values(buckets)
        length(idxs) < 2 && continue
        for i in idxs
            flags[i] = true
        end
        push!(groups, sort(idxs))
    end
    return flags, groups
end

"""
    fit_cba_surface(x, y, cba; degree)

Least-squares polynomial in *centered and scaled* coordinates

    xs = (x − mean(x)) / s ,   ys = (y − mean(y)) / s

with `s = hypot(std(x), std(y))` (metres). Degree 1: `a + b xs + c ys`.
Degree 2: `a + b xs + c ys + d xs² + e xs ys + f ys²`. Residuals are in
mGal (same unit as CBA) because the left-hand side is unscaled CBA.
"""
function fit_cba_surface(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                         cba::AbstractVector{<:Real}; degree::Int)::PolyFit
    n = length(cba)
    n == length(x) == length(y) || throw(DimensionMismatch("x/y/cba length"))
    (degree == 1 || degree == 2) || error("degree must be 1 or 2")
    μx = mean(x)
    μy = mean(y)
    s = hypot(std(x), std(y))
    s <= 0 && error("degenerate coordinate scale")
    xs = (Float64.(x) .- μx) ./ s
    ys = (Float64.(y) .- μy) ./ s
    X = degree == 1 ?
        hcat(ones(n), xs, ys) :
        hcat(ones(n), xs, ys, xs .^ 2, xs .* ys, ys .^ 2)
    coef = X \ Float64.(cba)
    pred = X * coef
    resid = Float64.(cba) .- pred
    rmse = sqrt(mean(resid .^ 2))
    return PolyFit(degree, coef, μx, μy, s, pred, resid, rmse)
end

function pearson(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})::Float64
    mask = isfinite.(a) .& isfinite.(b)
    count(mask) < 3 && return NaN
    return cor(Float64.(a[mask]), Float64.(b[mask]))
end

function run_qc(comments::Vector{String}, all_st::Vector{GravStation})::GravQC
    n_data = length(all_st)
    n_ok = count(s -> s.parse_ok, all_st)
    n_corrupt = n_data - n_ok
    usable = GravStation[s for s in all_st if s.usable]
    dropped = String[s.id for s in all_st if !s.usable]
    n_use = length(usable)
    n_drop = length(dropped)

    cba = Float64[s.cba for s in usable]
    iso = Float64[s.iso for s in usable]
    ftc = Float64[s.ftc for s in usable]
    ttc = Float64[s.ttc for s in usable]
    x = Float64[s.x for s in usable]
    y = Float64[s.y for s in usable]
    fpt = Float64[s.ftc + s.ttc for s in usable]

    ocba = outlier_flags(cba)
    oiso = outlier_flags(iso)
    dup, groups = duplicate_flags(x, y)
    dup_ids = Vector{Vector{String}}()
    for g in groups
        push!(dup_ids, String[usable[i].id for i in g])
    end

    fit1 = fit_cba_surface(x, y, cba; degree = 1)
    fit2 = fit_cba_surface(x, y, cba; degree = 2)

    return GravQC(comments, n_data, n_ok, n_corrupt, n_use, n_drop, usable,
                  dropped, dist_stats(cba), dist_stats(iso), dist_stats(ftc),
                  dist_stats(ttc), dist_stats(fpt), ocba, oiso, dup,
                  count(ocba), count(oiso), count(dup), dup_ids,
                  fit1, fit2, pearson(cba, iso), pearson(fit1.resid, iso),
                  pearson(fit2.resid, iso))
end

# ─────────────────────────────────────────────────────────────────────────────
# Header conclusion (FTC/TTC vs CBA) — quote the file, do not guess
# ─────────────────────────────────────────────────────────────────────────────

function header_ftc_ttc_conclusion(comments::Vector{String})::String
    buf = IOBuffer()
    println(buf, "Dosya `#` başlık satırlarından birebir:")
    for c in comments
        if occursin(r"CBA|FTC|TTC|ELEV|Bouger|Bouguer|Terrain|NGVD"i, c)
            println(buf, "  `", c, "`")
        end
    end
    println(buf)
    println(buf, "Sonuç: başlık CBA'yı **Complete Bouguer Anomaly** olarak adlandırıyor.")
    println(buf, "Complete Bouguer, serbest-hava anomalisi + Bouguer levha + eğrilik")
    println(buf, "+ arazi düzeltmesini (terrain correction) içerir; yani FTC ve TTC")
    println(buf, "CBA'ya **zaten uygulanmıştır**. Başlık FTC'yi istasyondan 68 m'ye kadar")
    println(buf, "alan arazi düzeltmesi, TTC'yi 68 m–166.7 km arazi düzeltmesi olarak")
    println(buf, "tanımlar. Bu değerler CBA'ya tekrar eklenmemelidir.")
    println(buf)
    println(buf, "FGDC metadata (ScienceBase `gz_geysers_metadata.xml`, bu script")
    println(buf, "tahmin etmiyor, işlem kaydını aktarıyor): *\"Bouguer, curvature, and")
    println(buf, "terrain adjustments to a radial distance of 166.7 km were applied")
    println(buf, "to the free-air anomaly at each station to determine the complete")
    println(buf, "Bouguer anomalies at a standard crustal reduction density of")
    println(buf, "2,670 kg/m3 (Plouff, 1977).\"*")
    return String(take!(buf))
end

function datum_note_md()::String
    return """
## Dikey datum (NGVD29 → NAVD88 → model z) — **uygulanmadı**

`ELEV_M` dosya başlığında **NGVD29, metre** olarak kayıtlıdır
(`#ELEV_m Elevation (NGVD29, meters)`). Bu script dönüşümü **uygulamaz**;
`gravity_clean.csv` içindeki `ELEV_M` hâlâ NGVD29'dur.

### NGVD29 → NAVD88 (The Geysers, California)

NOAA NGS VERTCON 3.0 (NCAT 2.0 içinde, 20190601 grid'i) işareti:

```
height_NAVD88 = height_NGVD29 + Δ
height_NGVD29 = height_NAVD88 − Δ
Δ ≡ (NAVD88 − NGVD29)
```

Kaynak: NGS VERTCON işaret kılavuzu
<https://geodesy.noaa.gov/TOOLS/Vertcon/vert_sign.html>
ve NOAA Technical Report NOS NGS 68 (VERTCON 3.0). Grid değerleri
**NAVD88 eksi NGVD29**'dur; NGVD29 yüksekliğine **cebirsel olarak eklenir**.

NCAT REST sorguları (2026-08-27, `vertconVersion=3.0`):

| Konum | lat, lon | Δ = NAVD88 − NGVD29 |
|---|---|---|
| MT ayakizi (NW Geysers) | 38.82°, −122.78° | **+0.942 m** (sig ≈ 0.012 m) |
| Yakın istasyon | 38.79°, −122.82° | **+0.920 m** (sig ≈ 0.010 m) |
| Gravite bbox GB | 38.7227°, −123.1494° | **+0.941 m** |
| Gravite bbox KD | 39.2369°, −122.4561° | **+0.852 m** |

**Öneri (uygulanmadı):** The Geysers MT/gravite örtüşmesi için
`ELEV_NAVD88 ≈ ELEV_NGVD29 + 0.93 m` (aralık **+0.85 … +0.94 m**).
İşaret: NAVD88, bu bölgede NGVD29'dan **yüksektir** (Δ > 0). Güney California
literatüründeki −0.7…−1.0 m değerleri buraya ait değildir; Kuzey California
VERTCON 3.0 grid'i pozitiftir.

Araç: [NGS NCAT](https://geodesy.noaa.gov/NCAT/) /
[VERTCON 3.0](https://www.ngs.noaa.gov/VERTCON3/).

### NAVD88 → proje model z çerçevesi

Aktif (non-archive) notlarda sayısal bir NAVD88→model-z kayması yok.
`data/geysers/model/geysers_modEM_data_file.dat` WS/ModEM formatındadır:
`X(m) Y(m) Z(m)` yerel mesh koordinatıdır (referans ≈ 38.831979, −122.828190).
WS/ModEM'de **z pozitif aşağı**dır. GZ01 örneği: EDI `REFELEV=2113.200`,
ModEM `Z=330.000` — bu iki sayı aynı nicelik değildir; `ELEV_M`'ye sabit
eklemek ModEM z'ye geçmez.

İleride gravite istasyonlarını modele yerleştirirken:

1. `ELEV_M` (NGVD29) → NAVD88 (`+0.93 m`, yukarıdaki Δ).
2. Aynı ModEM orijini / topoğrafya kuralı ile yerel `Z`'ye çevir
   (`geysers_modEM_data_file.dat` ile tutarlı).
3. Gravite `ELEV_M`'yi ModEM `Z` sanma.
"""
end

# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

function clean_dataframe(qc::GravQC)::DataFrame
    s = qc.stations
    return DataFrame(
        ID = String[st.id for st in s],
        LAT = Float64[st.lat for st in s],
        LONG = Float64[st.lon for st in s],
        UTM10x_NAD83 = Float64[st.x for st in s],
        UTM10y_NAD83 = Float64[st.y for st in s],
        ELEV_M = Float64[st.elev for st in s],
        CBA = Float64[st.cba for st in s],
        ISO = Float64[st.iso for st in s],
        OG = Float64[st.og for st in s],
        FTC = Float64[st.ftc for st in s],
        TTC = Float64[st.ttc for st in s],
        cba_resid_deg1 = qc.fit1.resid,
        cba_resid_deg2 = qc.fit2.resid,
        outlier_cba = qc.outlier_cba,
        outlier_iso = qc.outlier_iso,
        duplicate = qc.duplicate,
    )
end

function fmt_stats(st::DistStats, unit::AbstractString)::String
    st.n == 0 && return "n=0"
    return @sprintf("n=%d  min=%.4g  median=%.4g  max=%.4g  std=%.4g %s",
                    st.n, st.min, st.median, st.max, st.std, unit)
end

function write_qc_markdown(path::AbstractString, grav_path::AbstractString,
                           qc::GravQC, mt::Vector{MTSite})
    open(path, "w") do io
        println(io, "# Northwest Geysers — gravite QC")
        println(io)
        println(io, "Hazırlık / QC only. Ters çözüm, forward, U-Net yok.")
        @printf(io, "Üretim: %s\n\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM"))
        println(io, "Kaynak: `", relpath(grav_path, ROOT), "`")
        println(io, "Peacock et al. 2020, USGS data release DOI [10.5066/P94D21UL](https://doi.org/10.5066/P94D21UL).")
        println(io)
        println(io, "## 1. Parse")
        println(io)
        @printf(io, "- Data satırı okundu: **%d**\n", qc.n_data_rows)
        @printf(io, "- Parse OK (gerekli sayısal alanlar sonlu): **%d**\n", qc.n_parse_ok)
        @printf(io, "- Eksik/bozuk (NaN, non-numeric, required field): **%d**\n", qc.n_corrupt)
        @printf(io, "- Kullanılabilir (sonlu UTM x/y **ve** CBA) → clean CSV: **%d**\n", qc.n_usable)
        @printf(io, "- Dropped (eksik koordinat veya CBA): **%d**\n", qc.n_dropped)
        if !isempty(qc.dropped_ids)
            println(io, "- Dropped ID: ", join(qc.dropped_ids, ", "))
        end
        println(io)
        println(io, "Gerekli alanlar: `UTM10x_NAD83`, `UTM10y_NAD83`, `ELEV_M`, `CBA`, `ISO`.")
        println(io, "Ayrıca parse edilen: `OG`, `FTC`, `TTC`, `LAT`, `LONG`, `ID`.")
        println(io, "CRLF (`\\r\\n`) LF'ye çevrildi; `#` satırları yorum.")
        println(io)
        println(io, "## 2. Dağılımlar")
        println(io)
        println(io, "- **CBA:** ", fmt_stats(qc.cba, "mGal"))
        println(io, "- **ISO:** ", fmt_stats(qc.iso, "mGal"))
        println(io, "- **FTC:** ", fmt_stats(qc.ftc, "mGal"))
        println(io, "- **TTC:** ", fmt_stats(qc.ttc, "mGal"))
        println(io, "- **FTC+TTC:** ", fmt_stats(qc.ftc_plus_ttc, "mGal"))
        println(io)
        println(io, "## 3. FTC + TTC ve CBA")
        println(io)
        print(io, header_ftc_ttc_conclusion(qc.comments))
        println(io)
        println(io, "## 4. Aykırı değerler ve tekrarlar")
        println(io)
        println(io, "Aykırı tanımı: `|x − median(x)| > $(OUTLIER_NSTD) × std(x)`")
        println(io, "(ortalama değil median merkez; std örnek standart sapması).")
        println(io, "Aykırılar **silinmedi** — yalnızca `outlier_cba` / `outlier_iso` bayrağı.")
        @printf(io, "- CBA aykırı: **%d** / %d\n", qc.n_outlier_cba, qc.n_usable)
        @printf(io, "- ISO aykırı: **%d** / %d\n", qc.n_outlier_iso, qc.n_usable)
        if qc.n_outlier_cba > 0
            ids = String[qc.stations[i].id for i in eachindex(qc.stations) if qc.outlier_cba[i]]
            println(io, "- CBA aykırı ID: `", join(ids, "`, `"), "`")
        end
        if qc.n_outlier_iso > 0
            ids = String[qc.stations[i].id for i in eachindex(qc.stations) if qc.outlier_iso[i]]
            println(io, "- ISO aykırı ID: `", join(ids, "`, `"), "`")
        end
        println(io)
        @printf(io, "Tekrar (duplicate): aynı UTM x/y, `hypot(Δx,Δy) ≤ %.1f m`.\n", DUP_TOL_M)
        @printf(io, "- Duplicate olarak işaretlenen istasyon: **%d** (%d grup)\n",
                qc.n_duplicate, length(qc.dup_groups))
        if isempty(qc.dup_groups)
            println(io, "- Grup yok.")
        else
            for g in qc.dup_groups
                println(io, "- Grup: `", join(g, "`, `"), "`")
            end
        end
        println(io)
        print(io, datum_note_md())
        println(io)
        println(io, "## 5. Bölgesel–rezidüel ayrımı")
        println(io)
        println(io, "CBA'ya `(x, y)` üzerinde polinom yüzey. Sayısal kararlılık için")
        println(io, "koordinatlar **ortalanıp ölçeklendi**:")
        println(io)
        println(io, "```")
        println(io, "xs = (x − mean(x)) / s")
        println(io, "ys = (y − mean(y)) / s")
        println(io, "s  = hypot(std(x), std(y))   # metre")
        println(io, "```")
        @printf(io, "- mean(x) = %.3f m, mean(y) = %.3f m, s = %.3f m\n",
                qc.fit1.x_mean, qc.fit1.y_mean, qc.fit1.scale)
        println(io)
        println(io, "Derece 1 (düzlem): `CBA ≈ a + b xs + c ys`")
        c1 = qc.fit1.coef
        @printf(io, "- katsayılar (ölçekli): a=%.6g  b=%.6g  c=%.6g\n", c1[1], c1[2], c1[3])
        @printf(io, "- yüzey RMSE = **%.4g mGal**\n", qc.fit1.rmse)
        @printf(io, "- rezidüel: min=%.4g  median=%.4g  max=%.4g  std=%.4g mGal\n",
                minimum(qc.fit1.resid), median(qc.fit1.resid),
                maximum(qc.fit1.resid), std(qc.fit1.resid))
        println(io)
        println(io, "Derece 2 (kuadratik): `CBA ≈ a + b xs + c ys + d xs² + e xs ys + f ys²`")
        c2 = qc.fit2.coef
        @printf(io, "- katsayılar (ölçekli): a=%.6g b=%.6g c=%.6g d=%.6g e=%.6g f=%.6g\n",
                c2[1], c2[2], c2[3], c2[4], c2[5], c2[6])
        @printf(io, "- yüzey RMSE = **%.4g mGal**\n", qc.fit2.rmse)
        @printf(io, "- rezidüel: min=%.4g  median=%.4g  max=%.4g  std=%.4g mGal\n",
                minimum(qc.fit2.resid), median(qc.fit2.resid),
                maximum(qc.fit2.resid), std(qc.fit2.resid))
        println(io)
        println(io, "Hangi rezidüelin ileride kullanılacağı **seçilmedi**; ikisi de CSV'de.")
        println(io)
        println(io, "## 6. Pearson korelasyonu (rezidüel vs ISO)")
        println(io)
        println(io, "ISO, uzun dalga boylu izostatik (Airy–Heiskanen) etkiyi kısmen çıkarır.")
        @printf(io, "- raw CBA vs ISO: **r = %.4f**\n", qc.cor_cba_iso)
        @printf(io, "- cba_resid_deg1 vs ISO: **r = %.4f**\n", qc.cor_r1_iso)
        @printf(io, "- cba_resid_deg2 vs ISO: **r = %.4f**\n", qc.cor_r2_iso)
        println(io)
        println(io, "## 7. MT istasyonları (harita)")
        println(io)
        @printf(io, "- EDI dosyası: **%d** (`data/geysers/mt/edi/*.edi`)\n", length(mt))
        println(io, "- Koordinat: EDI `>HEAD`/`DEFINEMEAS` lat/lon → UTM Zone 10N")
        println(io, "  WGS84 (EPSG:32610). Gravite UTM **NAD83** Zone 10N.")
        println(io, "  WGS84–NAD83 farkı bu bölgede ~1 m; overlay için yeterli,")
        println(io, "  datum kayması uygulanmadı.")
        println(io)
        println(io, "## Artefaktlar")
        println(io)
        println(io, "- `data/geysers/gravity/gravity_clean.csv`")
        println(io, "- `results/geysers_gravity_qc.md` (bu dosya)")
        println(io, "- `results/geysers_gravity_map.png`")
        println(io, "- `results/DATA_NOTES.md` — dikey datum notu")
    end
    return path
end

function plot_gravity_maps(png_path::AbstractString, qc::GravQC, mt::Vector{MTSite})
    x = Float64[s.x for s in qc.stations]
    y = Float64[s.y for s in qc.stations]
    cba = Float64[s.cba for s in qc.stations]
    r1 = qc.fit1.resid
    r2 = qc.fit2.resid
    mx = Float64[s.east for s in mt]
    my = Float64[s.north for s in mt]

    CM = MTGeophysics.CairoMakie
    CM.activate!()
    fig = CM.Figure(size = (1680, 620), fontsize = 13)

    function panel!(col, z, title, clabel)
        lo, hi = extrema(z)
        # Symmetric diverging scale around 0 when the field straddles 0.
        if lo < 0 < hi
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        end
        ax = CM.Axis(fig[1, col];
                     xlabel = "Easting (m, UTM Zone 10N NAD83)",
                     ylabel = col == 1 ? "Northing (m, UTM Zone 10N NAD83)" : "",
                     title = title,
                     aspect = CM.DataAspect(),
                     xticklabelrotation = π / 6)
        sc = CM.scatter!(ax, x, y; color = z, colormap = :RdBu, colorrange = (lo, hi),
                         markersize = 7, strokewidth = 0.2, strokecolor = (:gray, 0.4))
        CM.scatter!(ax, mx, my; marker = :utriangle, markersize = 11,
                    color = :black, strokewidth = 0.8, strokecolor = :white,
                    label = "MT")
        CM.Colorbar(fig[2, col], sc; label = clabel, vertical = false,
                    flipaxis = false, height = 12)
        return ax
    end

    panel!(1, cba, "CBA (Complete Bouguer)", "CBA (mGal)")
    panel!(2, r1, "CBA residual, degree 1", "resid deg1 (mGal)")
    panel!(3, r2, "CBA residual, degree 2", "resid deg2 (mGal)")
    CM.Label(fig[0, 1:3],
             "Northwest Geysers gravity — CBA and planar/quadratic residuals  |  " *
             @sprintf("n_grav = %d   n_MT = %d   black △ = EDI stations",
                      length(x), length(mt));
             fontsize = 16, padding = (0, 0, 4, 0), tellheight = true)
    CM.rowgap!(fig.layout, 8)
    mkpath(dirname(abspath(png_path)))
    CM.save(png_path, fig)
    return png_path
end

function print_summary(qc::GravQC, csv_path::AbstractString,
                       md_path::AbstractString, png_path::AbstractString)
    println("═"^72)
    println("Geysers gravity prep  (QC only, no inversion)")
    @printf("  rows read:     %d\n", qc.n_data_rows)
    @printf("  parse OK:      %d\n", qc.n_parse_ok)
    @printf("  corrupt/miss:  %d\n", qc.n_corrupt)
    @printf("  usable (CSV):  %d   dropped: %d\n", qc.n_usable, qc.n_dropped)
    @printf("  CBA  min/med/max/std = %.3f / %.3f / %.3f / %.3f mGal\n",
            qc.cba.min, qc.cba.median, qc.cba.max, qc.cba.std)
    @printf("  ISO  min/med/max/std = %.3f / %.3f / %.3f / %.3f mGal\n",
            qc.iso.min, qc.iso.median, qc.iso.max, qc.iso.std)
    @printf("  FTC+TTC already in CBA (Complete Bouguer; see QC md)\n")
    @printf("  outliers CBA/ISO: %d / %d   duplicates: %d (tol=%.1f m)\n",
            qc.n_outlier_cba, qc.n_outlier_iso, qc.n_duplicate, DUP_TOL_M)
    @printf("  RMSE deg1/deg2: %.4g / %.4g mGal\n", qc.fit1.rmse, qc.fit2.rmse)
    @printf("  Pearson  CBA~ISO=%.3f  r1~ISO=%.3f  r2~ISO=%.3f\n",
            qc.cor_cba_iso, qc.cor_r1_iso, qc.cor_r2_iso)
    println("  datum: ELEV_M is NGVD29; NAVD88 = NGVD29 + ~0.93 m (NOT applied)")
    println("  wrote ", relpath(csv_path, ROOT))
    println("  wrote ", relpath(md_path, ROOT))
    println("  wrote ", relpath(png_path, ROOT))
    println("═"^72)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function parse_cli(args::Vector{String})
    grav = DEFAULT_GRAV_PATH
    edi = DEFAULT_EDI_DIR
    out = DEFAULT_OUT_DIR
    csv = DEFAULT_CLEAN_CSV
    length(args) >= 1 && !isempty(args[1]) && (grav = abspath(args[1]))
    length(args) >= 2 && !isempty(args[2]) && (edi = abspath(args[2]))
    length(args) >= 3 && !isempty(args[3]) && (out = abspath(args[3]))
    length(args) >= 4 && !isempty(args[4]) && (csv = abspath(args[4]))
    return grav, edi, out, csv
end

function main(args::Vector{String} = ARGS)
    grav_path, edi_dir, out_dir, csv_path = parse_cli(args)
    mkpath(out_dir)
    mkpath(dirname(csv_path))

    println("[1/4] Parsing ", relpath(grav_path, ROOT))
    comments, _, all_st = parse_gravity_file(grav_path)
    qc = run_qc(comments, all_st)
    @printf("      %d data rows, %d usable, %d corrupt/missing required\n",
            qc.n_data_rows, qc.n_usable, qc.n_corrupt)

    println("[2/4] MT stations from ", relpath(edi_dir, ROOT))
    mt = load_mt_sites(edi_dir)
    @printf("      %d EDI sites\n", length(mt))

    md_path = joinpath(out_dir, "geysers_gravity_qc.md")
    png_path = joinpath(out_dir, "geysers_gravity_map.png")

    println("[3/4] Writing CSV + QC markdown")
    df = clean_dataframe(qc)
    CSV.write(csv_path, df)
    write_qc_markdown(md_path, grav_path, qc, mt)

    println("[4/4] Map ", relpath(png_path, ROOT))
    plot_gravity_maps(png_path, qc, mt)

    print_summary(qc, csv_path, md_path, png_path)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    using MTGeophysics   # CairoMakie re-export; only needed for the PNG
    main()
end
