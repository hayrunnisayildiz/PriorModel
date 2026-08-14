"""
    Voxelizer

3D multi-physics data-fusion pipeline.

Reads 1D borehole tables (GTK ` ^ `-delimited TXT mirrors of shapefiles) and
2D surface/airborne XYZ surveys, then interpolates them onto a `GridSpec`
voxel grid as a column-major `Array{Float32,4}` of shape `(X, Y, Z, C)`.

Interpolation:
- 3D borehole assays / petrophysics: inverse-distance weighting (IDW) splat
- 2D surface potentials: `NearestNeighbors.KDTree` + IDW in `(X, Y)`, then
  exponential vertical decay into depth
- Lithology: nearest-cell categorical codes (not IDW)

Missing voxels are `NaN32` so a later Lux prior can mask them.
"""
module Voxelizer

using CSV
using DataFrames
using LinearAlgebra
using NearestNeighbors
using StaticArrays
using YAML

# `GridSpec.jl` defines module `GridSpecs` (struct is `GridSpec`).
if !isdefined(@__MODULE__, :GridSpecs)
    include(joinpath(@__DIR__, "GridSpec.jl"))
end
using .GridSpecs

export GridSpec, load_gridspec, load_config, nx, ny, nz, nxyz, grid_size
export borehole_to_grid, surface_to_grid, build_fusion_tensor
export fusion_channel_names
export read_gtk_txt, read_gtk_xyz

const FILL::Float32 = NaN32
const P2 = SVector{2,Float32}

"""GTK shapefile stem → TXT mirror actually present under `database/`."""
const SHP_TO_TXT = Dict{String,String}(
    "collar.shp"   => "reiat.txt",
    "survey.shp"   => "kalte.txt",
    "rocktype.shp" => "kivit.txt",
    "petroph.shp"  => "petro.txt",
    "lithdesc.shp" => "kikuv.txt",
)

# ─────────────────────────────────────────────────────────────────────────────
# Collar / survey records (metres, degrees)
# ─────────────────────────────────────────────────────────────────────────────

"""
Collar location and collar attitude.

# Units
- `x, y, z`: metres (easting, northing, RL)
- `azimuth`: degrees, 0/360 = north, 90 = east
- `dip`: degrees from horizontal (GTK `Kaltevuus`; 90 = vertical down)
- `length`: metres (total hole length)
"""
struct Collar
    x::Float32
    y::Float32
    z::Float32
    azimuth::Float32
    dip::Float32
    length::Float32
end

"""
Downhole survey station.

# Units
- `depth`: metres along hole
- `dip`: degrees from horizontal
- `azimuth`: degrees from north
"""
struct SurveyStation
    depth::Float32
    dip::Float32
    azimuth::Float32
end

"""Pre-loaded collar + survey tables keyed by hole id (`Tunnus`)."""
struct BoreholeContext
    collars::Dict{String,Collar}
    surveys::Dict{String,Vector{SurveyStation}}
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar / path helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _to_f32(x) -> Float32

Parse a table/YAML scalar to `Float32`, returning `NaN32` for blanks / sentinels.
"""
function _to_f32(x)::Float32
    x === missing && return FILL
    if x isa Real
        v = Float32(x)
        return isfinite(v) ? v : FILL
    end
    s = strip(string(x))
    (isempty(s) || s == "-" || s == "*" || s == "NA" || s == "NaN") && return FILL
    return parse(Float32, s)
end

"""
    _abspath_data(root, rel) -> String

Join a YAML-relative path to the project root.
"""
function _abspath_data(root::AbstractString, rel::AbstractString)::String
    p = String(rel)
    return isabspath(p) ? p : joinpath(root, p)
end

"""
    resolve_table_path(root, yaml_path) -> String

Map a YAML shapefile path to the GTK TXT mirror (CSV-readable). Assay files
`d511p.shp` resolve to `511P.txt` in the same folder.
"""
function resolve_table_path(root::AbstractString, yaml_path::AbstractString)::String
    full = _abspath_data(root, yaml_path)
    if endswith(lowercase(yaml_path), ".shp")
        dir = dirname(full)
        base = lowercase(basename(String(yaml_path)))
        if haskey(SHP_TO_TXT, base)
            txt = joinpath(dir, SHP_TO_TXT[base])
            isfile(txt) && return txt
        else
            stem = replace(basename(String(yaml_path)), r"\.[Ss][Hh][Pp]$" => "")
            if length(stem) > 1 && lowercase(stem)[1] == 'd' && isdigit(stem[2])
                stem = stem[2:end]
            end
            txt = joinpath(dir, uppercase(stem) * ".txt")
            isfile(txt) && return txt
        end
    end
    isfile(full) && return full
    error("Data table not found for $(yaml_path) (resolved $(full))")
end

"""
    _latin1_string(path) -> String

Read a GTK TXT file as ISO-8859-1 (Finnish characters) and return UTF-8.
"""
function _latin1_string(path::AbstractString)::String
    bytes = read(path)
    io = IOBuffer()
    @inbounds for b in bytes
        write(io, Char(b))
    end
    return String(take!(io))
end

"""
    find_column(df, candidates) -> Union{String,Nothing}

Case-insensitive column lookup. Returns the actual DataFrame name or `nothing`.
"""
function find_column(df::DataFrame, candidates::AbstractVector{<:AbstractString})
    nms = string.(names(df))
    lower = lowercase.(strip.(nms))
    for c in candidates
        idx = findfirst(==(lowercase(strip(String(c)))), lower)
        idx !== nothing && return nms[idx]
    end
    return nothing
end

"""
    col_f32(df, col) -> Vector{Float32}

Extract a numeric column as `Float32`, with blanks → `NaN32`.
"""
function col_f32(df::DataFrame, col)::Vector{Float32}
    raw = df[!, col]
    n = length(raw)
    out = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        out[i] = _to_f32(raw[i])
    end
    return out
end

"""
    col_str(df, col) -> Vector{String}

Extract a string column, stripped of GTK padding.
"""
function col_str(df::DataFrame, col)::Vector{String}
    raw = df[!, col]
    n = length(raw)
    out = Vector{String}(undef, n)
    @inbounds for i in 1:n
        v = raw[i]
        out[i] = v === missing ? "" : strip(string(v))
    end
    return out
end

"""
    lookup_source(cfg, dotted) -> Any

Walk a dotted YAML path such as `"ground_geophysics.gravimetric"`.
"""
function lookup_source(cfg::AbstractDict, dotted::AbstractString)
    node = cfg
    for part in split(String(dotted), '.')
        node isa AbstractDict || error("Invalid source path `$(dotted)` at `$(part)`")
        haskey(node, String(part)) || error("Config is missing source `$(dotted)`")
        node = node[String(part)]
    end
    return node
end

"""
    active_grid_name(cfg) -> String

`gridspec.active` (`"deposit"` or `"regional"`).
"""
function active_grid_name(cfg::AbstractDict)::String
    gs = get(cfg, "gridspec", Dict{String,Any}())
    gs isa AbstractDict || return "deposit"
    return string(get(gs, "active", "deposit"))
end

"""
    tensor_channel_table(cfg) -> Vector{Dict{String,Any}}

`tensor_channels` list from YAML, each entry string-keyed.
"""
function tensor_channel_table(cfg::AbstractDict)::Vector{Dict{String,Any}}
    raw = get(cfg, "tensor_channels", nothing)
    raw isa AbstractVector || error("YAML is missing `tensor_channels:`")
    out = Dict{String,Any}[]
    for item in raw
        item isa AbstractDict || continue
        push!(out, Dict{String,Any}(string(k) => v for (k, v) in item))
    end
    return out
end

"""
    fusion_channel_names(config_path) -> Vector{String}

Channel names in `(X, Y, Z, C)` order, taken from `tensor_channels`.
"""
function fusion_channel_names(config_path::AbstractString)::Vector{String}
    cfg = load_config(config_path)
    return [string(ch["name"]) for ch in tensor_channel_table(cfg)]
end

# ─────────────────────────────────────────────────────────────────────────────
# GTK table / XYZ readers
# ─────────────────────────────────────────────────────────────────────────────

"""
    read_gtk_txt(path::String) -> DataFrame

Read a GTK Relia / assay / petrophysics TXT export.

Format (4 header rows, delimiter `" ^ "`):
1. column names, 2. types, 3. widths, 4. decimal places, 5+ data.

# Returns
`DataFrame` with stripped string columns (numeric parse is deferred).
"""
function read_gtk_txt(path::AbstractString)::DataFrame
    text = _latin1_string(path)
    df = CSV.read(
        IOBuffer(text),
        DataFrame;
        delim=" ^ ",
        header=1,
        skipto=5,
        stripwhitespace=true,
        missingstring=["", "-", "*"],
        types=String,
        ignoreemptyrows=true,
        silencewarnings=true,
        ntasks=1,
    )
    rename!(df, names(df) .=> strip.(string.(names(df))))
    return df
end

"""
    _norm_token(s) -> String

Normalise a header / config column name for fuzzy matching (`Re%` ≡ `Re_pct`).
"""
function _norm_token(s::AbstractString)::String
    t = lowercase(strip(String(s)))
    t = replace(t, r"[/%].*$" => "")
    t = replace(t, r"_pct$" => "")
    buf = IOBuffer()
    for c in t
        isvalid(c) || continue
        ('a' <= c <= 'z' || '0' <= c <= '9') && write(buf, c)
    end
    return String(take!(buf))
end

"""
    _xyz_header_names(line) -> Vector{String}

Parse a GTK `/ X/m Y/m Station GR/mCal …` comment into column names.
"""
function _xyz_header_names(line::AbstractString)::Vector{String}
    s = strip(String(line))
    startswith(s, "/") && (s = strip(s[2:end]))
    tokens = split(s)
    names = String[]
    for t in tokens
        name = split(t, '/'; limit=2)[1]
        name = replace(name, "%" => "")
        push!(names, name)
    end
    return names
end

"""
    _is_xyz_data_line(line) -> Bool

`true` for numeric GTK XYZ records (rejects `/` comments and `Line …` rows).
"""
function _is_xyz_data_line(line::AbstractString)::Bool
    s = strip(line)
    isempty(s) && return false
    c = s[1]
    return c == '+' || c == '-' || c == '.' || isdigit(c)
end

"""
    read_gtk_xyz(path::String) -> NamedTuple

Read a GTK ground or aerogeophysics XYZ file.

# Returns
`(; x, y, columns, names)` where
- `x, y::Vector{Float32}` — easting / northing (m)
- `columns::Dict{String,Vector{Float32}}` — remaining fields (mGal, nT, …)
- `names::Vector{String}` — column names in file order

Sentinel `*` becomes `NaN32`.
"""
function read_gtk_xyz(path::AbstractString)
    header_names = String[]
    xs = Float32[]
    ys = Float32[]
    extra = Vector{Float32}[]
    n_extra = 0
    text = _latin1_string(path)
    for raw in eachline(IOBuffer(text))
        line = strip(raw)
        if startswith(line, "/")
            cand = _xyz_header_names(line)
            if length(cand) >= 2 && _norm_token(cand[1]) == "x" && _norm_token(cand[2]) == "y"
                header_names = cand
            end
            continue
        end
        _is_xyz_data_line(line) || continue
        tokens = split(line)
        length(tokens) < 3 && continue
        x = _to_f32(tokens[1])
        y = _to_f32(tokens[2])
        (isnan(x) || isnan(y)) && continue
        if n_extra == 0
            n_extra = length(tokens) - 2
            extra = [Float32[] for _ in 1:n_extra]
            if isempty(header_names)
                header_names = vcat(["X", "Y"], ["C$(i+2)" for i in 1:n_extra])
            end
        end
        nval = min(n_extra, length(tokens) - 2)
        nval < n_extra && continue
        push!(xs, x)
        push!(ys, y)
        @inbounds for i in 1:n_extra
            push!(extra[i], _to_f32(tokens[i + 2]))
        end
    end
    cols = Dict{String,Vector{Float32}}()
    n_hdr = length(header_names)
    for i in 1:n_extra
        name = (i + 2) <= n_hdr ? header_names[i + 2] : "C$(i+2)"
        cols[name] = extra[i]
    end
    return (x=xs, y=ys, columns=cols, names=header_names)
end

"""
    _match_xyz_column(columns, candidates) -> Union{Vector{Float32},Nothing}

Pick the XYZ field whose normalised name matches any candidate.
Skips the `Station` column when the match would be ambiguous.
"""
function _match_xyz_column(columns::Dict{String,Vector{Float32}},
                           candidates::AbstractVector{<:AbstractString})
    keys_ = collect(keys(columns))
    norms = _norm_token.(keys_)
    for cand in candidates
        nc = _norm_token(cand)
        isempty(nc) && continue
        for (k, n) in zip(keys_, norms)
            n == nc && return columns[k]
        end
    end
    for cand in candidates
        nc = _norm_token(cand)
        isempty(nc) && continue
        for (k, n) in zip(keys_, norms)
            (startswith(n, nc) || startswith(nc, n)) || continue
            n == "station" && continue
            return columns[k]
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Desurvey (metres along hole → easting / northing / RL)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _lerp_az(a0, a1, t) -> Float32

Linear interpolate azimuth (deg) across the 0/360 wrap.
"""
function _lerp_az(a0::Float32, a1::Float32, t::Float32)::Float32
    Δ = a1 - a0
    Δ = mod(Δ + 180.0f0, 360.0f0) - 180.0f0
    return mod(a0 + t * Δ, 360.0f0)
end

"""
    desurvey_md(collar, stations, md) -> (x, y, z)

Convert measured depth `md` (m along hole) to KKJ coordinates (m) with the
balanced-tangential method.

GTK `Kaltevuus` is inclination from horizontal (90° = vertical down).
"""
function desurvey_md(collar::Collar, stations::Vector{SurveyStation},
                     md::Float32)::Tuple{Float32,Float32,Float32}
    x = collar.x
    y = collar.y
    z = collar.z
    md <= 0.0f0 && return (x, y, z)

    prev_d = 0.0f0
    prev_az = collar.azimuth
    prev_dip = collar.dip

    @inbounds for st in stations
        if st.depth <= 0.0f0
            prev_az = st.azimuth
            prev_dip = st.dip
            continue
        end
        if md <= st.depth
            Δ = md - prev_d
            span = st.depth - prev_d
            t = span > 0.0f0 ? Δ / span : 0.0f0
            az = _lerp_az(prev_az, st.azimuth, t)
            dip = prev_dip + t * (st.dip - prev_dip)
            x += Δ * cosd(dip) * sind(az)
            y += Δ * cosd(dip) * cosd(az)
            z -= Δ * sind(dip)
            return (x, y, z)
        else
            Δ = st.depth - prev_d
            az = _lerp_az(prev_az, st.azimuth, 0.5f0)
            dip = 0.5f0 * (prev_dip + st.dip)
            x += Δ * cosd(dip) * sind(az)
            y += Δ * cosd(dip) * cosd(az)
            z -= Δ * sind(dip)
            prev_d = st.depth
            prev_az = st.azimuth
            prev_dip = st.dip
        end
    end
    Δ = md - prev_d
    x += Δ * cosd(prev_dip) * sind(prev_az)
    y += Δ * cosd(prev_dip) * cosd(prev_az)
    z -= Δ * sind(prev_dip)
    return (x, y, z)
end

"""
    load_borehole_context(root, cfg) -> BoreholeContext

Read collar (`reiat.txt`) and survey (`kalte.txt`) and index them by hole id.

GTK TXT swaps axis names: `Kkj_y` = easting, `Kkj_X` = northing (metres).
"""
function load_borehole_context(root::AbstractString, cfg::AbstractDict)::BoreholeContext
    bh = cfg["borehole"]
    collar_path = resolve_table_path(root, string(bh["collar"]["path"]))
    survey_path = resolve_table_path(root, string(bh["survey"]["path"]))
    cdf = read_gtk_txt(collar_path)
    sdf = read_gtk_txt(survey_path)

    idc = find_column(cdf, ["Tunnus", "HOLE_ID"])
    xc  = find_column(cdf, ["Kkj_y", "Kkj_Y", "KKJ_EAST", "ReiKkj_y"])
    yc  = find_column(cdf, ["Kkj_X", "Kkj_x", "KKJ_NORTH", "ReiKkj_X"])
    zc  = find_column(cdf, ["Z", "Rei_Z"])
    azc = find_column(cdf, ["Suunta", "STARTAZIM", "AZIMUTH"])
    dpc = find_column(cdf, ["Kaltevuus", "STARTDIP", "DIP"])
    lc  = find_column(cdf, ["Pituus", "LENGTH"])
    (idc === nothing || xc === nothing || yc === nothing || zc === nothing) &&
        error("Collar table $(collar_path) is missing Tunnus/Kkj_y/Kkj_X/Z")

    ids = col_str(cdf, idc)
    xs = col_f32(cdf, xc)
    ys = col_f32(cdf, yc)
    zs = col_f32(cdf, zc)
    azs = azc === nothing ? fill(0.0f0, length(ids)) : col_f32(cdf, azc)
    dps = dpc === nothing ? fill(90.0f0, length(ids)) : col_f32(cdf, dpc)
    ls  = lc === nothing ? fill(FILL, length(ids)) : col_f32(cdf, lc)

    collars = Dict{String,Collar}()
    @inbounds for i in eachindex(ids)
        hid = ids[i]
        (isempty(hid) || isnan(xs[i]) || isnan(ys[i]) || isnan(zs[i])) && continue
        az = isnan(azs[i]) ? 0.0f0 : azs[i]
        dp = isnan(dps[i]) ? 90.0f0 : dps[i]
        collars[hid] = Collar(xs[i], ys[i], zs[i], az, dp, ls[i])
    end

    sid = find_column(sdf, ["Tunnus", "HOLE_ID"])
    sdep = find_column(sdf, ["Syvyys", "DEPTH"])
    sdip = find_column(sdf, ["Kaltevuus", "DIP"])
    saz  = find_column(sdf, ["Suunta", "AZIMUTH"])
    (sid === nothing || sdep === nothing || sdip === nothing || saz === nothing) &&
        error("Survey table $(survey_path) is missing Tunnus/Syvyys/Kaltevuus/Suunta")

    sids = col_str(sdf, sid)
    sdeps = col_f32(sdf, sdep)
    sdips = col_f32(sdf, sdip)
    sazs = col_f32(sdf, saz)
    surveys = Dict{String,Vector{SurveyStation}}()
    @inbounds for i in eachindex(sids)
        hid = sids[i]
        (isempty(hid) || isnan(sdeps[i])) && continue
        dip = isnan(sdips[i]) ? 90.0f0 : sdips[i]
        az = isnan(sazs[i]) ? 0.0f0 : sazs[i]
        push!(get!(Vector{SurveyStation}, surveys, hid), SurveyStation(sdeps[i], dip, az))
    end
    for v in values(surveys)
        sort!(v; by=s -> s.depth)
    end
    @info "Loaded collars/surveys" n_holes=length(collars) n_survey_holes=length(surveys)
    return BoreholeContext(collars, surveys)
end

"""
    _stations(ctx, hole_id) -> Vector{SurveyStation}

Survey stations for a hole, or an empty vector (collar attitude is then used).
"""
function _stations(ctx::BoreholeContext, hole_id::AbstractString)::Vector{SurveyStation}
    return get(ctx.surveys, String(hole_id), SurveyStation[])
end

# ─────────────────────────────────────────────────────────────────────────────
# IDW / KDTree interpolation onto GridSpec
# ─────────────────────────────────────────────────────────────────────────────

"""
    idw_splat_3d!(out, grid, px, py, pz, pv; power, max_radius)

Accumulate IDW (`w = 1 / d^power`) from scattered 3D samples into `out`
of shape `(nx, ny, nz)`. Cells with no sample inside `max_radius` stay `NaN32`.

# Units
`px, py, pz, max_radius`: metres; `pv`: channel physical unit (ppm, kg/m³, …).
"""
function idw_splat_3d!(out::AbstractArray{Float32,3},
                       grid::GridSpec,
                       px::Vector{Float32},
                       py::Vector{Float32},
                       pz::Vector{Float32},
                       pv::Vector{Float32};
                       power::Float32=2.0f0,
                       max_radius::Float32=150.0f0)
    nxx, nyy, nzz = size(out)
    num = zeros(Float32, nxx, nyy, nzz)
    den = zeros(Float32, nxx, nyy, nzz)
    ir = max(1, Int(ceil(max_radius / grid.dx)))
    jr = max(1, Int(ceil(max_radius / grid.dy)))
    kr = max(1, Int(ceil(max_radius / grid.dz)))
    r2 = max_radius * max_radius
    half_p = power * 0.5f0
    @inbounds for p in eachindex(pv)
        v = pv[p]
        (isnan(v) || isnan(px[p]) || isnan(py[p]) || isnan(pz[p])) && continue
        i0, j0, k0 = coord_to_index(grid, px[p], py[p], pz[p])
        i1 = max(1, i0 - ir); i2 = min(nxx, i0 + ir)
        j1 = max(1, j0 - jr); j2 = min(nyy, j0 + jr)
        k1 = max(1, k0 - kr); k2 = min(nzz, k0 + kr)
        (i2 < 1 || i1 > nxx || j2 < 1 || j1 > nyy || k2 < 1 || k1 > nzz) && continue
        for k in k1:k2, j in j1:j2, i in i1:i2
            cx, cy, cz = index_to_coord(grid, i, j, k)
            dx_ = px[p] - cx
            dy_ = py[p] - cy
            dz_ = pz[p] - cz
            d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_
            d2 > r2 && continue
            w = 1.0f0 / (d2^half_p + 1.0f-12)
            num[i, j, k] += w * v
            den[i, j, k] += w
        end
    end
    @inbounds for k in 1:nzz, j in 1:nyy, i in 1:nxx
        d = den[i, j, k]
        out[i, j, k] = d > 0.0f0 ? num[i, j, k] / d : FILL
    end
    return out
end

"""
    nearest_fill_3d!(out, grid, px, py, pz, pv)

Write each sample into its containing voxel (categorical lithology). Last
write wins. Off-grid samples are skipped.
"""
function nearest_fill_3d!(out::AbstractArray{Float32,3},
                          grid::GridSpec,
                          px::Vector{Float32},
                          py::Vector{Float32},
                          pz::Vector{Float32},
                          pv::Vector{Float32})
    nxx, nyy, nzz = size(out)
    fill!(out, FILL)
    @inbounds for p in eachindex(pv)
        (isnan(pv[p]) || isnan(px[p]) || isnan(py[p]) || isnan(pz[p])) && continue
        i, j, k = coord_to_index(grid, px[p], py[p], pz[p])
        (1 <= i <= nxx && 1 <= j <= nyy && 1 <= k <= nzz) || continue
        out[i, j, k] = pv[p]
    end
    return out
end

"""
    idw_kdtree_2d(xc, yc, px, py, pv; k, power, max_radius) -> Matrix{Float32}

Interpolate scattered 2D samples onto the cell-centre `(X, Y)` mesh with a
`KDTree` and k-neighbour IDW. Shape `(nx, ny)`.

# Units
All coordinates and `max_radius` in metres; `pv` in the survey unit (mGal, nT, …).
"""
function idw_kdtree_2d(xc::Vector{Float32},
                       yc::Vector{Float32},
                       px::Vector{Float32},
                       py::Vector{Float32},
                       pv::Vector{Float32};
                       k::Int=8,
                       power::Float32=2.0f0,
                       max_radius::Float32=150.0f0)::Matrix{Float32}
    nxx = length(xc)
    nyy = length(yc)
    out = fill(FILL, nxx, nyy)
    keep = .!isnan.(pv) .& .!isnan.(px) .& .!isnan.(py)
    count(keep) == 0 && return out
    pts = P2[P2(px[i], py[i]) for i in eachindex(px) if keep[i]]
    vals = Float32[pv[i] for i in eachindex(pv) if keep[i]]
    tree = KDTree(pts)
    k_use = min(k, length(pts))
    k_use < 1 && return out
    @inbounds for j in 1:nyy
        y = yc[j]
        for i in 1:nxx
            idxs, dists = knn(tree, P2(xc[i], y), k_use, true)
            d0 = Float32(dists[1])
            d0 > max_radius && continue
            if d0 <= 1.0f-3
                out[i, j] = vals[idxs[1]]
                continue
            end
            num = 0.0f0
            den = 0.0f0
            for t in 1:k_use
                d = Float32(dists[t])
                d > max_radius && break
                w = 1.0f0 / (d^power + 1.0f-12)
                num += w * vals[idxs[t]]
                den += w
            end
            den > 0.0f0 && (out[i, j] = num / den)
        end
    end
    return out
end

"""
    apply_vertical_decay!(volume, grid, xy, decay_length)

Copy a 2D `(nx, ny)` field into every Z slice with
`A(z) = A_xy * exp(-(zmax - z) / λ)`.

# Units
`decay_length` (`λ`): metres; `xy` / `volume`: survey unit (mGal, nT, mV/V, …).
"""
function apply_vertical_decay!(volume::AbstractArray{Float32,3},
                               grid::GridSpec,
                               xy::AbstractMatrix{Float32},
                               decay_length::Float32)
    nxx, nyy, nzz = size(volume)
    zc = z_centers(grid)
    λ = max(decay_length, 1.0f0)
    @inbounds for k in 1:nzz
        depth = grid.zmax - zc[k]
        att = Float32(exp(-depth / λ))
        @views slice = volume[:, :, k]
        for j in 1:nyy, i in 1:nxx
            v = xy[i, j]
            slice[i, j] = isnan(v) ? FILL : v * att
        end
    end
    return volume
end

# ─────────────────────────────────────────────────────────────────────────────
# Borehole → grid
# ─────────────────────────────────────────────────────────────────────────────

"""
    _push_desurveyed!(px, py, pz, pv, ctx, hole, md, value)

Append one desurveyed sample if the hole exists and `value` is finite.
"""
function _push_desurveyed!(px::Vector{Float32}, py::Vector{Float32},
                           pz::Vector{Float32}, pv::Vector{Float32},
                           ctx::BoreholeContext, hole::AbstractString,
                           md::Float32, value::Float32)
    (isnan(value) || isnan(md) || isempty(hole)) && return
    haskey(ctx.collars, hole) || return
    x, y, z = desurvey_md(ctx.collars[hole], _stations(ctx, hole), md)
    push!(px, x); push!(py, y); push!(pz, z); push!(pv, value)
    return
end

"""
    _load_assay_points(root, cfg, ctx, element) -> (px, py, pz, pv)

Read the primary ICP-OES assay table (`511P`) and desurvey interval midpoints.

# Units
`pv`: ppm (GTK default); coordinates in metres.
"""
function _load_assay_points(root::AbstractString, cfg::AbstractDict,
                            ctx::BoreholeContext, element::AbstractString)
    assay = cfg["borehole"]["assay"]
    base = string(assay["base_path"])
    methods = assay["methods"]
    file = "d511p.shp"
    if methods isa AbstractVector
        for m in methods
            m isa AbstractDict || continue
            code = string(get(m, "code", ""))
            uppercase(code) == "511P" && (file = string(get(m, "file", file)))
        end
    end
    path = resolve_table_path(root, joinpath(base, file))
    df = read_gtk_txt(path)
    idc = find_column(df, ["Tunnus", "HOLE_ID"])
    fromc = find_column(df, ["Ylasyvyys", "FROM"])
    toc = find_column(df, ["Alasyvyys", "TO"])
    elc = find_column(df, [String(element), uppercase(String(element)),
                           lowercase(String(element)), titlecase(String(element))])
    (idc === nothing || fromc === nothing || toc === nothing || elc === nothing) &&
        error("Assay $(path) is missing hole/from/to/$(element)")

    ids = col_str(df, idc)
    froms = col_f32(df, fromc)
    tos = col_f32(df, toc)
    vals = col_f32(df, elc)
    px = Float32[]; py = Float32[]; pz = Float32[]; pv = Float32[]
    @inbounds for i in eachindex(ids)
        md = 0.5f0 * (froms[i] + tos[i])
        _push_desurveyed!(px, py, pz, pv, ctx, ids[i], md, vals[i])
    end
    @info "Assay points" element=element n=length(pv) file=basename(path)
    return px, py, pz, pv
end

"""
    _load_density_points(root, cfg, ctx) -> (px, py, pz, pv)

Read core petrophysics (`petro.txt`). Prefer dry density `DSR_D`, fall back
to `PTR_D`.

# Units
`pv`: kg/m³; `md`: metres along hole.
"""
function _load_density_points(root::AbstractString, cfg::AbstractDict,
                              ctx::BoreholeContext)
    petro_yaml = cfg["borehole"]["petrophysics"]["path"]
    path = resolve_table_path(root, string(petro_yaml))
    df = read_gtk_txt(path)
    idc = find_column(df, ["Tunnus", "HOLE_ID"])
    depc = find_column(df, ["Syvyys", "DEPTH"])
    dsrc = find_column(df, ["DSR_D"])
    ptrc = find_column(df, ["PTR_D"])
    (idc === nothing || depc === nothing) && error("Petrophysics missing Tunnus/Syvyys")
    (dsrc === nothing && ptrc === nothing) && error("Petrophysics missing DSR_D/PTR_D")

    ids = col_str(df, idc)
    deps = col_f32(df, depc)
    dsr = dsrc === nothing ? fill(FILL, length(ids)) : col_f32(df, dsrc)
    ptr = ptrc === nothing ? fill(FILL, length(ids)) : col_f32(df, ptrc)
    px = Float32[]; py = Float32[]; pz = Float32[]; pv = Float32[]
    @inbounds for i in eachindex(ids)
        v = isnan(dsr[i]) ? ptr[i] : dsr[i]
        _push_desurveyed!(px, py, pz, pv, ctx, ids[i], deps[i], v)
    end
    @info "Density points" n=length(pv) unit="kg/m3"
    return px, py, pz, pv
end

"""
    _load_lithology_points(root, cfg, ctx, dz) -> (px, py, pz, pv)

Sample lithology intervals (`kivit.txt`, `Kivilaji`) every `dz/2` metres along
the hole. Codes are `1, 2, …` over sorted unique rock names (`NaN32` = unset).

# Units
Depths in metres; `pv` is a dimensionless class index.
"""
function _load_lithology_points(root::AbstractString, cfg::AbstractDict,
                                ctx::BoreholeContext, dz::Float32)
    lith_yaml = cfg["borehole"]["lithology"]["path"]
    path = resolve_table_path(root, string(lith_yaml))
    df = read_gtk_txt(path)
    idc = find_column(df, ["Tunnus", "HOLE_ID"])
    fromc = find_column(df, ["Ylasyvyys", "FROM"])
    toc = find_column(df, ["Alasyvyys", "TO"])
    roc = find_column(df, ["Kivilaji", "ROCKTYPE"])
    (idc === nothing || fromc === nothing || toc === nothing || roc === nothing) &&
        error("Lithology table missing Tunnus/Ylasyvyys/Alasyvyys/Kivilaji")

    ids = col_str(df, idc)
    froms = col_f32(df, fromc)
    tos = col_f32(df, toc)
    rocks = col_str(df, roc)
    uniq = sort(unique(filter(!isempty, rocks)))
    code = Dict{String,Float32}(r => Float32(i) for (i, r) in enumerate(uniq))
    step = max(dz * 0.5f0, 1.0f0)
    px = Float32[]; py = Float32[]; pz = Float32[]; pv = Float32[]
    @inbounds for i in eachindex(ids)
        r = rocks[i]
        (isempty(r) || !haskey(code, r)) && continue
        a = froms[i]; b = tos[i]
        (isnan(a) || isnan(b) || b < a) && continue
        md = a
        while md <= b + 1.0f-3
            _push_desurveyed!(px, py, pz, pv, ctx, ids[i], md, code[r])
            md += step
        end
    end
    @info "Lithology points" n=length(pv) n_classes=length(uniq)
    return px, py, pz, pv
end

"""
    _fill_borehole_channel!(slice, grid, root, cfg, ctx, ch; kwargs...)

Dispatch one `tensor_channels` borehole entry onto a `(nx, ny, nz)` slice.
"""
function _fill_borehole_channel!(slice::AbstractArray{Float32,3},
                                 grid::GridSpec,
                                 root::AbstractString,
                                 cfg::AbstractDict,
                                 ctx::BoreholeContext,
                                 ch::AbstractDict;
                                 power::Float32=2.0f0,
                                 max_radius::Float32=150.0f0)
    source = string(ch["source"])
    name = string(ch["name"])
    if source == "borehole.assay"
        element = string(get(ch, "element", startswith(lowercase(name), "ni") ? "NI" : "CU"))
        px, py, pz, pv = _load_assay_points(root, cfg, ctx, element)
        idw_splat_3d!(slice, grid, px, py, pz, pv; power=power, max_radius=max_radius)
    elseif source == "borehole.petrophysics"
        px, py, pz, pv = _load_density_points(root, cfg, ctx)
        idw_splat_3d!(slice, grid, px, py, pz, pv; power=power, max_radius=max_radius)
    elseif source == "borehole.lithology"
        px, py, pz, pv = _load_lithology_points(root, cfg, ctx, grid.dz)
        nearest_fill_3d!(slice, grid, px, py, pz, pv)
    else
        @warn "Unhandled borehole source" source name
        fill!(slice, FILL)
    end
    return slice
end

"""
    borehole_to_grid(grid, config_path; power, max_radius) -> Array{Float32,4}

Voxelize collar + assay + downhole/core tables onto `grid`.

# Arguments
- `grid`: [`GridSpec`](@ref)
- `config_path`: YAML path (`config/config.yaml`)
- `power`: IDW exponent (dimensionless, default 2)
- `max_radius`: IDW cutoff (m, default 150)

# Returns
`Array{Float32,4}` of shape `(nx, ny, nz, C_bh)` where `C_bh` are the
`tensor_channels` whose `source` starts with `borehole.` (column-major
`(X, Y, Z, C)`). Units follow each channel (`ppm`, `kg/m³`, class index).
"""
function borehole_to_grid(grid::GridSpec, config_path::AbstractString;
                          power::Float32=2.0f0,
                          max_radius::Float32=150.0f0)::Array{Float32,4}
    cfg = load_config(config_path)
    root = project_root(resolve_config_path(config_path))
    channels = [ch for ch in tensor_channel_table(cfg)
                if startswith(string(ch["source"]), "borehole.")]
    nxx, nyy, nzz = nxyz(grid)
    nc = length(channels)
    tensor = fill(FILL, nxx, nyy, nzz, nc)
    ctx = load_borehole_context(root, cfg)
    for (c, ch) in enumerate(channels)
        @info "Borehole channel" index=c name=ch["name"]
        @views sl = tensor[:, :, :, c]
        _fill_borehole_channel!(sl, grid, root, cfg, ctx, ch;
                                power=power, max_radius=max_radius)
    end
    return tensor
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface → grid
# ─────────────────────────────────────────────────────────────────────────────

"""
    _iter_xyz_files(spec, active_block) -> Vector{Tuple{String,String}}

`(filename, scope)` pairs. Regional-only surveys are skipped on the deposit grid.
"""
function _iter_xyz_files(spec::AbstractDict, active_block::AbstractString)
    files = get(spec, "files", nothing)
    files === nothing && return Tuple{String,String}[]
    items = files isa AbstractVector ? files : Any[files]
    out = Tuple{String,String}[]
    for f in items
        if f isa AbstractString
            push!(out, (String(f), ""))
        elseif f isa AbstractDict
            fd = Dict{String,Any}(string(k) => v for (k, v) in f)
            fname = string(get(fd, "file", ""))
            scope = string(get(fd, "scope", ""))
            isempty(fname) && continue
            if scope == "regional" && String(active_block) == "deposit"
                continue
            end
            push!(out, (fname, scope))
        end
    end
    return out
end

"""
    _channel_value_candidates(spec, channel_name) -> Vector{String}

Header names to try when extracting one tensor channel from an XYZ file.
"""
function _channel_value_candidates(spec::AbstractDict, channel_name::AbstractString)::Vector{String}
    cand = String[]
    if haskey(spec, "channel_name") && string(spec["channel_name"]) == String(channel_name)
        haskey(spec, "value_column") && push!(cand, string(spec["value_column"]))
    end
    if haskey(spec, "channel_names") && haskey(spec, "value_columns")
        cns = spec["channel_names"]
        vcs = spec["value_columns"]
        if cns isa AbstractVector && vcs isa AbstractVector
            for (cn, vc) in zip(cns, vcs)
                if string(cn) == String(channel_name)
                    push!(cand, string(vc))
                end
            end
        end
    end
    haskey(spec, "value_column") && push!(cand, string(spec["value_column"]))
    extra = Dict(
        "gravity"          => ["GR", "gravity"],
        "magnetic_tmi"     => ["MG", "MGL"],
        "aero_magnetic"    => ["MGL", "MG"],
        "ip_chargeability" => ["IP"],
        "self_potential"   => ["SP"],
        "vlf_resistivity"  => ["ov", "OV"],
        "slingram_real"    => ["Re", "Re_pct", "Re%"],
    )
    haskey(extra, String(channel_name)) && append!(cand, extra[String(channel_name)])
    return unique(cand)
end

"""
    _load_surface_xy(root, spec, channel_name, active_block) -> (px, py, pv)

Concatenate every in-scope XYZ file for one surface channel.

# Units
`px, py`: metres; `pv`: survey unit from YAML (`mGal`, `nT`, `mV/V`, …).
"""
function _load_surface_xy(root::AbstractString, spec::AbstractDict,
                          channel_name::AbstractString,
                          active_block::AbstractString)
    dir = _abspath_data(root, string(get(spec, "path", "")))
    files = _iter_xyz_files(spec, active_block)
    if isempty(files) && isdir(dir)
        files = [(basename(f), "") for f in readdir(dir; join=false)
                 if endswith(lowercase(f), ".xyz")]
    end
    cand = _channel_value_candidates(spec, channel_name)
    px = Float32[]; py = Float32[]; pv = Float32[]
    for (fname, _) in files
        fnlow = lowercase(fname)
        if String(channel_name) == "aero_magnetic" && occursin(r"mr94|rtp", fnlow)
            continue  # keep TMI (ml94) off the RTP grid
        end
        if String(channel_name) == "magnetic_tmi" && startswith(basename(fnlow), "j")
            continue  # Jalander relative nT ≠ proton TMI
        end
        fp = joinpath(dir, fname)
        isfile(fp) || (fp = joinpath(dir, uppercase(fname)))
        if !isfile(fp)
            @warn "XYZ file missing" file=fname dir=dir
            continue
        end
        xyz = read_gtk_xyz(fp)
        col = _match_xyz_column(xyz.columns, cand)
        if col === nothing
            # first non-Station extra column
            for (k, v) in xyz.columns
                _norm_token(k) == "station" && continue
                col = v
                break
            end
        end
        col === nothing && continue
        n = min(length(xyz.x), length(col))
        @inbounds for i in 1:n
            isnan(col[i]) && continue
            push!(px, xyz.x[i]); push!(py, xyz.y[i]); push!(pv, col[i])
        end
        @info "XYZ loaded" file=basename(fp) n=n channel=channel_name
    end
    return px, py, pv
end

"""
    _fill_surface_channel!(slice, grid, root, cfg, ch; kwargs...)

Interpolate one 2D survey onto `(nx, ny)` then decay it through `Z`.
"""
function _fill_surface_channel!(slice::AbstractArray{Float32,3},
                                grid::GridSpec,
                                root::AbstractString,
                                cfg::AbstractDict,
                                ch::AbstractDict;
                                k::Int=8,
                                power::Float32=2.0f0,
                                max_radius::Float32=150.0f0,
                                decay_length::Float32=250.0f0)
    spec = lookup_source(cfg, string(ch["source"]))
    spec isa AbstractDict || error("Source $(ch["source"]) is not a mapping")
    active = active_grid_name(cfg)
    px, py, pv = _load_surface_xy(root, spec, string(ch["name"]), active)
    xc = x_centers(grid)
    yc = y_centers(grid)
    xy = idw_kdtree_2d(xc, yc, px, py, pv; k=k, power=power, max_radius=max_radius)
    apply_vertical_decay!(slice, grid, xy, decay_length)
    return slice
end

"""
    surface_to_grid(grid, config_path; k, power, max_radius, decay_length) -> Array{Float32,4}

Map gravity / mag / IP / SP / VLF / slingram (and aero mag) onto the top of
`grid` and continue them downward with `exp(-depth / λ)`.

# Arguments
- `k`: KDTree neighbours (default 8)
- `power`: IDW exponent
- `max_radius`: planar IDW cutoff (m)
- `decay_length`: vertical skin length `λ` (m, default 250)

# Returns
`Array{Float32,4}` of shape `(nx, ny, nz, C_surf)` for `tensor_channels`
whose source is `ground_geophysics.*` or `aero_geophysics.*`.
Layout `(X, Y, Z, C)`, `Float32`.
"""
function surface_to_grid(grid::GridSpec, config_path::AbstractString;
                         k::Int=8,
                         power::Float32=2.0f0,
                         max_radius::Float32=150.0f0,
                         decay_length::Float32=250.0f0)::Array{Float32,4}
    cfg = load_config(config_path)
    root = project_root(resolve_config_path(config_path))
    channels = [ch for ch in tensor_channel_table(cfg)
                if startswith(string(ch["source"]), "ground_geophysics.") ||
                   startswith(string(ch["source"]), "aero_geophysics.")]
    nxx, nyy, nzz = nxyz(grid)
    nc = length(channels)
    tensor = fill(FILL, nxx, nyy, nzz, nc)
    for (c, ch) in enumerate(channels)
        @info "Surface channel" index=c name=ch["name"] unit=get(ch, "unit", "")
        @views sl = tensor[:, :, :, c]
        _fill_surface_channel!(sl, grid, root, cfg, ch;
                               k=k, power=power, max_radius=max_radius,
                               decay_length=decay_length)
    end
    return tensor
end

# ─────────────────────────────────────────────────────────────────────────────
# Full fusion tensor
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_fusion_tensor(grid, config_path; kwargs...) -> Array{Float32,4}

Fuse every `tensor_channels` stream into one multi-channel volume.

# Returns
`Array{Float32,4}` of shape `(nx, ny, nz, C)` in column-major `(X, Y, Z, C)`,
`Float32`. Channel `c` (1-based) matches `tensor_channels[c]`
(YAML `id` is 0-based). Empty cells are `NaN32`.

# Keyword arguments
- `power::Float32=2`: IDW exponent
- `max_radius::Float32=150`: cutoff (m)
- `k::Int=8`: KDTree neighbours for 2D surveys
- `decay_length::Float32=250`: vertical decay length (m)

See [`fusion_channel_names`](@ref) for the `C` axis labels.
"""
function build_fusion_tensor(grid::GridSpec, config_path::AbstractString;
                             power::Float32=2.0f0,
                             max_radius::Float32=150.0f0,
                             k::Int=8,
                             decay_length::Float32=250.0f0)::Array{Float32,4}
    cfg = load_config(config_path)
    root = project_root(resolve_config_path(config_path))
    channels = tensor_channel_table(cfg)
    nxx, nyy, nzz = nxyz(grid)
    nc = length(channels)
    tensor = fill(FILL, nxx, nyy, nzz, nc)
    @info "Fusion tensor" size=(nxx, nyy, nzz, nc) epsg=grid.epsg_code

    need_bh = any(ch -> startswith(string(ch["source"]), "borehole."), channels)
    ctx = need_bh ? load_borehole_context(root, cfg) : BoreholeContext(Dict{String,Collar}(),
                                                                      Dict{String,Vector{SurveyStation}}())

    for (c, ch) in enumerate(channels)
        source = string(ch["source"])
        name = string(ch["name"])
        @info "Filling channel" c=c-1 name source
        @views sl = tensor[:, :, :, c]
        if startswith(source, "borehole.")
            _fill_borehole_channel!(sl, grid, root, cfg, ctx, ch;
                                    power=power, max_radius=max_radius)
        elseif startswith(source, "ground_geophysics.") || startswith(source, "aero_geophysics.")
            _fill_surface_channel!(sl, grid, root, cfg, ch;
                                   k=k, power=power, max_radius=max_radius,
                                   decay_length=decay_length)
        else
            @warn "Unknown tensor source; leaving NaN" source name
        end
    end
    return tensor
end

"""
    build_fusion_tensor(config_path; block, kwargs...) -> Array{Float32,4}

Load `GridSpec` from YAML (`block` defaults to `gridspec.active`) then fuse.
"""
function build_fusion_tensor(config_path::AbstractString;
                             block::Union{Nothing,AbstractString}=nothing,
                             kwargs...)::Array{Float32,4}
    grid = load_gridspec(config_path; block=block)
    return build_fusion_tensor(grid, config_path; kwargs...)
end

end # module Voxelizer
