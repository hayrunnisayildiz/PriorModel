"""
    GridSpecs

Type-stable 3D voxel-grid specification for the Smart Prior fusion pipeline.

Coordinates are metres in KKJ / Finland Zone 3 (EPSG:2391). The `z` axis is
metres RL (reduced level), positive upward. Arrays that consume this spec
MUST use Julia column-major layout `(X, Y, Z, C)`.

The module is named `GridSpecs` so it does not clash with `struct GridSpec`.
"""
module GridSpecs

using YAML

export GridSpec
export load_config, load_gridspec, resolve_config_path, project_root
export nx, ny, nz, nxyz, grid_size
export x_centers, y_centers, z_centers
export in_bounds, coord_to_index, index_to_coord

# ─────────────────────────────────────────────────────────────────────────────
# Grid specification
# ─────────────────────────────────────────────────────────────────────────────

"""
    GridSpec

Regular 3D Cartesian voxel grid.

# Fields
- `xmin, xmax, ymin, ymax, zmin, zmax`: bounding box, metres
- `dx, dy, dz`: cell size, metres
- `epsg_code`: EPSG integer (2391 for KKJ Zone 3)

# Layout
Cell `(i, j, k)` (1-based) occupies
`[xmin+(i-1)dx, xmin+i·dx) × [ymin+(j-1)dy, ymin+j·dy) × [zmin+(k-1)dz, zmin+k·dz)`
with centre `(xmin+(i-0.5)dx, ymin+(j-0.5)dy, zmin+(k-0.5)dz)`.
Index `k = 1` is the deepest slice (`zmin`); `k = nz` is the top (`zmax`).
"""
struct GridSpec
    xmin::Float32
    xmax::Float32
    ymin::Float32
    ymax::Float32
    zmin::Float32
    zmax::Float32
    dx::Float32
    dy::Float32
    dz::Float32
    epsg_code::Int32
end

function Base.show(io::IO, g::GridSpec)
    nxx, nyy, nzz = nxyz(g)
    print(io, "GridSpec(EPSG:", g.epsg_code,
          " X=[", g.xmin, ",", g.xmax, "] dx=", g.dx,
          " Y=[", g.ymin, ",", g.ymax, "] dy=", g.dy,
          " Z=[", g.zmin, ",", g.zmax, "] dz=", g.dz,
          " size=(", nxx, ",", nyy, ",", nzz, "))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────────────────────────────────────

"""
    project_root(start_path::String) -> String

Walk upward from `start_path` until a directory containing both `config/` and
`database/` is found. Used to resolve YAML-relative data paths.
"""
function project_root(start_path::AbstractString)::String
    d = abspath(start_path)
    isfile(d) && (d = dirname(d))
    while true
        if isdir(joinpath(d, "database")) && isdir(joinpath(d, "config"))
            return d
        end
        parent = dirname(d)
        parent == d && return dirname(abspath(start_path))
        d = parent
    end
end

"""
    resolve_config_path(config_path::String) -> String

Return an existing YAML path. If `config.yaml` is missing, fall back to
`dataset_config.yaml` in the same directory.
"""
function resolve_config_path(config_path::AbstractString)::String
    p = abspath(config_path)
    isfile(p) && return p
    dir = dirname(p)
    for name in ("config.yaml", "dataset_config.yaml")
        alt = joinpath(dir, name)
        isfile(alt) && return alt
    end
    root = project_root(p)
    for name in ("config.yaml", "dataset_config.yaml")
        alt = joinpath(root, "config", name)
        isfile(alt) && return alt
    end
    error("Config YAML not found: tried $(p) and config/dataset_config.yaml under $(root)")
end

"""
    default_config_path() -> String

Locate `config/config.yaml` (or `dataset_config.yaml`) relative to this file.
"""
function default_config_path()::String
    root = project_root(@__DIR__)
    return resolve_config_path(joinpath(root, "config", "config.yaml"))
end

"""
    _stringify_keys(x) -> Any

Recursively convert dictionary keys to `String` so YAML lookups are type-stable
at the call site (`cfg["gridspec"]` rather than mixing `Symbol`/`String`).
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
    load_config(config_path::String) -> Dict{String,Any}

Parse the fusion YAML (`config.yaml` / `dataset_config.yaml`) and return a
string-keyed dictionary. Paths inside the dict are relative to `project_root`.
"""
function load_config(config_path::AbstractString)::Dict{String,Any}
    path = resolve_config_path(config_path)
    raw = YAML.load_file(path)
    cfg = _stringify_keys(raw)
    cfg isa Dict{String,Any} || error("Top-level YAML must be a mapping: $(path)")
    return cfg
end

"""
    _yaml_f32(x) -> Float32

Coerce a YAML scalar to `Float32` (metres / cell size).
"""
_yaml_f32(x::Real)::Float32 = Float32(x)
_yaml_f32(x::AbstractString)::Float32 = parse(Float32, x)
_yaml_f32(x)::Float32 = Float32(x)

"""
    _yaml_i32(x) -> Int32

Coerce a YAML scalar to `Int32` (EPSG code).
"""
_yaml_i32(x::Integer)::Int32 = Int32(x)
_yaml_i32(x::Real)::Int32 = Int32(round(Int, x))
_yaml_i32(x::AbstractString)::Int32 = Int32(parse(Int, x))

"""
    _gridspec_block(cfg, block) -> Dict{String,Any}

Select the `deposit` or `regional` GridSpec mapping from the parsed YAML.
"""
function _gridspec_block(cfg::AbstractDict, block::Union{Nothing,AbstractString})
    haskey(cfg, "gridspec") || error("YAML is missing a `gridspec:` section")
    gs = cfg["gridspec"]
    gs isa AbstractDict || error("`gridspec` must be a mapping")
    name = block === nothing ? string(get(gs, "active", "deposit")) : string(block)
    haskey(gs, name) || error("Unknown GridSpec block `$(name)`. Expected `deposit` or `regional`.")
    spec = gs[name]
    spec isa AbstractDict || error("gridspec.$(name) must be a mapping of xmin/xmax/…")
    return spec, name
end

"""
    load_gridspec(config_path::String; block=nothing) -> GridSpec

Read `xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz` and `epsg_code` from YAML.

# Arguments
- `config_path`: path to `config/config.yaml` (falls back to `dataset_config.yaml`)
- `block`: `"deposit"` or `"regional"`; defaults to `gridspec.active`

# Returns
`GridSpec` with all extents in metres, `z` in metres RL.

# Units
metres (easting / northing / RL); EPSG from `project.crs.epsg` (2391).
"""
function load_gridspec(config_path::AbstractString;
                       block::Union{Nothing,AbstractString}=nothing)::GridSpec
    cfg = load_config(config_path)
    return load_gridspec(cfg; block=block)
end

"""
    load_gridspec() -> GridSpec

Load the active GridSpec from the project `config/` directory.
"""
load_gridspec()::GridSpec = load_gridspec(default_config_path())

"""
    load_gridspec(cfg::AbstractDict; block=nothing) -> GridSpec

Build a `GridSpec` from an already-parsed config dictionary.
"""
function load_gridspec(cfg::AbstractDict;
                       block::Union{Nothing,AbstractString}=nothing)::GridSpec
    spec, _ = _gridspec_block(cfg, block)
    required = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "dx", "dy", "dz")
    for key in required
        haskey(spec, key) || error("gridspec is missing `$(key)`")
    end
    epsg = Int32(2391)
    if haskey(cfg, "project")
        proj = cfg["project"]
        if proj isa AbstractDict && haskey(proj, "crs")
            crs = proj["crs"]
            if crs isa AbstractDict && haskey(crs, "epsg")
                epsg = _yaml_i32(crs["epsg"])
            end
        end
    end
    g = GridSpec(
        _yaml_f32(spec["xmin"]), _yaml_f32(spec["xmax"]),
        _yaml_f32(spec["ymin"]), _yaml_f32(spec["ymax"]),
        _yaml_f32(spec["zmin"]), _yaml_f32(spec["zmax"]),
        _yaml_f32(spec["dx"]),   _yaml_f32(spec["dy"]),   _yaml_f32(spec["dz"]),
        epsg,
    )
    g.dx > 0 && g.dy > 0 && g.dz > 0 || error("GridSpec cell sizes must be positive")
    g.xmax > g.xmin && g.ymax > g.ymin && g.zmax > g.zmin ||
        error("GridSpec maxima must exceed minima")
    return g
end

# ─────────────────────────────────────────────────────────────────────────────
# Grid dimensions & coordinates
# ─────────────────────────────────────────────────────────────────────────────

"""
    nx(g::GridSpec) -> Int

Number of cells along easting (`X`). Dimension 1 of `(X, Y, Z, C)`.
"""
nx(g::GridSpec)::Int = max(1, Int(round((g.xmax - g.xmin) / g.dx)))

"""
    ny(g::GridSpec) -> Int

Number of cells along northing (`Y`). Dimension 2 of `(X, Y, Z, C)`.
"""
ny(g::GridSpec)::Int = max(1, Int(round((g.ymax - g.ymin) / g.dy)))

"""
    nz(g::GridSpec) -> Int

Number of cells along RL (`Z`). Dimension 3 of `(X, Y, Z, C)`.
"""
nz(g::GridSpec)::Int = max(1, Int(round((g.zmax - g.zmin) / g.dz)))

"""
    nxyz(g::GridSpec) -> Tuple{Int,Int,Int}

Grid shape `(nx, ny, nz)` matching Julia column-major tensor axes `(X, Y, Z)`.
"""
nxyz(g::GridSpec)::Tuple{Int,Int,Int} = (nx(g), ny(g), nz(g))

"""
    grid_size(g::GridSpec) -> Tuple{Int,Int,Int}

Alias for [`nxyz`](@ref).
"""
grid_size(g::GridSpec)::Tuple{Int,Int,Int} = nxyz(g)

Base.size(g::GridSpec) = nxyz(g)

"""
    x_centers(g::GridSpec) -> Vector{Float32}

Cell-centre easting coordinates (m), length `nx`.
"""
function x_centers(g::GridSpec)::Vector{Float32}
    n = nx(g)
    return Float32[g.xmin + (Float32(i) - 0.5f0) * g.dx for i in 1:n]
end

"""
    y_centers(g::GridSpec) -> Vector{Float32}

Cell-centre northing coordinates (m), length `ny`.
"""
function y_centers(g::GridSpec)::Vector{Float32}
    n = ny(g)
    return Float32[g.ymin + (Float32(j) - 0.5f0) * g.dy for j in 1:n]
end

"""
    z_centers(g::GridSpec) -> Vector{Float32}

Cell-centre RL coordinates (m, positive upward), length `nz`.
`z_centers[1]` is the deepest cell.
"""
function z_centers(g::GridSpec)::Vector{Float32}
    n = nz(g)
    return Float32[g.zmin + (Float32(k) - 0.5f0) * g.dz for k in 1:n]
end

"""
    in_bounds(g::GridSpec, x, y, z) -> Bool

`true` if the point (m) lies inside the closed-open grid box
`[xmin, xmax) × [ymin, ymax) × [zmin, zmax)`.
"""
function in_bounds(g::GridSpec, x::Real, y::Real, z::Real)::Bool
    return (g.xmin <= Float32(x) < g.xmax) &&
           (g.ymin <= Float32(y) < g.ymax) &&
           (g.zmin <= Float32(z) < g.zmax)
end

"""
    coord_to_index(g::GridSpec, x, y, z) -> Tuple{Int,Int,Int}

Convert world coordinates (m) to 1-based voxel indices `(i, j, k)`.
Indices may fall outside `1:n*` if the point is off-grid; callers must clamp
or skip.
"""
function coord_to_index(g::GridSpec, x::Real, y::Real, z::Real)::Tuple{Int,Int,Int}
    i = Int(floor((Float32(x) - g.xmin) / g.dx)) + 1
    j = Int(floor((Float32(y) - g.ymin) / g.dy)) + 1
    k = Int(floor((Float32(z) - g.zmin) / g.dz)) + 1
    return (i, j, k)
end

"""
    index_to_coord(g::GridSpec, i, j, k) -> Tuple{Float32,Float32,Float32}

World coordinates (m) of the centre of voxel `(i, j, k)`.
"""
function index_to_coord(g::GridSpec, i::Integer, j::Integer, k::Integer)::Tuple{Float32,Float32,Float32}
    x = g.xmin + (Float32(i) - 0.5f0) * g.dx
    y = g.ymin + (Float32(j) - 0.5f0) * g.dy
    z = g.zmin + (Float32(k) - 0.5f0) * g.dz
    return (x, y, z)
end

end # module GridSpecs
