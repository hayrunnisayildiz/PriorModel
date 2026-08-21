"""
    MeshParams

Shared mesh / grid contract for the MT-only prior pipeline.

All MT-only prior modules (synthetic model generator, forward modelling, U-Net,
training) **must import this file** and treat [`MeshParams`](@ref) as the single
source of truth for grid geometry and MT sampling.

# Data-type contract

## Resistivity model
- Type: `Matrix{Float32}`
- Shape: `(nz, nx)` — depth first, profile second (same axis order as
  `MTGeophysics.jl` solver arrays, restricted to the uniform CNN window).
- Values: `log10(ρ)` in Ω·m.

## MT response tensor
- Type: `Array{Float32,3}`
- Shape: `(n_stations, n_periods, n_components)` with `n_components = 2` by default:
  `[log10_apparent_resistivity, phase_degrees]` per station and period.
- When TE and TM modes are stored jointly, use `n_components = 4`:
  `[log10_ρ_TE, phase_TE, log10_ρ_TM, phase_TM]` (document this layout in the
  consuming module).

# Default mesh
[`DEFAULT_MESH`](@ref) follows the uniform core of `MTGeophysics.build_default_mt2d_mesh`
(COMMEMI 2-D benchmark): 45 × 400 m profile cells, 68 × 200 m ground cells,
11 stations, seven log-spaced periods matching the package default frequencies
`10.^range(-2, 2, length=7)`.

[`UNET_MESH`](@ref) is the canonical **network** survey (30 stations × 20
log-spaced periods, `T ∈ [10⁻³, 10³]` s, profile covering at least
`[-9000, 9000]` m) matching `scripts/build_train_pairs.jl`. Horizontal
cell size is 80 m (v4–v7 used 160 m at the same 19.2 km span).
Use [`station_positions`](@ref) for the training x-axis; resample field data
with `MTInputStandardizer.standardize_mt_input`.

# Example
```julia
include("src/synthetic/MeshParams.jl")
using .MTMeshParams

mp = DEFAULT_MESH
save_mesh_params(mp, "config/mesh_params.json")
mp2 = load_mesh_params("config/mesh_params.json")
```
"""
module MTMeshParams

using JLD2
using JSON3

export MeshParams, DEFAULT_MESH, UNET_MESH
export save_mesh_params, load_mesh_params
export n_periods, n_components, profile_length, depth_extent, frequencies_hz
export station_positions, validate_mesh_params
export unet_log_periods, UNET_N_PERIODS
export RESISTIVITY_LAYOUT, MT_DATA_LAYOUT, MT_DATA_LAYOUT_TETM

# ─────────────────────────────────────────────────────────────────────────────
# Data-layout constants (documented contract for downstream modules)
# ─────────────────────────────────────────────────────────────────────────────

"""Resistivity prior tensor: `Matrix{Float32}` of shape `(nz, nx)`, values log10(Ω·m)."""
const RESISTIVITY_LAYOUT = (axis_depth = 1, axis_profile = 2, log10_ohm_m = true)

"""
MT response tensor default layout: `(n_stations, n_periods, 2)` with
`[:, :, 1] = log10(apparent_resistivity)` and `[:, :, 2] = phase (degrees)`.
"""
const MT_DATA_LAYOUT = (
    axis_station = 1,
    axis_period = 2,
    axis_component = 3,
    components = ("log10_rho", "phase_deg"),
)

"""
Joint TE/TM layout: `(n_stations, n_periods, 4)` with
`[log10_ρ_TE, phase_TE, log10_ρ_TM, phase_TM]`.
"""
const MT_DATA_LAYOUT_TETM = (
    axis_station = 1,
    axis_period = 2,
    axis_component = 3,
    components = ("log10_rho_te", "phase_te_deg", "log10_rho_tm", "phase_tm_deg"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Mesh specification
# ─────────────────────────────────────────────────────────────────────────────

"""
    MeshParams

Uniform 2-D MT grid and survey sampling shared across the MT-only prior pipeline.

# Fields
- `nx`: number of horizontal (profile) cells
- `nz`: number of depth cells (positive `z` downward from the surface)
- `dx`: horizontal cell width, metres
- `dz`: depth cell height, metres
- `n_stations`: MT receiver count along the profile
- `periods`: period samples, seconds, typically log-spaced (`T = 1/f`)

Related frequencies: `frequencies_hz(mp) = 1 ./ mp.periods`.
"""
struct MeshParams
    nx::Int
    nz::Int
    dx::Float64
    dz::Float64
    n_stations::Int
    periods::Vector{Float64}
end

function Base.show(io::IO, mp::MeshParams)
    print(io, "MeshParams(cells=(", mp.nz, ",", mp.nx, ") Δ=(", mp.dz, ",", mp.dx,
          ") m stations=", mp.n_stations,
          " periods=", length(mp.periods),
          " T=[", mp.periods[1], ",", mp.periods[end], "] s)")
end

"""
    DEFAULT_MESH

COMMEMI-compatible defaults taken from `MTGeophysics.build_default_mt2d_mesh`:
- profile core: 45 cells × 400 m (`y_core_range = (-9000, 9000)`)
- ground column: 68 cells × 200 m (total 13.6 km, matching the default
  `ground_layers = vcat(fill(200, 8), fill(400, 10), fill(800, 10))`)
- 11 stations on `-8000:1600:8000` m
- 7 log-spaced periods inverse to `10.^range(-2, 2, length=7)` Hz
"""
const DEFAULT_MESH = MeshParams(
    45,                                                     # nx
    68,                                                     # nz
    400.0,                                                  # dx  (m)
    200.0,                                                  # dz  (m)
    11,                                                     # n_stations
    [1.0 / f for f in collect(10 .^ range(-2, 2, length = 7))],  # periods (s)
)

const UNET_N_PERIODS = 20

"""
    unet_log_periods(n=UNET_N_PERIODS) -> Vector{Float64}

Log-spaced periods covering typical broadband MT field surveys:
`T ∈ [10⁻³, 10³]` s. COMMEMI 2-D-I (`T ∈ [0.01, 100]` s) sits strictly inside
this interval, so resampling does not clamp COMMEMI periods as out-of-distribution.
"""
unet_log_periods(n::Integer = UNET_N_PERIODS) =
    collect(10 .^ range(-3, 3; length = Int(n)))

"""
    UNET_MESH

Canonical U-Net survey used by `scripts/build_train_pairs.jl`
(`SyntheticGenerator.GeneratorConfig` defaults):

- profile core: 240 cells × 80 m (`y_core_range = (-9600, 9600)`, 19.2 km)
  so `station_positions` span at least `[-9000, 9000]` m and COMMEMI's
  `±8000` m receivers stay in-distribution (v4–v7 used 120 × 160 m;
  v1–v3 used 120 × 25 m ≈ 3 km)
- target column: 48 cells × 25 m (unchanged; depth resolution is a
  separate experiment)
- 30 stations (every 8th core-cell centre; `receiver_stride = 8`;
  station spacing stays 640 m)
- 20 log-spaced periods `T = 10.^range(-3, 3, length=20)` s
  (field-MT band; frequencies `f = 1/T ∈ [10⁻³, 10³]` Hz)

[`DEFAULT_MESH`](@ref) remains the COMMEMI solver mesh (11 × 7). All MT-only
prior **network** I/O should use `UNET_MESH` (or a checkpoint `MeshParams`
with the same survey) via [`station_positions`](@ref).
"""
const UNET_MESH = MeshParams(
    240,                                                    # nx  (was 120 at dx=160 m)
    48,                                                     # nz  — unchanged
    80.0,                                                   # dx  (m)  — 19.2 km core
    25.0,                                                   # dz  (m)  — unchanged
    30,                                                     # n_stations
    unet_log_periods(),                                     # T ∈ [1e-3, 1e3] s
)

n_periods(mp::MeshParams)::Int = length(mp.periods)
n_components(::MeshParams; tetm::Bool = false)::Int = tetm ? 4 : 2
profile_length(mp::MeshParams)::Float64 = mp.nx * mp.dx
depth_extent(mp::MeshParams)::Float64 = mp.nz * mp.dz
frequencies_hz(mp::MeshParams)::Vector{Float64} = 1.0 ./ mp.periods

"""
    station_positions(nx, dx, n_stations) -> Vector{Float64}
    station_positions(mp::MeshParams) -> Vector{Float64}

Profile coordinates (m) of the MT receivers on the uniform core.

Matches synthetic training (`SyntheticGenerator` `receiver_stride`): cell
centres of the `nx` core cells, subsampled with
`stride = max(1, round(Int, nx / n_stations))`. This is **not** index-based
resampling of an arbitrary field survey; it is the training-grid x-axis.
"""
function station_positions(nx::Int, dx::Real, n_stations::Int)::Vector{Float64}
    nx > 0 || error("nx must be positive, got $nx")
    n_stations > 0 || error("n_stations must be positive, got $n_stations")
    dx > 0 || error("dx must be positive, got $dx")
    y1 = -Float64(nx) * Float64(dx) / 2
    centres = [(j - 0.5) * Float64(dx) + y1 for j in 1:nx]
    stride = max(1, round(Int, nx / n_stations))
    ys = centres[1:stride:end]
    length(ys) >= n_stations ||
        error("cannot place $n_stations stations on nx=$nx (got $(length(ys)) cell-centre samples)")
    return Float64.(ys[1:n_stations])
end

station_positions(mp::MeshParams)::Vector{Float64} =
    station_positions(mp.nx, mp.dx, mp.n_stations)

"""
    validate_mesh_params(mp; require_log_spaced_periods=false) -> MeshParams

Check field ranges and optionally assert that `periods` are strictly decreasing
(equivalent to log-spaced increasing frequencies).
"""
function validate_mesh_params(mp::MeshParams; require_log_spaced_periods::Bool = false)::MeshParams
    mp.nx > 0 || error("nx must be positive, got $(mp.nx)")
    mp.nz > 0 || error("nz must be positive, got $(mp.nz)")
    mp.dx > 0 || error("dx must be positive, got $(mp.dx)")
    mp.dz > 0 || error("dz must be positive, got $(mp.dz)")
    mp.n_stations > 0 || error("n_stations must be positive, got $(mp.n_stations)")
    length(mp.periods) > 0 || error("periods must be non-empty")
    all(>(0.0), mp.periods) || error("all periods must be positive")
    if require_log_spaced_periods
        all(diff(mp.periods) .< 0) ||
            error("periods should decrease (log-spaced increasing frequencies)")
    end
    return mp
end

# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

function _mesh_params_dict(mp::MeshParams)::Dict{String,Any}
    Dict{String,Any}(
        "schema" => "mesh_params/v1",
        "nx" => mp.nx,
        "nz" => mp.nz,
        "dx" => mp.dx,
        "dz" => mp.dz,
        "n_stations" => mp.n_stations,
        "periods" => mp.periods,
    )
end

function _mesh_params_from_dict(d::AbstractDict)::MeshParams
    mp = MeshParams(
        Int(d["nx"]),
        Int(d["nz"]),
        Float64(d["dx"]),
        Float64(d["dz"]),
        Int(d["n_stations"]),
        Float64.(collect(d["periods"])),
    )
    return validate_mesh_params(mp)
end

"""
    save_mesh_params(mp::MeshParams, path::String) -> String

Serialize mesh parameters to `path`. Format is chosen from the extension:
`.json` → JSON, `.jld2` / `.jld` → JLD2.
"""
function save_mesh_params(mp::MeshParams, path::AbstractString)::String
    out = abspath(path)
    ext = lowercase(splitext(out)[2])
    validate_mesh_params(mp)
    if ext == ".json"
        open(out, "w") do io
            JSON3.pretty(io, JSON3.write(_mesh_params_dict(mp)))
        end
    elseif ext in (".jld2", ".jld")
        jldsave(out; mesh_params = mp, schema = "mesh_params/v1")
    else
        error("unsupported mesh-params extension $(ext); use .json or .jld2")
    end
    return out
end

"""
    load_mesh_params(path::String) -> MeshParams

Load [`MeshParams`](@ref) from a JSON or JLD2 file written by [`save_mesh_params`](@ref).
"""
function load_mesh_params(path::AbstractString)::MeshParams
    src = abspath(path)
    isfile(src) || error("mesh params file not found: $src")
    ext = lowercase(splitext(src)[2])
    if ext == ".json"
        d = JSON3.read(read(src, String), Dict{String,Any})
        return _mesh_params_from_dict(d)
    elseif ext in (".jld2", ".jld")
        data = load(src)
        if haskey(data, "mesh_params")
            mp = data["mesh_params"]
            mp isa MeshParams || error("JLD2 file does not contain a MeshParams struct")
            return validate_mesh_params(mp)
        end
        # legacy / hand-written JLD2 with scalar fields
        return _mesh_params_from_dict(data)
    else
        error("unsupported mesh-params extension $(ext); use .json or .jld2")
    end
end

end # module MTMeshParams
