"""
    ExportPriorToMTGeophysics

Export a trained [`MTResistivityUNet2D`](@ref) prior as an `.ini` start-model
file consumable by `MTGeophysics.jl`'s `VFSA2DMT` inversion engine.

The `.ini` format follows `MTGeophysics.write_model2d` exactly:

```
# <title>
# NZA=<n_air_cells>
1 <n_y> <n_z> 0 LOGE
<x_cell_sizes>          (always [1.0] for a 2-D profile)
<y_cell_sizes>
<z_cell_sizes>
<log_e(ρ) values, iz-major>
<origin: 3 floats>
<rotation: 1 float>
```

# Single-call workflow

```julia
include("src/inference/export_prior_to_mtgeophysics.jl")
using .ExportPriorToMTGeophysics

generate_prior("obs/site.h5", "models/best_mt_resistivity_prior.jld2",
               "output/prior_start.ini")
```

The returned path is ready for `VFSA2DMTParams(start_model_path = ...)`.
"""
module ExportPriorToMTGeophysics

using JLD2
using HDF5
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "synthetic", "MeshParams.jl"))
include(joinpath(ROOT, "src", "networks", "mt_resistivity_unet2d.jl"))

using .MTMeshParams: MeshParams, DEFAULT_MESH, n_periods, validate_mesh_params, frequencies_hz
using .MTResistivityUNet2DLayers: MTResistivityUNet2D, replace_nan, add_batch_dim

export load_trained_model, predict_prior, write_ini_prior, generate_prior
export load_mt_observations

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_trained_model(checkpoint_path) -> (model, ps, st, mesh_params)

Load a trained `MTResistivityUNet2D` checkpoint (JLD2) written by
`train_mt_resistivity.jl`.
"""
function load_trained_model(checkpoint_path::String)
    isfile(checkpoint_path) || error("Checkpoint not found: $checkpoint_path")
    ckpt = load(checkpoint_path)
    model = ckpt["model"]::MTResistivityUNet2D
    ps    = ckpt["ps"]
    st    = ckpt["st"]
    mp    = get(ckpt, "mesh_params", DEFAULT_MESH)::MeshParams
    return model, ps, st, mp
end

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

"""
    predict_prior(model, ps, st, mt_data::Array{Float32,3}) -> Matrix{Float32}

Run a forward pass and return `(nz, nx)` log10-resistivity grid.

`mt_data` shape: `(n_stations, n_periods, n_components)`.
"""
function predict_prior(model::MTResistivityUNet2D, ps, st,
                       mt_data::Array{Float32,3})::Matrix{Float32}
    logres, _ = model(mt_data, ps, st)
    return Float32.(logres)  # (nz, nx)
end

# ─────────────────────────────────────────────────────────────────────────────
# .ini writer  (mirrors MTGeophysics.write_model2d exactly)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _write_vector_lines(io, values; per_line=12)

Write a vector of floats in `%.8e` format, `per_line` values per line.
Identical to `MTGeophysics._write_vector_lines`.
"""
function _write_vector_lines(io::IO, values::AbstractVector{<:Real}; per_line::Int=12)
    for first_idx in 1:per_line:length(values)
        last_idx = min(first_idx + per_line - 1, length(values))
        println(io, join([@sprintf("%.8e", Float64(values[idx]))
                          for idx in first_idx:last_idx], " "))
    end
end

"""
    write_ini_prior(resistivity_grid, output_path, mp; kwargs...)

Write a log10-resistivity grid as an MTGeophysics.jl-compatible `.ini` model file.

# Arguments
- `resistivity_grid`: `(nz, nx)` `Matrix{Float32}` of **log10(ρ)** values
- `output_path`: destination `.ini` path
- `mp`: [`MeshParams`](@ref) that defines the mesh geometry

# Keyword arguments
- `title`: header comment (default `"PriorModel neural prior"`)
- `n_air_cells`: air rows prepended above the target zone (default `8`)
- `air_resistivity`: air-cell ρ in Ω·m (default `1e9`)
- `background_resistivity`: ρ for lateral padding cells, Ω·m (default `100.0`)
- `y_padding`: total lateral padding extent, m (default `9000.0`)
- `pad_factor`: geometric growth factor for padding cells (default `1.25`)
- `air_top`: top of air column, m — negative above surface (default `-12000.0`)

The file follows the `write_model2d` format:
  `1 n_y n_z 0 LOGE`, then cell sizes, then `log_e(ρ)` values row by row
  (iz-major: for iz in 1:n_z, for iy in 1:n_y).
"""
function write_ini_prior(resistivity_grid::Matrix{Float32},
                         output_path::String,
                         mp::MeshParams;
                         title::String="PriorModel neural prior",
                         n_air_cells::Int=8,
                         air_resistivity::Float64=1e9,
                         background_resistivity::Float64=100.0,
                         y_padding::Float64=9000.0,
                         pad_factor::Float64=1.25,
                         air_top::Float64=-12000.0)
    mp = validate_mesh_params(mp)
    nz_target, nx_core = size(resistivity_grid)
    nz_target == mp.nz || error("grid rows ($nz_target) ≠ mp.nz ($(mp.nz))")
    nx_core == mp.nx   || error("grid cols ($nx_core) ≠ mp.nx ($(mp.nx))")

    # ── Build the full solver mesh (core + padding + air) ─────────────────────

    # Horizontal: core cells
    y_core_half = (nx_core * mp.dx) / 2.0
    y_core_sizes = fill(mp.dx, nx_core)

    # Lateral padding (geometric growth, both sides)
    left_pad = Float64[]
    Δy = mp.dx
    accum = 0.0
    while accum < y_padding - 1e-9
        Δy *= pad_factor
        push!(left_pad, Δy)
        accum += Δy
    end
    right_pad = copy(left_pad)
    reverse!(left_pad)
    y_cell_sizes = vcat(left_pad, y_core_sizes, right_pad)
    n_y = length(y_cell_sizes)

    # Vertical: air + ground target
    z_air = collect(range(air_top, 0.0, length=n_air_cells + 1))
    z_air_sizes = diff(z_air)
    z_ground_sizes = fill(mp.dz, nz_target)
    z_cell_sizes = vcat(z_air_sizes, z_ground_sizes)
    n_z = length(z_cell_sizes)

    # ── Build full resistivity array (linear Ω·m → log_e for file) ───────────
    # Full array: (n_z, n_y) in linear Ω·m
    full_rho = fill(background_resistivity, n_z, n_y)

    # Air cells
    for iz in 1:n_air_cells
        full_rho[iz, :] .= air_resistivity
    end

    # Target zone (convert log10 → linear)
    n_pad_left = length(left_pad)
    for iz in 1:nz_target, ix in 1:nx_core
        full_rho[n_air_cells + iz, n_pad_left + ix] = 10.0^Float64(resistivity_grid[iz, ix])
    end

    # ── Write .ini file ──────────────────────────────────────────────────────
    mkpath(dirname(abspath(output_path)))
    y_nodes_start = -sum(left_pad) - y_core_half

    open(output_path, "w") do io
        println(io, "# $title")
        println(io, "# NZA=$n_air_cells")
        println(io, "1 $n_y $n_z 0 LOGE")
        _write_vector_lines(io, [1.0])
        _write_vector_lines(io, y_cell_sizes)
        _write_vector_lines(io, z_cell_sizes)

        values = Float64[]
        sizehint!(values, n_y * n_z)
        for iz in 1:n_z, iy in 1:n_y
            push!(values, log(full_rho[iz, iy]))
        end
        _write_vector_lines(io, values)

        @printf(io, "%.8e %.8e %.8e\n", 0.0, y_nodes_start, air_top)
        println(io, "0.0")
    end

    @info "INI prior written" path=output_path n_z n_y nz_target nx_core
    return output_path
end

# ─────────────────────────────────────────────────────────────────────────────
# MT observation loading
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_mt_observations(path; mp=nothing) -> Array{Float32,3}

Load MT observations from an HDF5 file.

Expected keys (in order of preference):
- `"X"` — `(n_stations, n_periods, n_components)` or `(n_stations, n_periods, n_components, 1)`
- `"mt_data"` — same layout
- `"log10_rho_app"` + `"phase_deg"` — each `(n_stations, n_periods)`, stacked on dim 3

Returns `(n_stations, n_periods, n_components)` Float32 array.
"""
function load_mt_observations(path::String; mp::Union{Nothing,MeshParams}=nothing)::Array{Float32,3}
    isfile(path) || error("MT observation file not found: $path")
    h5open(path, "r") do f
        if haskey(f, "X")
            x = Float32.(read(f["X"]))
            ndims(x) == 4 && size(x, 4) == 1 && (x = x[:, :, :, 1])
            return x
        elseif haskey(f, "mt_data")
            x = Float32.(read(f["mt_data"]))
            ndims(x) == 4 && size(x, 4) == 1 && (x = x[:, :, :, 1])
            return x
        elseif haskey(f, "log10_rho_app") && haskey(f, "phase_deg")
            rho = Float32.(read(f["log10_rho_app"]))
            phase = Float32.(read(f["phase_deg"]))
            return cat(rho, phase; dims=3)
        end
        error("Cannot find MT data keys in $path (tried X, mt_data, log10_rho_app+phase_deg)")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Single-call workflow
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_prior(mt_obs_path, checkpoint_path, output_ini_path; kwargs...) -> String

End-to-end workflow: load MT observations → predict log10(ρ) prior →
write `.ini` file ready for `VFSA2DMTParams(start_model_path = ...)`.

# Arguments
- `mt_obs_path`: HDF5 file with MT response data (see [`load_mt_observations`](@ref))
- `checkpoint_path`: JLD2 checkpoint from `train_mt_resistivity.jl`
- `output_ini_path`: destination `.ini` file

# Keyword arguments
Forwarded to [`write_ini_prior`](@ref) (padding, air cells, title, etc.).

# Returns
The absolute path to the written `.ini` file.
"""
function generate_prior(mt_obs_path::String,
                        checkpoint_path::String,
                        output_ini_path::String;
                        kwargs...)
    model, ps, st, mp = load_trained_model(checkpoint_path)
    mt_data = load_mt_observations(mt_obs_path; mp=mp)
    logres = predict_prior(model, ps, st, mt_data)
    write_ini_prior(logres, output_ini_path, mp; kwargs...)
    return abspath(output_ini_path)
end

end # module ExportPriorToMTGeophysics
