"""
    MTForwardBatch

Batch forward modelling for the MT-only prior pipeline.

Reads uniform `log10(ρ)` models produced by [`ResistivityModelGenerator`](@ref)
(`model_<id>.jld2` with `model` + `mp`) and runs them through MTGeophysics.jl's
2-D TE forward solver to build `(MT data, resistivity model)` training pairs.

# Data contract
- MT input `X`: `(n_stations, n_periods, 2, n_samples)` — `[log10 ρ_a, phase_deg]`
- Resistivity target `Y`: `(nz, nx, n_samples)` — `log10(ρ)` on [`MeshParams`](@ref)

# Example
```julia
include("src/synthetic/MTForwardBatch.jl")
using .MTForwardBatch

mp = DEFAULT_MESH
batch_generate_training_pairs(
    "data/synthetic/resistivity_models",
    "data/processed/mt_resistivity_pairs.h5",
    mp,
)
```
"""
module MTForwardBatch

using HDF5
using JLD2
using MTGeophysics
using Printf
using ProgressMeter
using Random
using Statistics

include("MeshParams.jl")
using .MTMeshParams: MeshParams, DEFAULT_MESH, validate_mesh_params
using .MTMeshParams: n_periods, frequencies_hz, profile_length, station_positions

function _coerce_mesh_params(mp)::MeshParams
    mp isa MeshParams && return validate_mesh_params(mp)
    required = (:nx, :nz, :dx, :dz, :n_stations, :periods)
    all(hasproperty(mp, p) for p in required) ||
        error("`mp` is not a MeshParams-compatible object: $(typeof(mp))")
    return validate_mesh_params(MeshParams(
        Int(mp.nx), Int(mp.nz), Float64(mp.dx), Float64(mp.dz),
        Int(mp.n_stations), Float64.(collect(mp.periods)),
    ))
end

mesh_params_equal(a::MeshParams, b::MeshParams)::Bool =
    a.nx == b.nx && a.nz == b.nz && a.dx == b.dx && a.dz == b.dz &&
    a.n_stations == b.n_stations && a.periods == b.periods

export forward_mt_response, add_noise!, batch_generate_training_pairs
export build_solver_context, SolverContext, DEFAULT_MESH
export list_resistivity_model_files, load_resistivity_model_file

const SCHEMA_VERSION = "mt_resistivity_pairs/v1"
const AIR_RESISTIVITY::Float64 = 1.0e9

# ─────────────────────────────────────────────────────────────────────────────
# Solver mesh bridge (MeshParams ↔ MTGeophysics.MT2DMesh)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SolverContext

Cached MTGeophysics solver mesh and index ranges that map a uniform
`(nz, nx)` `MeshParams` model onto the padded `(n_z, n_y)` resistivity array.
"""
struct SolverContext
    mesh::MTGeophysics.MT2DMesh
    core_y::UnitRange{Int}
    target_z::UnitRange{Int}
    n_air_cells::Int
end

"""
    _default_y_core_range(mp::MeshParams) -> Tuple{Float64,Float64}

Horizontal core extent centred on zero, matching COMMEMI-style profiles.
"""
function _default_y_core_range(mp::MeshParams)::Tuple{Float64,Float64}
    half = profile_length(mp) / 2
    return (-half, half)
end

"""
    _default_receiver_positions(mp::MeshParams) -> Vector{Float64}

Receivers on the uniform core, matching synthetic training
([`station_positions`](@ref)): cell centres subsampled to `mp.n_stations`.
"""
function _default_receiver_positions(mp::MeshParams)::Vector{Float64}
    return station_positions(mp)
end

"""
    _ground_layer_sizes(mp::MeshParams) -> Vector{Float64}

Uniform target column (`nz × dz`) followed by geometrically growing deep
layers so the lowest periods see a conductive half-space continuation.
"""
function _ground_layer_sizes(mp::MeshParams)::Vector{Float64}
    deep = vcat(fill(400.0, 10), fill(800.0, 10))
    return vcat(fill(Float64(mp.dz), mp.nz), deep)
end

"""
    build_solver_context(mp::MeshParams) -> SolverContext

Build an MTGeophysics.jl mesh from [`MeshParams`](@ref) and record the index
ranges that align the uniform CNN window with the solver grid.
"""
function build_solver_context(mp::MeshParams)::SolverContext
    mp = validate_mesh_params(mp)
    y1, y2 = _default_y_core_range(mp)
    ground_layers = _ground_layer_sizes(mp)

    mesh = MTGeophysics.build_mt2d_mesh(
        frequencies = frequencies_hz(mp),
        y_core_range = (y1, y2),
        y_core_cell = mp.dx,
        y_padding = 12_000.0,
        pad_factor = 1.25,
        air_top = -15_000.0,
        air_cells = 6,
        ground_layers = ground_layers,
        receiver_positions = _default_receiver_positions(mp),
    )

    y_centers = MTGeophysics.mt2d_y_centers(mesh)
    in_core = [(y1 - 1e-6) <= yc <= (y2 + 1e-6) for yc in y_centers]
    core_indices = findall(in_core)
    length(core_indices) == mp.nx ||
        error("expected $(mp.nx) core columns on solver mesh, found $(length(core_indices))")
    core_y = first(core_indices):last(core_indices)

    target_z = (mesh.n_air_cells + 1):(mesh.n_air_cells + mp.nz)
    last(target_z) <= length(mesh.z_cell_sizes) ||
        error("MeshParams nz=$(mp.nz) exceeds solver ground column")

    length(mesh.receiver_positions) == mp.n_stations ||
        @warn "receiver count mismatch" expected=mp.n_stations got=length(mesh.receiver_positions)
    length(mesh.frequencies) == n_periods(mp) ||
        @warn "period count mismatch" expected=n_periods(mp) got=length(mesh.frequencies)

    return SolverContext(mesh, core_y, target_z, mesh.n_air_cells)
end

"""
    _expand_to_solver_grid(log10_rho::AbstractMatrix{<:Real}, ctx::SolverContext)
        -> Matrix{Float64}

Embed a `(nz, nx)` log10-resistivity window into the full `(n_z, n_y)` linear-Ω·m
array expected by [`MTGeophysics.run_mt2d_forward`](@ref).
"""
function _expand_to_solver_grid(log10_rho::AbstractMatrix{<:Real},
                                ctx::SolverContext)::Matrix{Float64}
    nzt, nyc = size(log10_rho)
    expected = (length(ctx.target_z), length(ctx.core_y))
    (nzt, nyc) == expected ||
        error("model size $(size(log10_rho)) does not match solver window $expected")

    mesh = ctx.mesh
    n_z = length(mesh.z_cell_sizes)
    n_y = length(mesh.y_cell_sizes)
    full = Matrix{Float64}(undef, n_z, n_y)

    @inbounds for (jj, iy) in enumerate(ctx.core_y), (ii, iz) in enumerate(ctx.target_z)
        full[iz, iy] = Float64(log10_rho[ii, jj])
    end

    z_deep = (last(ctx.target_z) + 1):n_z
    @inbounds for (jj, iy) in enumerate(ctx.core_y)
        base = Float64(log10_rho[end, jj])
        for iz in z_deep
            full[iz, iy] = base
        end
    end

    left, right = first(ctx.core_y), last(ctx.core_y)
    ground = (ctx.n_air_cells + 1):n_z
    @inbounds for iz in ground
        vl, vr = full[iz, left], full[iz, right]
        for iy in 1:(left - 1)
            full[iz, iy] = vl
        end
        for iy in (right + 1):n_y
            full[iz, iy] = vr
        end
    end

    ρ = 10.0 .^ full
    ρ[1:ctx.n_air_cells, :] .= AIR_RESISTIVITY
    return ρ
end

"""
    _response_to_mt_tensor(response::MTGeophysics.MT2DResponse, mp::MeshParams)
        -> Array{Float32,3}

Convert `MT2DResponse` (TE: `rho_xy`, `phase_xy` sized `(n_freq, n_station)`)
to `(n_stations, n_periods, 2)` with log10 apparent resistivity and phase (deg).
"""
function _response_to_mt_tensor(response::MTGeophysics.MT2DResponse,
                                mp::MeshParams)::Array{Float32,3}
    n_r = mp.n_stations
    n_f = n_periods(mp)
    size(response.rho_xy) == (n_f, n_r) ||
        error("TE rho_xy size $(size(response.rho_xy)) != ($n_f, $n_r)")
    size(response.phase_xy) == (n_f, n_r) ||
        error("TE phase_xy size $(size(response.phase_xy)) != ($n_f, $n_r)")

    out = zeros(Float32, n_r, n_f, 2)
    @inbounds for ir in 1:n_r, ip in 1:n_f
        ρa = Float64(response.rho_xy[ip, ir])
        out[ir, ip, 1] = ρa > 0 ? log10(ρa) : NaN32
        out[ir, ip, 2] = Float32(response.phase_xy[ip, ir])
    end
    all(isfinite, out) || error("non-finite MT response after forward solve")
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

"""
    forward_mt_response(resistivity_model::Matrix{Float32}, mp::MeshParams;
                        ctx=nothing, mode=:TE) -> Array{Float32,3}

Run MTGeophysics.jl's 2-D forward solver on a uniform log10-resistivity model.

Returns `(n_stations, n_periods, 2)` with `[log10 ρ_a, phase_deg]` per the
[`MeshParams`](@ref) MT data contract. Uses TE mode by default for the MT-only
pipeline (`mode=:TE`; pass `:TM` or `:TETM` if needed).
"""
function forward_mt_response(resistivity_model::Matrix{Float32}, mp::MeshParams;
                             ctx::Union{SolverContext,Nothing}=nothing,
                             mode::Symbol=:TE)
    mp = validate_mesh_params(mp)
    size(resistivity_model) == (mp.nz, mp.nx) ||
        error("resistivity_model size $(size(resistivity_model)) != ($(mp.nz), $(mp.nx))")

    ctx === nothing && (ctx = build_solver_context(mp))
    ρ = _expand_to_solver_grid(resistivity_model, ctx)
    response = MTGeophysics.run_mt2d_forward(ctx.mesh, ρ; mode=mode)
    return _response_to_mt_tensor(response, mp)
end

"""
    add_noise!(data::Array{Float32,3}; noise_level::Float64=0.05,
               rng=Random.default_rng()) -> Array{Float32,3}

Add relative Gaussian noise (`noise_level` × value × 𝒩(0,1)) independently to
log10 apparent resistivity (`[:,:,1]`) and phase degrees (`[:,:,2]`).
"""
function add_noise!(data::Array{Float32,3};
                    noise_level::Float64=0.05,
                    rng::AbstractRNG=Random.default_rng())::Array{Float32,3}
    ndims(data) == 3 || error("expected 3-D MT tensor, got $(ndims(data)) dimensions")
    size(data, 3) == 2 || error("expected 2 MT components, got $(size(data, 3))")
    noise_level >= 0 || error("noise_level must be non-negative")

    logρ = @view data[:, :, 1]
    phase = @view data[:, :, 2]
    logρ .+= Float32(noise_level) .* randn(rng, size(logρ)) .* max.(abs.(logρ), 1.0f-3)
    phase .+= Float32(noise_level) .* randn(rng, size(phase)) .* max.(abs.(phase), 1.0f0)
    return data
end

"""
    list_resistivity_model_files(model_dir::String) -> Vector{String}

Sorted paths to `model_*.jld2` files from [`ResistivityModelGenerator`](@ref).
"""
function list_resistivity_model_files(model_dir::AbstractString)::Vector{String}
    root = abspath(model_dir)
    isdir(root) || error("model directory not found: $root")
    files = filter(re -> startswith(re, "model_") && endswith(re, ".jld2"), readdir(root))
    sort!(files)
    return [joinpath(root, f) for f in files]
end

"""
    load_resistivity_model_file(path::String) -> (model::Matrix{Float32}, mp::MeshParams)

Load one Prompt-4 model file (`model` + `mp` keys).
"""
function load_resistivity_model_file(path::AbstractString)
    data = load(abspath(path))
    haskey(data, "model") || error("missing `model` in $(path)")
    haskey(data, "mp") || error("missing `mp` in $(path)")
    model = data["model"] isa Matrix{Float32} ? data["model"] : Matrix{Float32}(data["model"])
    mp = _coerce_mesh_params(data["mp"])
    return model, mp
end

"""Write MeshParams fields as HDF5 root attributes."""
function _write_mesh_attrs(g, mp::MeshParams)
    attrs = attributes(g)
    attrs["schema"] = SCHEMA_VERSION
    attrs["mesh_nx"] = mp.nx
    attrs["mesh_nz"] = mp.nz
    attrs["mesh_dx"] = mp.dx
    attrs["mesh_dz"] = mp.dz
    attrs["mesh_n_stations"] = mp.n_stations
    attrs["mesh_n_periods"] = n_periods(mp)
    attrs["mesh_periods"] = mp.periods
    attrs["X_layout"] = "n_stations,n_periods,2,n_samples"
    attrs["Y_layout"] = "nz,nx,n_samples"
    attrs["X_components"] = "log10_apparent_resistivity,phase_deg"
    attrs["Y_units"] = "log10_ohm_m"
    return nothing
end

"""
    batch_generate_training_pairs(model_dir::String, output_path::String,
                                  mp::MeshParams; noise_level::Float64=0.05,
                                  mode=:TE, seed=nothing) -> String

Forward every `model_*.jld2` in `model_dir` and write a single HDF5 file:

- `X`: `(n_stations, n_periods, 2, n_samples)` — noisy MT data
- `Y`: `(nz, nx, n_samples)` — true log10-resistivity models

[`MeshParams`](@ref) fields are stored as HDF5 root attributes.
"""
function batch_generate_training_pairs(model_dir::AbstractString,
                                       output_path::AbstractString,
                                       mp::MeshParams;
                                       noise_level::Float64=0.05,
                                       mode::Symbol=:TE,
                                       seed::Union{Nothing,Integer}=nothing)
    mp = validate_mesh_params(mp)
    paths = list_resistivity_model_files(model_dir)
    isempty(paths) && error("no model_*.jld2 files in $(abspath(model_dir))")

    n_samples = length(paths)
    n_r = mp.n_stations
    n_f = n_periods(mp)
    ctx = build_solver_context(mp)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(Int(seed))

    out = abspath(output_path)
    mkpath(dirname(out))

    X = zeros(Float32, n_r, n_f, 2, n_samples)
    Y = zeros(Float32, mp.nz, mp.nx, n_samples)

    @info "Forward batch" n_samples n_stations=n_r n_periods=n_f grid=(mp.nz, mp.nx) out=out

    pm = Progress(n_samples; desc="MT forward ", showspeed=true)
    try
        @inbounds for (i, path) in enumerate(paths)
            model, file_mp = load_resistivity_model_file(path)
            !mesh_params_equal(file_mp, mp) &&
                @warn "MeshParams mismatch; using batch mp" file=basename(path) batch=mp file_mp=file_mp

            mt = forward_mt_response(model, mp; ctx=ctx, mode=mode)
            add_noise!(mt; noise_level=noise_level, rng=rng)
            X[:, :, :, i] = mt
            Y[:, :, i] = model
            next!(pm)
        end
    finally
        finish!(pm)
    end

    HDF5.h5open(out, "w") do h5
        h5["X"] = X
        h5["Y"] = Y
        _write_mesh_attrs(h5, mp)
        attrs = attributes(h5)
        attrs["n_samples"] = n_samples
        attrs["noise_level"] = noise_level
        attrs["forward_mode"] = string(mode)
        attrs["source_dir"] = abspath(model_dir)
    end

    @info "Wrote training pairs" path=out size_X=size(X) size_Y=size(Y)
    return out
end

end # module MTForwardBatch
