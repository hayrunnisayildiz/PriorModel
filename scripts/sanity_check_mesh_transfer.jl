#!/usr/bin/env julia
#=
Mesh-transfer gate (U-Net bypass): feed COMMEMI ground truth through the
prior-export path under three depth conditions, plus a homogeneous reference,
and compare VFSA2DMT initial/best RMS.

Arms (identical VFSA settings: 1 chain, 200 ctrl, 100 iter, seed=20260308):
  (a) FLOOR      — examples/0COMEMI2D-I/Comemi2D1.true as start model
  (b) FULL-DEPTH — .true ground resampled onto the generator solver mesh
                   (full ground column incl. geometric deep zone) and written
                   in write_ini_prior's LOGE layout with that mesh's cell sizes
  (c) TRUNCATED  — same model, top 1200 m only (UNET_MESH 48×25 m), exported
                   with write_ini_prior as currently shipped
  (+) HOMOGENEOUS — examples/0COMEMI2D-I/Comemi2D1.ini

Usage (from project root):
    julia --project=. scripts/sanity_check_mesh_transfer.jl

Writes: results/mesh_transfer_gate.md
=#

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

get!(ENV, "GKSwstype", "nul")

using Printf, Dates, MTGeophysics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics: write_ini_prior, _write_vector_lines
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams:
    MeshParams, UNET_MESH

include(joinpath(ROOT, "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator:
    GeneratorConfig, build_generator_mesh, n_y, n_z, y_centers, z_centers

# ─────────────────────────────────────────────────────────────────────────────
# Constants (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

const TRUE_MODEL_PATH = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
const HOMO_MODEL_PATH = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.ini")
const OBS_DATA_PATH   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
const OUTPUT_DIR      = joinpath(ROOT, "results", "mesh_transfer_gate")
const REPORT_PATH     = joinpath(ROOT, "results", "mesh_transfer_gate.md")

const VFSA_N_CHAINS = 1
const VFSA_N_CTRL   = 200
const VFSA_MAX_ITER = 100
const VFSA_SEED     = 20260308
const VFSA_N_TRIALS = 1
const VFSA_LOG_BOUNDS = (0.0, 4.0)

const TARGET_DEPTH_M = UNET_MESH.nz * UNET_MESH.dz  # 48 × 25 m = 1200 m
const BG_LOG10 = Float32(log10(100.0))
const AIR_RESISTIVITY = 1.0e9
const BACKGROUND_RESISTIVITY = 100.0

# Cell-centre nearest-neighbour on log10(ρ); origins aligned by profile midpoints.
const RESAMPLE_METHOD = "cell-centre nearest neighbour (log10 ρ; y origins mid-aligned)"

# ─────────────────────────────────────────────────────────────────────────────
# .ini parser (MTGeophysics write_model2d / LOGE)
# ─────────────────────────────────────────────────────────────────────────────

"""
    read_ini_model(path) -> NamedTuple

Parse an MTGeophysics `.ini` / `.true` model file. Values are log_e(ρ).
"""
function read_ini_model(path::String)
    isfile(path) || error("Model file not found: $path")
    lines = readlines(path)

    nza = 0
    data_lines = String[]
    for line in lines
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, "#")
            m = match(r"NZA\s*=\s*(\d+)", stripped)
            m !== nothing && (nza = parse(Int, m.captures[1]))
            continue
        end
        push!(data_lines, stripped)
    end

    header = split(data_lines[1])
    n_y_ = parse(Int, header[2])
    n_z_ = parse(Int, header[3])

    all_numbers = Float64[]
    for i in 2:length(data_lines)
        append!(all_numbers, parse.(Float64, split(data_lines[i])))
    end

    idx = 1
    _nx = 1
    idx += _nx
    y_sizes = all_numbers[idx:idx+n_y_-1]; idx += n_y_
    z_sizes = all_numbers[idx:idx+n_z_-1]; idx += n_z_
    n_vals = n_y_ * n_z_
    loge_values = all_numbers[idx:idx+n_vals-1]; idx += n_vals
    origin = (all_numbers[idx], all_numbers[idx+1], all_numbers[idx+2]); idx += 3
    rotation = all_numbers[idx]

    return (; n_y=n_y_, n_z=n_z_, nza, y_sizes, z_sizes, loge_values, origin, rotation)
end

"""
    extract_ground_resistivity(ini; as_log10=true) -> Matrix{Float64}

Ground-zone resistivity, air rows stripped. Shape `(nz_ground, n_y)`.
"""
function extract_ground_resistivity(ini; as_log10::Bool=true)
    (; n_y, n_z, nza, loge_values) = ini
    rho = reshape(loge_values, n_y, n_z)'  # (n_z, n_y)
    ground = rho[nza+1:end, :]
    if as_log10
        ground = ground ./ log(10.0)
    end
    return ground
end

function ground_depth_m(ini)::Float64
    sum(ini.z_sizes[ini.nza+1:end])
end

# ─────────────────────────────────────────────────────────────────────────────
# Resample
# ─────────────────────────────────────────────────────────────────────────────

"""
    resample_nn(ground_log10, src_y_sizes, src_z_ground, tgt_y_centres, tgt_z_centres)

Cell-centre nearest-neighbour resample in log10(ρ). Source and target y origins
are mid-aligned (profile centres coincide).
"""
function resample_nn(ground_log10::Matrix{Float64},
                     src_y_sizes::Vector{Float64},
                     src_z_ground::Vector{Float64},
                     tgt_y_centres::Vector{Float64},
                     tgt_z_centres::Vector{Float64})::Matrix{Float32}
    src_nz, src_ny = size(ground_log10)
    src_y_centres = cumsum(src_y_sizes) .- src_y_sizes ./ 2
    src_z_centres = cumsum(src_z_ground) .- src_z_ground ./ 2

    # Mid-align: shift source so its midpoint matches target midpoint.
    src_mid = (minimum(src_y_centres) + maximum(src_y_centres)) / 2
    tgt_mid = (minimum(tgt_y_centres) + maximum(tgt_y_centres)) / 2
    src_y_centres = src_y_centres .+ (tgt_mid - src_mid)

    nzt, nyt = length(tgt_z_centres), length(tgt_y_centres)
    out = Matrix{Float32}(undef, nzt, nyt)
    for jt in 1:nyt
        jn = argmin(abs.(src_y_centres .- tgt_y_centres[jt]))
        for it in 1:nzt
            in_ = argmin(abs.(src_z_centres .- tgt_z_centres[it]))
            if in_ <= src_nz && jn <= src_ny
                out[it, jt] = Float32(ground_log10[in_, jn])
            else
                out[it, jt] = BG_LOG10
            end
        end
    end
    return out
end

"""Resample onto a uniform `MeshParams` grid (write_ini_prior path)."""
function resample_to_mesh(ground_log10::Matrix{Float64},
                          src_y_sizes::Vector{Float64},
                          src_z_ground::Vector{Float64},
                          mp::MeshParams)::Matrix{Float32}
    tgt_y = [(j - 0.5) * mp.dx for j in 1:mp.nx]
    tgt_z = [(i - 0.5) * mp.dz for i in 1:mp.nz]
    return resample_nn(ground_log10, src_y_sizes, src_z_ground, tgt_y, tgt_z)
end

# ─────────────────────────────────────────────────────────────────────────────
# FULL-DEPTH export (generator solver mesh; write_ini_prior LOGE layout)
# ─────────────────────────────────────────────────────────────────────────────

"""
Write a core-ground log10(ρ) grid on the generator solver mesh as a LOGE `.ini`.

`write_ini_prior` only emits uniform `mp.nz × mp.dz` ground cells, so the
geometric deep zone of `GeneratorMesh` cannot be represented through it.
This helper uses the same LOGE layout / `_write_vector_lines` as
`write_ini_prior`, but keeps the generator's air + ground cell sizes.
"""
function write_ini_generator_mesh(log10_ground::Matrix{Float32},
                                  output_path::String,
                                  mesh;
                                  title::String="FULL-DEPTH generator mesh prior")
    n_air = mesh.n_air_cells
    n_ground = n_z(mesh) - n_air
    core = mesh.core_y
    n_core = length(core)
    size(log10_ground) == (n_ground, n_core) ||
        error("grid $(size(log10_ground)) ≠ generator ground×core ($n_ground, $n_core)")

    ny = n_y(mesh)
    nz = n_z(mesh)
    full_rho = fill(BACKGROUND_RESISTIVITY, nz, ny)

    for iz in 1:n_air
        full_rho[iz, :] .= AIR_RESISTIVITY
    end

    left, right = first(core), last(core)
    for (jj, iy) in enumerate(core), iz_g in 1:n_ground
        full_rho[n_air + iz_g, iy] = 10.0^Float64(log10_ground[iz_g, jj])
    end
    # Lateral pad: replicate nearest core column (solver_resistivity :replicate)
    for iz in (n_air+1):nz
        vl, vr = full_rho[iz, left], full_rho[iz, right]
        for iy in 1:(left-1)
            full_rho[iz, iy] = vl
        end
        for iy in (right+1):ny
            full_rho[iz, iy] = vr
        end
    end

    mkpath(dirname(abspath(output_path)))
    air_top = mesh.z_nodes[1]
    y_nodes_start = mesh.y_nodes[1]

    open(output_path, "w") do io
        println(io, "# $title")
        println(io, "# NZA=$n_air")
        println(io, "1 $ny $nz 0 LOGE")
        _write_vector_lines(io, [1.0])
        _write_vector_lines(io, mesh.y_cell_sizes)
        _write_vector_lines(io, mesh.z_cell_sizes)

        values = Float64[]
        sizehint!(values, ny * nz)
        for iz in 1:nz, iy in 1:ny
            push!(values, log(full_rho[iz, iy]))
        end
        _write_vector_lines(io, values)

        @printf(io, "%.8e %.8e %.8e\n", 0.0, y_nodes_start, air_top)
        println(io, "0.0")
    end

    @info "FULL-DEPTH generator INI written" path=output_path n_z=nz n_y=ny n_ground n_air
    return output_path
end

# ─────────────────────────────────────────────────────────────────────────────
# RMS extraction
# ─────────────────────────────────────────────────────────────────────────────

function extract_initial_rms(run_dir::String)
    log_path = joinpath(run_dir, "chain_01", "0vfsa2DMT.log")
    isfile(log_path) || return nothing
    for line in readlines(log_path)
        m = match(r"^\s+1\s+1\s+", line)
        if m !== nothing
            cols = split(strip(line))
            length(cols) >= 7 || continue
            return parse(Float64, cols[7])
        end
    end
    return nothing
end

function extract_best_rms(run_dir::String)
    summary_path = joinpath(run_dir, "Summary.md")
    isfile(summary_path) || return nothing
    for line in readlines(summary_path)
        m = match(r"best_chain_rms:\s*([\d.]+)", line)
        m !== nothing && return parse(Float64, m.captures[1])
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# VFSA runner
# ─────────────────────────────────────────────────────────────────────────────

function run_vfsa(start_model_path::String, data_path::String)
    MTGeophysics.VFSA2DMT(
        MTGeophysics.VFSA2DMTParams(
            script_path      = @__FILE__,
            start_model_path = start_model_path,
            data_path        = data_path,
            config = MTGeophysics.VFSA2DMTConfig(
                n_chains    = VFSA_N_CHAINS,
                n_ctrl      = VFSA_N_CTRL,
                max_iter    = VFSA_MAX_ITER,
                n_trials    = VFSA_N_TRIALS,
                log_bounds  = VFSA_LOG_BOUNDS,
                seed        = VFSA_SEED,
                keep_models = true,
            ),
        ),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct ArmResult
    name::String
    ground_depth_m::Float64
    n_z::Int
    n_air::Int
    initial_rms::Union{Float64,Nothing}
    best_rms::Union{Float64,Nothing}
    start_model::String
    run_dir::String
end

function write_report(arms::Vector{ArmResult}, extra_notes::Vector{String})
    mkpath(dirname(REPORT_PATH))
    open(REPORT_PATH, "w") do io
        println(io, "# Mesh-transfer gate")
        println(io)
        println(io, "U-Net bypass: COMMEMI true model through prior-transfer paths.")
        println(io)
        println(io, "## VFSA settings")
        println(io)
        @printf(io, "- n_chains = %d, n_ctrl = %d, max_iter = %d, seed = %d\n",
                VFSA_N_CHAINS, VFSA_N_CTRL, VFSA_MAX_ITER, VFSA_SEED)
        println(io, "- data: `examples/0COMEMI2D-I/Comemi2D1.obs`")
        println(io)
        println(io, "## Resampling")
        println(io)
        println(io, RESAMPLE_METHOD)
        println(io)
        for note in extra_notes
            println(io, note)
        end
        println(io)
        println(io, "## Results")
        println(io)
        println(io, "| kol adı | mesh yer derinliği (m) | n_z | n_air | initial RMS | best RMS |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for a in arms
            init_s = a.initial_rms === nothing ? "n/a" : @sprintf("%.4f", a.initial_rms)
            best_s = a.best_rms === nothing ? "n/a" : @sprintf("%.4f", a.best_rms)
            @printf(io, "| %s | %.1f | %d | %d | %s | %s |\n",
                    a.name, a.ground_depth_m, a.n_z, a.n_air, init_s, best_s)
        end
        println(io)
        println(io, "## Artefacts")
        println(io)
        for a in arms
            println(io, "- **$(a.name)**: start=`$(a.start_model)`, run_dir=`$(a.run_dir)`")
        end
        println(io)
        println(io, "Generated: $(Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
    end
    return REPORT_PATH
end

function run_arm(name::String, start_path::String)::ArmResult
    ini = read_ini_model(start_path)
    println("\n── $name ──")
    @printf("  start: %s\n", start_path)
    @printf("  mesh: n_z=%d n_air=%d ground_depth=%.1f m\n",
            ini.n_z, ini.nza, ground_depth_m(ini))
    println("  Running VFSA2DMT...")
    result = run_vfsa(start_path, OBS_DATA_PATH)
    run_dir = result.run_info.run_dir
    init_rms = extract_initial_rms(run_dir)
    best_rms = extract_best_rms(run_dir)
    @printf("  initial RMS = %s   best RMS = %s\n",
            init_rms === nothing ? "n/a" : @sprintf("%.4f", init_rms),
            best_rms === nothing ? "n/a" : @sprintf("%.4f", best_rms))
    return ArmResult(
        name = name,
        ground_depth_m = ground_depth_m(ini),
        n_z = ini.n_z,
        n_air = ini.nza,
        initial_rms = init_rms,
        best_rms = best_rms,
        start_model = start_path,
        run_dir = run_dir,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    mkpath(OUTPUT_DIR)

    println("═" ^ 65)
    println(" Mesh-transfer gate (U-Net bypass)")
    println("═" ^ 65)

    # Source: COMMEMI true ground
    true_ini = read_ini_model(TRUE_MODEL_PATH)
    ground_log10 = extract_ground_resistivity(true_ini; as_log10=true)
    src_z_ground = true_ini.z_sizes[true_ini.nza+1:end]
    @printf("Source .true: n_y=%d n_z=%d n_air=%d ground_depth=%.1f m\n",
            true_ini.n_y, true_ini.n_z, true_ini.nza, ground_depth_m(true_ini))

    # Generator solver mesh (FULL-DEPTH target)
    gmesh = build_generator_mesh(GeneratorConfig())
    n_air_g = gmesh.n_air_cells
    ground_idx = (n_air_g+1):n_z(gmesh)
    core = gmesh.core_y
    tgt_y = y_centers(gmesh)[core]
    tgt_z = z_centers(gmesh)[ground_idx]
    full_grid = resample_nn(ground_log10, true_ini.y_sizes, src_z_ground, tgt_y, tgt_z)
    full_ini_path = joinpath(OUTPUT_DIR, "full_depth_generator.ini")
    write_ini_generator_mesh(full_grid, full_ini_path, gmesh;
                             title="FULL-DEPTH — COMMEMI true on generator solver mesh")
    @printf("FULL-DEPTH grid: %s  ground_depth=%.1f m  n_air=%d\n",
            size(full_grid), sum(gmesh.z_cell_sizes[ground_idx]), n_air_g)

    # Truncated target zone (current write_ini_prior path)
    trunc_grid = resample_to_mesh(ground_log10, true_ini.y_sizes, src_z_ground, UNET_MESH)
    trunc_ini_path = joinpath(OUTPUT_DIR, "truncated_unet_1200m.ini")
    write_ini_prior(trunc_grid, trunc_ini_path, UNET_MESH;
                    title="TRUNCATED — COMMEMI true top $(Int(TARGET_DEPTH_M)) m (UNET_MESH)")
    @printf("TRUNCATED grid: %s  ground_depth=%.1f m (UNET_MESH)\n",
            size(trunc_grid), TARGET_DEPTH_M)

    notes = String[
        "- **FLOOR**: native `Comemi2D1.true` (no resample / no re-export).",
        "- **FULL-DEPTH**: $(RESAMPLE_METHOD) onto `build_generator_mesh(GeneratorConfig())` " *
            "ground cell centres (target zone + geometric deep zone); LOGE export with " *
            "generator `y_cell_sizes` / `z_cell_sizes` via `_write_vector_lines` " *
            "(`write_ini_prior` cannot emit non-uniform deep layers).",
        "- **TRUNCATED**: $(RESAMPLE_METHOD) onto `UNET_MESH` " *
            "($(UNET_MESH.nz)×$(UNET_MESH.dz) m = $(Int(TARGET_DEPTH_M)) m); " *
            "`write_ini_prior` current defaults.",
        "- **HOMOGENEOUS**: native `Comemi2D1.ini`.",
    ]

    arms = ArmResult[
        run_arm("FLOOR", TRUE_MODEL_PATH),
        run_arm("FULL-DEPTH", full_ini_path),
        run_arm("TRUNCATED", trunc_ini_path),
        run_arm("HOMOGENEOUS", HOMO_MODEL_PATH),
    ]

    report = write_report(arms, notes)
    println("\n", "═" ^ 65)
    println(" Report: $report")
    println("═" ^ 65)

    println("\nRaw RMS table:")
    for a in arms
        @printf("  %-12s  depth=%8.1f m  n_z=%3d  n_air=%2d  init=%s  best=%s\n",
                a.name, a.ground_depth_m, a.n_z, a.n_air,
                a.initial_rms === nothing ? "n/a" : @sprintf("%.4f", a.initial_rms),
                a.best_rms === nothing ? "n/a" : @sprintf("%.4f", a.best_rms))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
