#!/usr/bin/env julia
#=
Input-shift gate: isolate standardize/clamp vs COMMEMI OOD vs in-distribution
network performance. U-Net checkpoint is fixed; export path is write_ini_prior.

Arms (identical VFSA: 1 chain, 200 ctrl, 100 iter, seed=20260308):
  (A) IN-DIST   — 8 validation tensors fed directly (no standardize_mt_input);
                  each prior scored against that sample's own synthetic .obs
  (B) CLEAN-OOD — Comemi2D1.true forward-modelled on the canonical 30×20 survey;
                  tensor → network (no standardize) → export → vs Comemi2D1.obs
  (C) CURRENT   — Comemi2D1.obs → standardize_mt_input → network → export →
                  vs Comemi2D1.obs  (+ clamp diagnostics)

Usage (from project root):
    julia --project=. scripts/input_shift_gate.jl

Writes: results/input_shift_gate.md
=#

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

get!(ENV, "GKSwstype", "nul")

using Printf, Dates, Statistics, HDF5, JSON3, MTGeophysics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics:
    load_trained_model, predict_prior, write_ini_prior, standardize_mt_input,
    pack_te_response, pack_tetm_response
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams:
    MeshParams, station_positions

include(joinpath(ROOT, "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator:
    GeneratorConfig, SyntheticModel, build_generator_mesh, forward_response

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

const CHECKPOINT_PATH = joinpath(ROOT, "models", "production_prior_v7.jld2")
const TRAIN_PAIRS_H5  = joinpath(ROOT, "data", "synthetic", "train_pairs_v7.h5")
const SPLIT_JSON      = joinpath(ROOT, "results", "train_val_split_v7.json")
const TRUE_MODEL_PATH = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
const OBS_COMMEMI     = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
const OUTPUT_DIR      = joinpath(ROOT, "results", "input_shift_gate")
const REPORT_PATH     = joinpath(ROOT, "results", "input_shift_gate.md")

const N_VAL_SAMPLES = 8

const VFSA_N_CHAINS = 1
const VFSA_N_CTRL   = 200
const VFSA_MAX_ITER = 100
const VFSA_SEED     = 20260308
const VFSA_N_TRIALS = 1
const VFSA_LOG_BOUNDS = (0.0, 4.0)

const IMPEDANCE_ERROR_FRACTION = 0.05
const BG_LOG10 = Float32(log10(100.0))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: .true parse / resample / clamp diagnostics / VFSA
# ─────────────────────────────────────────────────────────────────────────────

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
    idx += 1  # x sizes
    y_sizes = all_numbers[idx:idx+n_y_-1]; idx += n_y_
    z_sizes = all_numbers[idx:idx+n_z_-1]; idx += n_z_
    loge_values = all_numbers[idx:idx+n_y_*n_z_-1]
    return (; n_y=n_y_, n_z=n_z_, nza, y_sizes, z_sizes, loge_values)
end

function extract_ground_resistivity(ini; as_log10::Bool=true)
    rho = reshape(ini.loge_values, ini.n_y, ini.n_z)'
    ground = rho[ini.nza+1:end, :]
    as_log10 && (ground = ground ./ log(10.0))
    return ground
end

"""Cell-centre nearest-neighbour onto uniform `MeshParams` (log10 ρ)."""
function resample_to_mesh(ground_log10::Matrix{Float64},
                          src_y_sizes::Vector{Float64},
                          src_z_ground::Vector{Float64},
                          mp::MeshParams)::Matrix{Float32}
    src_nz, src_ny = size(ground_log10)
    src_y = cumsum(src_y_sizes) .- src_y_sizes ./ 2
    src_z = cumsum(src_z_ground) .- src_z_ground ./ 2
    tgt_y = [(j - 0.5) * mp.dx for j in 1:mp.nx]
    tgt_z = [(i - 0.5) * mp.dz for i in 1:mp.nz]
    src_mid = (minimum(src_y) + maximum(src_y)) / 2
    tgt_mid = (minimum(tgt_y) + maximum(tgt_y)) / 2
    src_y = src_y .+ (tgt_mid - src_mid)
    out = Matrix{Float32}(undef, mp.nz, mp.nx)
    for jt in 1:mp.nx
        jn = argmin(abs.(src_y .- tgt_y[jt]))
        for it in 1:mp.nz
            in_ = argmin(abs.(src_z .- tgt_z[it]))
            out[it, jt] = (in_ <= src_nz && jn <= src_ny) ?
                          Float32(ground_log10[in_, jn]) : BG_LOG10
        end
    end
    return out
end

"""
Count canonical survey queries that fall outside the observed (src) range
and are therefore edge-clamped by `standardize_mt_input`.
"""
function clamp_diagnostics(src_stations::AbstractVector,
                           src_periods::AbstractVector,
                           mp::MeshParams)
    tgt_x = station_positions(mp)
    tgt_T = Float64.(collect(mp.periods))
    x_lo, x_hi = extrema(Float64.(src_stations))
    T_lo, T_hi = extrema(Float64.(src_periods))
    st_idx = findall(x -> x < x_lo - 1e-12 || x > x_hi + 1e-12, tgt_x)
    T_idx  = findall(T -> T < T_lo - 1e-12 || T > T_hi + 1e-12, tgt_T)
    return (;
        n_stations_clamped = length(st_idx),
        n_periods_clamped  = length(T_idx),
        station_indices    = st_idx,
        period_indices     = T_idx,
        station_x_clamped  = tgt_x[st_idx],
        periods_clamped    = tgt_T[T_idx],
        src_x_range        = (x_lo, x_hi),
        src_T_range        = (T_lo, T_hi),
        tgt_x_range        = (minimum(tgt_x), maximum(tgt_x)),
        tgt_T_range        = (minimum(tgt_T), maximum(tgt_T)),
    )
end

function extract_initial_rms(run_dir::String)
    log_path = joinpath(run_dir, "chain_01", "0vfsa2DMT.log")
    isfile(log_path) || return nothing
    for line in readlines(log_path)
        if match(r"^\s+1\s+1\s+", line) !== nothing
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

function export_and_vfsa(logres::Matrix{Float32}, mp::MeshParams,
                         ini_path::String, obs_path::String; title::String)
    write_ini_prior(logres, ini_path, mp; title=title)
    result = run_vfsa(ini_path, obs_path)
    run_dir = result.run_info.run_dir
    return extract_initial_rms(run_dir), extract_best_rms(run_dir), run_dir
end

# ─────────────────────────────────────────────────────────────────────────────
# Arms
# ─────────────────────────────────────────────────────────────────────────────

"""(A) IN-DIST: validation tensors → network directly → own synthetic obs."""
function run_arm_in_dist(model, ps, st, mp::MeshParams, gmesh)
    isfile(TRAIN_PAIRS_H5) || error("Missing $TRAIN_PAIRS_H5")
    isfile(SPLIT_JSON)     || error("Missing $SPLIT_JSON")
    split = JSON3.read(read(SPLIT_JSON, String))
    val_idx = Int.(collect(split["val_idx"]))
    length(val_idx) >= N_VAL_SAMPLES ||
        error("val set has $(length(val_idx)) < $N_VAL_SAMPLES samples")
    sample_ids = val_idx[1:N_VAL_SAMPLES]

    X_all = h5open(TRAIN_PAIRS_H5, "r") do f; read(f["X"]); end
    Y_all = h5open(TRAIN_PAIRS_H5, "r") do f; read(f["Y"]); end
    n_comp = size(X_all, 3)
    n_comp == Int(model.in_channels) ||
        error("X channels=$n_comp ≠ model.in_channels=$(model.in_channels)")

    arm_dir = joinpath(OUTPUT_DIR, "A_indist")
    mkpath(arm_dir)
    inits = Float64[]
    bests = Float64[]
    rows = String[]

    for (k, idx) in enumerate(sample_ids)
        println("\n  [A $k/$N_VAL_SAMPLES] val_idx=$idx")
        mt = Array{Float32,3}(X_all[:, :, :, idx])          # (30, 20, C)
        Y  = Matrix{Float32}(Y_all[:, :, idx])              # (48, 120)

        # Own obs: re-forward the stored resistivity on the generator mesh
        syn = SyntheticModel(Y, :train_pairs_val, idx, Dict{String,Any}("val_idx" => idx))
        resp = forward_response(syn, gmesh; mode=:TETM)
        obs_path = joinpath(arm_dir, "sample_$(idx).obs")
        MTGeophysics.write_data2d(obs_path, resp;
                                  impedance_error_fraction=IMPEDANCE_ERROR_FRACTION,
                                  title="IN-DIST val_idx=$idx synthetic obs")

        logres = predict_prior(model, ps, st, mt)
        ini_path = joinpath(arm_dir, "sample_$(idx).ini")
        init_rms, best_rms, run_dir = export_and_vfsa(
            logres, mp, ini_path, obs_path;
            title="IN-DIST prior val_idx=$idx (no standardize)")
        push!(inits, something(init_rms, NaN))
        push!(bests, something(best_rms, NaN))
        @printf("    init=%.4f  best=%.4f  run=%s\n",
                something(init_rms, NaN), something(best_rms, NaN), run_dir)
        push!(rows, @sprintf("| %d | %d | %.4f | %.4f |", k, idx,
                             something(init_rms, NaN), something(best_rms, NaN)))
    end

    return (;
        mean_init = mean(inits), std_init = std(inits),
        mean_best = mean(bests), std_best = std(bests),
        inits, bests, sample_ids, rows,
    )
end

"""(B) CLEAN-OOD: true model → canonical forward → network (no standardize)."""
function run_arm_clean_ood(model, ps, st, mp::MeshParams, gmesh)
    println("\n── (B) CLEAN-OOD ──")
    ini = read_ini_model(TRUE_MODEL_PATH)
    ground = extract_ground_resistivity(ini; as_log10=true)
    src_z = ini.z_sizes[ini.nza+1:end]
    target = resample_to_mesh(ground, ini.y_sizes, src_z, mp)

    syn = SyntheticModel(target, :commemi_true_resampled, 0,
                         Dict{String,Any}("source" => TRUE_MODEL_PATH))
    resp = forward_response(syn, gmesh; mode=:TETM)
    raw = if Int(model.in_channels) == 4
        pack_tetm_response(resp.rho_xy, resp.phase_xy, resp.rho_yx, resp.phase_yx)
    else
        pack_te_response(resp.rho_xy, resp.phase_xy)
    end
    @printf("  canonical tensor %s (no standardize)\n", size(raw))

    logres = predict_prior(model, ps, st, raw)
    arm_dir = joinpath(OUTPUT_DIR, "B_clean_ood")
    mkpath(arm_dir)
    ini_path = joinpath(arm_dir, "clean_ood_prior.ini")
    init_rms, best_rms, run_dir = export_and_vfsa(
        logres, mp, ini_path, OBS_COMMEMI;
        title="CLEAN-OOD — COMMEMI true on canonical survey (no standardize)")
    @printf("  init=%.4f  best=%.4f\n", something(init_rms, NaN), something(best_rms, NaN))
    return (; init_rms, best_rms, run_dir, ini_path)
end

"""(C) CURRENT: Comemi2D1.obs → standardize → network (+ clamp log)."""
function run_arm_current(model, ps, st, mp::MeshParams)
    println("\n── (C) CURRENT ──")
    data = MTGeophysics.load_data2d(OBS_COMMEMI)
    src_y = Float64.(data.receivers)
    src_T = Float64.(data.periods)
    clamp_info = clamp_diagnostics(src_y, src_T, mp)
    @printf("  clamp: %d/%d stations, %d/%d periods\n",
            clamp_info.n_stations_clamped, mp.n_stations,
            clamp_info.n_periods_clamped, length(mp.periods))
    @printf("  station indices: %s\n", clamp_info.station_indices)
    @printf("  period  indices: %s\n", clamp_info.period_indices)

    raw = if Int(model.in_channels) == 4
        pack_tetm_response(data.rho_xy, data.phase_xy, data.rho_yx, data.phase_yx)
    else
        pack_te_response(data.rho_xy, data.phase_xy)
    end
    mt = standardize_mt_input(src_y, src_T, raw; mp=mp, method=:bilinear)
    logres = predict_prior(model, ps, st, mt)

    arm_dir = joinpath(OUTPUT_DIR, "C_current")
    mkpath(arm_dir)
    ini_path = joinpath(arm_dir, "current_prior.ini")
    init_rms, best_rms, run_dir = export_and_vfsa(
        logres, mp, ini_path, OBS_COMMEMI;
        title="CURRENT — Comemi2D1.obs via standardize_mt_input")
    @printf("  init=%.4f  best=%.4f\n", something(init_rms, NaN), something(best_rms, NaN))
    return (; init_rms, best_rms, run_dir, ini_path, clamp_info)
end

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

function write_report(A, B, C, mp::MeshParams)
    mkpath(dirname(REPORT_PATH))
    A_mean = A.mean_init
    B_init = something(B.init_rms, NaN)
    C_init = something(C.init_rms, NaN)
    clamp = C.clamp_info
    n_st = mp.n_stations
    n_T  = length(mp.periods)

    open(REPORT_PATH, "w") do io
        println(io, "# Input-shift gate")
        println(io)
        println(io, "Checkpoint: `models/production_prior_v7.jld2`")
        println(io, "Export: `write_ini_prior` (current defaults)")
        @printf(io, "VFSA: n_chains=%d, n_ctrl=%d, max_iter=%d, seed=%d\n",
                VFSA_N_CHAINS, VFSA_N_CTRL, VFSA_MAX_ITER, VFSA_SEED)
        println(io)
        println(io, "## Results")
        println(io)
        println(io, "| kol | initial RMS | best RMS | notes |")
        println(io, "|---|---:|---:|---|")
        @printf(io, "| (A) IN-DIST | %.4f ± %.4f | %.4f ± %.4f | mean±std over %d val samples |\n",
                A.mean_init, A.std_init, A.mean_best, A.std_best, N_VAL_SAMPLES)
        @printf(io, "| (B) CLEAN-OOD | %.4f | %.4f | COMMEMI true → canonical forward |\n",
                B_init, something(B.best_rms, NaN))
        @printf(io, "| (C) CURRENT | %.4f | %.4f | obs → standardize → network |\n",
                C_init, something(C.best_rms, NaN))
        println(io)
        println(io, "### (A) per-sample")
        println(io)
        println(io, "| k | val_idx | initial RMS | best RMS |")
        println(io, "|---:|---:|---:|---:|")
        for row in A.rows
            println(io, row)
        end
        println(io)
        println(io, "## (C) standardize_mt_input clamp diagnostics")
        println(io)
        @printf(io, "- observed: stations x∈[%.0f, %.0f] m, periods T∈[%.4g, %.4g] s\n",
                clamp.src_x_range[1], clamp.src_x_range[2],
                clamp.src_T_range[1], clamp.src_T_range[2])
        @printf(io, "- canonical: %d stations x∈[%.0f, %.0f] m, %d periods T∈[%.4g, %.4g] s\n",
                n_st, clamp.tgt_x_range[1], clamp.tgt_x_range[2],
                n_T, clamp.tgt_T_range[1], clamp.tgt_T_range[2])
        @printf(io, "- **stations clamped:** %d / %d — indices `%s` (x = %s m)\n",
                clamp.n_stations_clamped, n_st, clamp.station_indices,
                join([@sprintf("%.0f", x) for x in clamp.station_x_clamped], ", "))
        @printf(io, "- **periods clamped:** %d / %d — indices `%s` (T = %s s)\n",
                clamp.n_periods_clamped, n_T, clamp.period_indices,
                join([@sprintf("%.4g", T) for T in clamp.periods_clamped], ", "))
        println(io)
        println(io, "## Interpretation")
        println(io)
        @printf(io, "1. **(B) − (C) = %.4f** — standardizer/clamp cost on initial RMS.\n",
                B_init - C_init)
        @printf(io, "2. **(A) − (B) = %.4f** — cost of COMMEMI lying outside the training distribution (clean canonical forward of the true model vs in-dist val mean).\n",
                A_mean - B_init)
        @printf(io, "3. **(A) = %.4f ± %.4f** — network base performance in data space (no survey mismatch).\n",
                A.mean_init, A.std_init)
        println(io, "4. If |(B)−(C)| is large, fix/replace `standardize_mt_input` clamping before more training.")
        println(io, "5. If |(A)−(B)| is large while |(B)−(C)| is small, the bottleneck is COMMEMI OOD content, not the standardizer.")
        println(io)
        println(io, "Generated: $(Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
    end
    return REPORT_PATH
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    mkpath(OUTPUT_DIR)
    println("═" ^ 65)
    println(" Input-shift gate (v7 checkpoint)")
    println("═" ^ 65)

    model, ps, st, mp = load_trained_model(CHECKPOINT_PATH)
    @printf("Checkpoint mesh: %s  in_channels=%d\n", mp, Int(model.in_channels))
    gmesh = build_generator_mesh(GeneratorConfig())

    println("\n── (A) IN-DIST ──")
    A = run_arm_in_dist(model, ps, st, mp, gmesh)
    @printf("  mean init=%.4f ± %.4f\n", A.mean_init, A.std_init)

    B = run_arm_clean_ood(model, ps, st, mp, gmesh)
    C = run_arm_current(model, ps, st, mp)

    report = write_report(A, B, C, mp)
    println("\n", "═" ^ 65)
    println(" Report: $report")
    println("═" ^ 65)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
