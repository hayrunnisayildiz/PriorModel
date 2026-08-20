#!/usr/bin/env julia
#=
Evaluate mid_scale_prior_v2.jld2 on the COMMEMI 2-D benchmark.

Usage (from project root):
    julia --project=. scripts/evaluate_mid_scale_v2.jl

Colab (cwd is /content — never `--project=.` from there):
    julia --project=/content/PriorModel /content/PriorModel/scripts/evaluate_mid_scale_v2.jl

Steps:
  1. Standardize COMMEMI .obs onto the U-Net survey, write HDF5, call generate_prior
  2. Run VFSA2DMT with identical settings to all previous tests
     (1 chain, 200 ctrl, 100 iter, seed=20260308)
  3. Print five-way RMS table + automatic interpretation
  4. Heatmap: predicted prior vs resampled COMMEMI true model
     → results/evaluate_v2/predicted_vs_true.png
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf, Dates, Statistics, HDF5, Plots, MTGeophysics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics:
    generate_prior, write_ini_prior, load_trained_model, predict_prior,
    load_mt_observations, standardize_mt_input
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams: MeshParams

# Historical RMS from previous identical-setting VFSA2DMT runs
const HOMO_INIT   = 12.31
const HOMO_BEST   = 5.7354
const UNET20_INIT = 61.50
const UNET20_BEST = 8.3573
const V1_INIT     = 54.39
const V1_BEST     = 9.9083
const TRUE_INIT   = 1.30
const TRUE_BEST   = 1.3002

"""
    commemi_obs_to_mt_tensor(obs_path, mp) -> Array{Float32,3}

Load COMMEMI `.obs` via `MTGeophysics.load_data2d` and resample onto the
U-Net survey with [`standardize_mt_input`](@ref).
"""
function commemi_obs_to_mt_tensor(obs_path::String, mp::MeshParams)::Array{Float32,3}
    isfile(obs_path) || error("COMMEMI obs not found: $obs_path")
    data = MTGeophysics.load_data2d(obs_path)
    src_y = Float64.(data.receivers)
    src_T = Float64.(data.periods)
    out = standardize_mt_input(data; mp=mp, method=:bilinear)
    nS, nP = size(out, 1), size(out, 2)
    @printf("  COMMEMI survey: %d stations  y∈[%.0f, %.0f] m,  %d periods T∈[%.4g, %.4g] s\n",
            length(src_y), minimum(src_y), maximum(src_y),
            length(src_T), minimum(src_T), maximum(src_T))
    @printf("  U-Net survey:   %d stations  %d periods T∈[%.4g, %.4g] s  tensor=%s\n",
            nS, nP, minimum(mp.periods), maximum(mp.periods), size(out))
    return out
end

function write_mt_h5(path::String, mt_data::Array{Float32,3})
    mkpath(dirname(abspath(path)))
    h5open(path, "w") do f
        f["X"] = mt_data
        f["mt_data"] = mt_data
        a = HDF5.attributes(f)
        a["source"] = "COMMEMI Comemi2D1.obs (interpolated to U-Net survey)"
        a["schema"] = "commemi_mt/v2"
    end
    return abspath(path)
end

# ─────────────────────────────────────────────────────────────────────────────
# COMMEMI true-model resample (same algorithm as sanity_check_mesh_transfer.jl)
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
    n_y = parse(Int, header[2])
    n_z = parse(Int, header[3])
    all_numbers = Float64[]
    for i in 2:length(data_lines)
        append!(all_numbers, parse.(Float64, split(data_lines[i])))
    end
    idx = 1
    _nx = 1
    idx += _nx
    y_sizes = all_numbers[idx:idx + n_y - 1]; idx += n_y
    z_sizes = all_numbers[idx:idx + n_z - 1]; idx += n_z
    n_vals = n_y * n_z
    loge_values = all_numbers[idx:idx + n_vals - 1]
    return (; n_y, n_z, nza, y_sizes, z_sizes, loge_values)
end

function extract_ground_resistivity(ini; as_log10::Bool=true)
    (; n_y, n_z, nza, loge_values) = ini
    rho = reshape(loge_values, n_y, n_z)'
    ground = rho[nza + 1:end, :]
    as_log10 && (ground = ground ./ log(10.0))
    return ground
end

"""Nearest-neighbour resample of a non-uniform grid onto `mp` (log10 ρ)."""
function resample_to_uniform(ground_log10::Matrix{Float64},
                             src_y_sizes::Vector{Float64},
                             src_z_sizes_ground::Vector{Float64},
                             mp::MeshParams)
    src_nz, src_ny = size(ground_log10)
    src_y_centres = cumsum(src_y_sizes) .- src_y_sizes ./ 2
    src_z_centres = cumsum(src_z_sizes_ground) .- src_z_sizes_ground ./ 2
    tgt_y_centres = [(j - 0.5) * mp.dx for j in 1:mp.nx]
    tgt_z_centres = [(i - 0.5) * mp.dz for i in 1:mp.nz]
    src_y_centres .+= (mp.nx * mp.dx - sum(src_y_sizes)) / 2
    out = Matrix{Float32}(undef, mp.nz, mp.nx)
    for jt in 1:mp.nx
        jn = argmin(abs.(src_y_centres .- tgt_y_centres[jt]))
        for it in 1:mp.nz
            in_ = argmin(abs.(src_z_centres .- tgt_z_centres[it]))
            if in_ <= src_nz && jn <= src_ny
                out[it, jt] = Float32(ground_log10[in_, jn])
            else
                out[it, jt] = Float32(log10(100.0))
            end
        end
    end
    return out
end

function load_resampled_commemi_true(true_path::String, mp::MeshParams)::Matrix{Float32}
    ini = read_ini_model(true_path)
    ground_log10 = extract_ground_resistivity(ini; as_log10=true)
    src_z_ground = ini.z_sizes[ini.nza + 1:end]
    return resample_to_uniform(ground_log10, ini.y_sizes, src_z_ground, mp)
end

# ─────────────────────────────────────────────────────────────────────────────
# RMS extraction (identical to evaluate_mid_scale.jl / sanity_check)
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
# VFSA runner — identical settings to every previous COMMEMI test
# ─────────────────────────────────────────────────────────────────────────────

function run_vfsa_eval(start_model_path::String, data_path::String)
    MTGeophysics.VFSA2DMT(
        MTGeophysics.VFSA2DMTParams(
            script_path      = @__FILE__,
            start_model_path = start_model_path,
            data_path        = data_path,
            config = MTGeophysics.VFSA2DMTConfig(
                n_chains    = 1,
                n_ctrl      = 200,
                max_iter    = 100,
                n_trials    = 1,
                log_bounds  = (0.0, 4.0),
                seed        = 20260308,
                keep_models = true,
            ),
        ),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Table, interpretation, heatmap
# ─────────────────────────────────────────────────────────────────────────────

fmt(x) = x === nothing ? "N/A" : @sprintf("%.4f", x)

function interpret_v2(v2_best)
    if v2_best === nothing
        return "v2 best RMS okunamadı — VFSA çıktısını kontrol edin."
    elseif v2_best < HOMO_BEST
        return "Homojeni geçti — production ölçeğine (n=1000+) geçmeye hazır"
    elseif v2_best < V1_BEST
        return "Doğru yönde iyileşme, ama henüz homojeni geçmedi — overfitting/kapasite sınırı sinyaline göre daha fazla veri gerekebilir"
    else
        return "Düzeltmeler RMS'i iyileştirmedi, sorun train/val split'ten farklı bir yerde — mimari veya veri çeşitliliği incelenmeli"
    end
end

function print_five_way_table(v2_init, v2_best)
    println("─" ^ 72)
    @printf("  %-36s  initial=%-12s  best=%-12s\n",
            "Homojen:", fmt(HOMO_INIT), fmt(HOMO_BEST))
    @printf("  %-36s  initial=%-12s  best=%-12s\n",
            "U-Net (20 örnek/10ep):", fmt(UNET20_INIT), fmt(UNET20_BEST))
    @printf("  %-36s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v1):", fmt(V1_INIT), fmt(V1_BEST))
    @printf("  %-36s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v2*):", fmt(v2_init), fmt(v2_best))
    @printf("  %-36s  initial=%-12s  best=%-12s\n",
            "Gerçek model (sanity):", fmt(TRUE_INIT), fmt(TRUE_BEST))
    println("─" ^ 72)
    println("  *v2 = düzeltilmiş train/val split + LR + grad clipping + best_val checkpoint")
end

function plot_predicted_vs_true(pred::Matrix{Float32}, truth::Matrix{Float32},
                                out_png::String)
    lo = min(minimum(pred), minimum(truth))
    hi = max(maximum(pred), maximum(truth))
    if hi - lo < 1.0f-3
        lo, hi = lo - 0.1f0, hi + 0.1f0
    end

    p_true = Plots.heatmap(truth;
        title="COMMEMI true (resampled)",
        xlabel="Profile cell", ylabel="Depth cell",
        yflip=true, color=:turbo, clims=(lo, hi),
        colorbar_title="log₁₀(ρ) [Ω·m]",
        aspect_ratio=:equal)
    p_pred = Plots.heatmap(pred;
        title="U-Net v2 prior",
        xlabel="Profile cell", ylabel="Depth cell",
        yflip=true, color=:turbo, clims=(lo, hi),
        colorbar_title="log₁₀(ρ) [Ω·m]",
        aspect_ratio=:equal)
    fig = Plots.plot(p_true, p_pred;
        layout=(1, 2), size=(1400, 520), dpi=150,
        plot_title="Predicted vs true (shared color scale)")
    mkpath(dirname(abspath(out_png)))
    Plots.savefig(fig, out_png)
    @info "Heatmap saved" path=out_png clims=(lo, hi)
    return out_png
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    checkpoint   = joinpath(ROOT, "models", "mid_scale_prior_v2.jld2")
    vfsa_data    = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    true_model   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
    results_dir  = joinpath(ROOT, "results", "evaluate_v2")
    mkpath(results_dir)

    isfile(checkpoint) || error("v2 checkpoint not found: $checkpoint")
    isfile(vfsa_data)  || error("COMMEMI obs not found: $vfsa_data")
    isfile(true_model) || error("COMMEMI true model not found: $true_model")

    println("═" ^ 72)
    println(" Mid-scale v2 Evaluation: U-Net (180/25ep, corrected split + LR + clip)")
    println("═" ^ 72)

    # ── Step 1: COMMEMI MT → prior .ini ──────────────────────────────────────
    println("\n[1/4] Generating COMMEMI prior from mid_scale_prior_v2.jld2...")
    _, _, _, mp = load_trained_model(checkpoint)
    mt_data = commemi_obs_to_mt_tensor(vfsa_data, mp)
    mt_h5 = write_mt_h5(joinpath(results_dir, "commemi_mt.h5"), mt_data)
    println("  COMMEMI MT tensor ", size(mt_data), " → $mt_h5")

    prior_ini = joinpath(results_dir, "mid_scale_prior_v2.ini")
    generate_prior(mt_h5, checkpoint, prior_ini;
                   title="U-Net prior v2 (180 samples, 25 ep, best_val, COMMEMI MT)")
    println("  → $prior_ini")

    model, ps, st, mp = load_trained_model(checkpoint)
    pred_logres = predict_prior(model, ps, st, mt_data)
    @printf("  Predicted log10(ρ): size=%s  mean=%.3f  std=%.3f  range=[%.3f, %.3f]\n",
            size(pred_logres), mean(pred_logres), std(pred_logres),
            minimum(pred_logres), maximum(pred_logres))

    # ── Step 2: VFSA2DMT ─────────────────────────────────────────────────────
    println("\n[2/4] Running VFSA2DMT (1 chain, 200 ctrl, 100 iter, seed=20260308)...")
    result = run_vfsa_eval(prior_ini, vfsa_data)
    run_dir = result.run_info.run_dir
    @info "VFSA2DMT complete" dir=run_dir

    v2_init = extract_initial_rms(run_dir)
    v2_best = extract_best_rms(run_dir)

    # ── Step 3: Five-way table + interpretation ──────────────────────────────
    println("\n[3/4] Five-way RMS comparison")
    print_five_way_table(v2_init, v2_best)
    interpretation = interpret_v2(v2_best)
    println()
    println("  Yorum: ", interpretation)
    println()

    table_path = joinpath(results_dir, "comparison.txt")
    open(table_path, "w") do io
        println(io, "Five-way COMMEMI RMS comparison (1 chain, 200 ctrl, 100 iter, seed=20260308)")
        println(io, "v2 checkpoint: models/mid_scale_prior_v2.jld2")
        println(io, "VFSA run_dir:  $run_dir")
        println(io, "")
        @printf(io, "  %-36s  initial=%-12s  best=%-12s\n",
                "Homojen:", fmt(HOMO_INIT), fmt(HOMO_BEST))
        @printf(io, "  %-36s  initial=%-12s  best=%-12s\n",
                "U-Net (20 örnek/10ep):", fmt(UNET20_INIT), fmt(UNET20_BEST))
        @printf(io, "  %-36s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v1):", fmt(V1_INIT), fmt(V1_BEST))
        @printf(io, "  %-36s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v2*):", fmt(v2_init), fmt(v2_best))
        @printf(io, "  %-36s  initial=%-12s  best=%-12s\n",
                "Gerçek model (sanity):", fmt(TRUE_INIT), fmt(TRUE_BEST))
        println(io, "")
        println(io, "*v2 = düzeltilmiş train/val split + LR + grad clipping + best_val checkpoint")
        println(io, "Yorum: ", interpretation)
    end
    println("  → $table_path")

    # ── Step 4: Predicted vs true heatmap ────────────────────────────────────
    println("\n[4/4] Plotting predicted prior vs resampled COMMEMI true model...")
    truth = load_resampled_commemi_true(true_model, mp)
    @printf("  True (resampled) log10(ρ): size=%s  mean=%.3f  std=%.3f  range=[%.3f, %.3f]\n",
            size(truth), mean(truth), std(truth), minimum(truth), maximum(truth))
    rmse = sqrt(mean((Float64.(pred_logres) .- Float64.(truth)) .^ 2))
    @printf("  Grid RMSE (pred vs true, log10 Ω·m): %.4f\n", rmse)

    png_path = joinpath(results_dir, "predicted_vs_true.png")
    plot_predicted_vs_true(pred_logres, truth, png_path)
    println("  → $png_path")

    println("\n", "═" ^ 72)
    println(" Done. Results in: $results_dir")
    println("═" ^ 72)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
