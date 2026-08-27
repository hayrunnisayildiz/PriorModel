#!/usr/bin/env julia
#=
COMMEMI 2-D-I evaluation harness (same VFSA settings as v4–v8).

Default checkpoint is v8 TE+TM (n=200, 15 ep, dx=160). Pass any later
checkpoint as ARGS[1]; table labels, footnotes, and the prior `.ini`
name are taken from the checkpoint stem so a v9 run is not reported as v8.

    Usage (from project root):
    julia --project=. scripts/evaluate_mid_scale_v8.jl
    julia --project=. scripts/evaluate_mid_scale_v8.jl \
        models/prior_v8_tetm_n200_dx160.jld2 results/evaluate_v8_tetm_dx160
    julia --project=. scripts/evaluate_mid_scale_v8.jl \
        models/prior_v9_deepbody_n1000.jld2 results/evaluate_v9_deepbody_n1000
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf, Dates, Statistics, HDF5, Plots, MTGeophysics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics:
    generate_prior, write_ini_prior, load_trained_model, predict_prior,
    load_mt_observations, standardize_mt_input, pack_te_response, pack_tetm_response
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams:
    MeshParams, station_positions
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers:
    count_parameters, report_capacity_change

# Historical RMS from previous identical-setting VFSA2DMT runs
const HOMO_INIT   = 12.31
const HOMO_BEST   = 5.7354
const UNET20_INIT = 61.50
const UNET20_BEST = 8.3573
const V1_INIT     = 54.39
const V1_BEST     = 9.9083
const V2_INIT     = 7.9178
const V2_BEST     = 4.8981
const V3_INIT     = 9.1006
const V3_BEST     = 4.9095
const V4_INIT     = 62.65
const V4_BEST     = 18.8611
const V5_INIT     = 39.8638
const V5_BEST     = 11.8539
const V6_INIT     = 37.4294
const V6_BEST     = 13.3179
const V7_INIT     = 38.56
const V7_BEST     = 12.4032
const TRUE_INIT   = 1.30
const TRUE_BEST   = 1.3002
const V7_GRID_RMSE = 1.0633          # pred-vs-true log10 Ω·m on the v7 (dx=160) grid
const RMSE_CONTINUE = 0.7            # v8 grid RMSE below this → keep refining dx
const V4_IMPROVE_REL = 0.15          # relative drop vs previous U-Net to count as "belirgin"

# COMMEMI .obs → U-Net tensor. n_channels=4 uses pack_tetm_response (training --tetm).
function commemi_obs_to_mt_tensor(obs_path::String, mp::MeshParams; n_channels::Int=4)
    isfile(obs_path) || error("COMMEMI obs not found: $obs_path")
    data = MTGeophysics.load_data2d(obs_path)
    src_y = Float64.(data.receivers)
    src_T = Float64.(data.periods)
    raw = if n_channels == 4
        pack_tetm_response(data.rho_xy, data.phase_xy, data.rho_yx, data.phase_yx)
    elseif n_channels == 2
        pack_te_response(data.rho_xy, data.phase_xy)
    else
        error("n_channels=$n_channels; expected 2 or 4")
    end
    out = standardize_mt_input(src_y, src_T, raw; mp=mp, method=:bilinear)
    size(out, 3) == n_channels ||
        error("COMMEMI tensor $(size(out)) ≠ C=$n_channels")
    nS, nP = size(out, 1), size(out, 2)
    @printf("  COMMEMI survey: %d stations  y∈[%.0f, %.0f] m,  %d periods T∈[%.4g, %.4g] s\n",
            length(src_y), minimum(src_y), maximum(src_y),
            length(src_T), minimum(src_T), maximum(src_T))
    @printf("  U-Net survey:   %d stations  %d periods T∈[%.4g, %.4g] s  tensor=%s\n",
            nS, nP, minimum(mp.periods), maximum(mp.periods), size(out))
    T_ood = minimum(src_T) < minimum(mp.periods) - 1e-12 ||
            maximum(src_T) > maximum(mp.periods) + 1e-12
    println("  T_ood (COMMEMI ⊂ eğitim periyotları): ", T_ood)
    T_ood && @warn "COMMEMI periods still fall outside the U-Net training band"
    return out, T_ood
end

function write_mt_h5(path::String, mt_data::Array{Float32,3})
    mkpath(dirname(abspath(path)))
    h5open(path, "w") do f
        f["X"] = mt_data
        f["mt_data"] = mt_data
        a = HDF5.attributes(f)
        a["source"] = "COMMEMI Comemi2D1.obs (interpolated to U-Net survey)"
        a["schema"] = size(mt_data, 3) == 4 ? "commemi_mt/v2-tetm" : "commemi_mt/v2"
        a["n_components"] = size(mt_data, 3)
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

"""Stem of a checkpoint path, e.g. `prior_v9_deepbody_n1000`."""
function checkpoint_tag(path::String)
    return replace(basename(path), r"\.(jld2|JLD2)$" => "")
end

function short_version(tag::String)
    m = match(r"v\d+", tag)
    return m === nothing ? tag : String(m.match)
end

function table_row_label(tag::String)
    occursin("v9_deepbody", tag) && return "U-Net (1000/50ep, v9 deepbody TE+TM):"
    occursin("v8_tetm", tag)     && return "U-Net (200/15ep, v8 TE+TM, dx=160):"
    return "U-Net ($tag):"
end

function data_footnote(tag::String)
    occursin("v9_deepbody", tag) &&
        return "v9 data: train_pairs_deepbody_n1000.h5, 50 epochs, --commemi-every 5, --commemi-iter 25, TE+TM 4-channel"
    occursin("v8_tetm", tag) &&
        return "v8 data: n=200, 15 epochs, --commemi-every 5, TE+TM 4-channel"
    return "checkpoint: $tag"
end

function training_curve_relpath(tag::String)
    occursin("v9_deepbody", tag) && return "results/training_curve_v9_deepbody_n1000.png"
    occursin("v8_tetm", tag)     && return "results/training_curve_v8_tetm_n200_dx160.png"
    return "results/training_curve_$(tag).png"
end

"""Go/stop on pred-vs-true grid RMSE (primary) plus VFSA best RMS (context)."""
function interpret_run(best_rms, grid_rmse, tag::String)
    ver = short_version(tag)
    if grid_rmse === nothing
        return "$ver grid RMSE hesaplanamadı — DUR, çıktıyı kontrol edin."
    end
    delta = grid_rmse - V7_GRID_RMSE
    if grid_rmse < RMSE_CONTINUE
        msg = "DEVAM ET: $ver grid RMSE=$(round(grid_rmse; digits=4)) < $(RMSE_CONTINUE) " *
              "(v7=$(V7_GRID_RMSE), Δ=$(round(delta; digits=4)))."
        occursin("v8_tetm", tag) &&
            (msg *= " TE+TM n=200, dx=160 grid RMSE eşiğin altında — n=1000 ayrı bir karar.")
    else
        msg = "DUR: $ver grid RMSE=$(round(grid_rmse; digits=4)) v7 ($(V7_GRID_RMSE)) ile " *
              "aynı mertebede (Δ=$(round(delta; digits=4)), eşik <$(RMSE_CONTINUE))."
        if occursin("v8_tetm", tag)
            msg *= " Çözünürlük tek başına açıklamıyor — n=1000'e çıkmadan pred-vs-true " *
                   "hata haritasını görselleştir (mimari / kayıp / forward-eğitim hatası)."
        else
            msg *= " Pred-vs-true hata haritasını görselleştir (mimari / kayıp / forward-eğitim hatası)."
        end
    end
    if best_rms !== nothing
        msg *= " VFSA best=$(round(best_rms; digits=4)) vs v7=$(V7_BEST) vs homojen=$(HOMO_BEST)."
    end
    return msg
end

function print_eleven_way_table(run_init, run_best, tag::String)
    ver = short_version(tag)
    row = table_row_label(tag)
    println("─" ^ 88)
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "Homojen:", fmt(HOMO_INIT), fmt(HOMO_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (20/10ep):", fmt(UNET20_INIT), fmt(UNET20_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v1):", fmt(V1_INIT), fmt(V1_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v2, dar):", fmt(V2_INIT), fmt(V2_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v3, gT):", fmt(V3_INIT), fmt(V3_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (180/25ep, v4, gT+x):", fmt(V4_INIT), fmt(V4_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (400/35ep, v5, gT+x, büyük kapasite):", fmt(V5_INIT), fmt(V5_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (400/60ep, v6, v5 devam):", fmt(V6_INIT), fmt(V6_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "U-Net (1000/50ep, v7, senaryo+commemi_rms):", fmt(V7_INIT), fmt(V7_BEST))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            row, fmt(run_init), fmt(run_best))
    @printf("  %-58s  initial=%-12s  best=%-12s\n",
            "Gerçek model (sanity):", fmt(TRUE_INIT), fmt(TRUE_BEST))
    println("─" ^ 88)
    println("  v4+ mesh: T∈[1e-3,1e3] s, dx=160 m, nx=120 (same UNET_MESH)")
    if run_best !== nothing
        println()
        @printf("  %s vs homojen (5.7354):  Δbest=%+.4f   %s\n",
                ver, run_best - HOMO_BEST,
                run_best < HOMO_BEST ? "HOMOJENİ GEÇTİ" : "homojeni GEÇMEDİ")
        @printf("  %s vs v7     (12.4032):  Δbest=%+.4f   %s\n",
                ver, run_best - V7_BEST,
                run_best < V7_BEST ? "v7'den iyi" : "v7'den kötü/eşit")
        @printf("  %s vs v5     (11.8539):  Δbest=%+.4f   %s\n",
                ver, run_best - V5_BEST,
                run_best < V5_BEST ? "v5'ten iyi" : "v5'ten kötü/eşit")
    end
end

function plot_predicted_vs_true(pred::Matrix{Float32}, truth::Matrix{Float32},
                                out_png::String; pred_title::String="U-Net prior")
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
        title=pred_title,
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
    checkpoint   = length(ARGS) >= 1 ? abspath(ARGS[1]) :
                   joinpath(ROOT, "models", "prior_v8_tetm_n200_dx160.jld2")
    results_dir  = length(ARGS) >= 2 ? abspath(ARGS[2]) :
                   joinpath(ROOT, "results", "evaluate_v8_tetm_dx160")
    tag          = checkpoint_tag(checkpoint)
    ver          = short_version(tag)
    vfsa_data    = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    true_model   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
    mkpath(results_dir)

    isfile(checkpoint) || error("checkpoint not found: $checkpoint")
    isfile(vfsa_data)  || error("COMMEMI obs not found: $vfsa_data")
    isfile(true_model) || error("COMMEMI true model not found: $true_model")

    println("═" ^ 88)
    println(" $ver COMMEMI eval: $tag")
    println("═" ^ 88)
    println("  checkpoint: ", checkpoint)
    println("  results:    ", results_dir)
    println("  ", data_footnote(tag))

    # ── Step 1: COMMEMI MT → prior .ini ──────────────────────────────────────
    println("\n[1/4] Generating COMMEMI prior from ", basename(checkpoint), "...")
    model, ps, st, mp = load_trained_model(checkpoint)
    n_params = count_parameters(ps)
    cap = report_capacity_change(mp)
    n_ch = Int(model.in_channels)
    println("  model: ", model)
    @printf("  mesh: nz=%d nx=%d dx=%.1f m dz=%.1f m  C_in=%d\n",
            mp.nz, mp.nx, mp.dx, mp.dz, n_ch)
    @printf("  params this ckpt: %d    v4 layout: %d    v5 layout: %d  (×%.2f)\n",
            n_params, cap.n_old, cap.n_new, cap.ratio)

    mt_data, T_ood = commemi_obs_to_mt_tensor(vfsa_data, mp; n_channels=n_ch)
    xs = station_positions(mp)
    data_obs = MTGeophysics.load_data2d(vfsa_data)
    src_x = Float64.(data_obs.receivers)
    x_ood = count(x -> x < first(xs) - 1e-9 || x > last(xs) + 1e-9, src_x)
    @printf("  x_ood: %d/%d  canonical x∈[%.1f, %.1f] m\n",
            x_ood, length(src_x), first(xs), last(xs))

    mt_h5 = write_mt_h5(joinpath(results_dir, "commemi_mt.h5"), mt_data)
    println("  COMMEMI MT tensor ", size(mt_data), " → $mt_h5")

    prior_ini = joinpath(results_dir, "$(tag).ini")
    generate_prior(mt_h5, checkpoint, prior_ini;
                   title="U-Net $ver prior ($tag)")
    println("  → $prior_ini")

    pred_logres = predict_prior(model, ps, st, mt_data)
    @printf("  Predicted log10(ρ): size=%s  mean=%.3f  std=%.3f  range=[%.3f, %.3f]\n",
            size(pred_logres), mean(pred_logres), std(pred_logres),
            minimum(pred_logres), maximum(pred_logres))

    # Grid RMSE before VFSA so a failed inversion still yields the go/stop metric
    println("\n[1b] Pred-vs-true grid RMSE (computed before VFSA)...")
    truth = load_resampled_commemi_true(true_model, mp)
    @printf("  True (resampled) log10(ρ): size=%s  mean=%.3f  std=%.3f  range=[%.3f, %.3f]\n",
            size(truth), mean(truth), std(truth), minimum(truth), maximum(truth))
    rmse = sqrt(mean((Float64.(pred_logres) .- Float64.(truth)) .^ 2))
    @printf("  Grid RMSE (pred vs true, log10 Ω·m): %.4f   (v7=%.4f, Δ=%+.4f)\n",
            rmse, V7_GRID_RMSE, rmse - V7_GRID_RMSE)

    # ── Step 2: VFSA2DMT ─────────────────────────────────────────────────────
    println("\n[2/4] Running VFSA2DMT (1 chain, 200 ctrl, 100 iter, seed=20260308)...")
    result = run_vfsa_eval(prior_ini, vfsa_data)
    run_dir = result.run_info.run_dir
    @info "VFSA2DMT complete" dir=run_dir

    run_init = extract_initial_rms(run_dir)
    run_best = extract_best_rms(run_dir)

    # ── Step 3: Eleven-way table + go/stop ───────────────────────────────────
    println("\n[3/4] Eleven-way RMS comparison")
    print_eleven_way_table(run_init, run_best, tag)
    interpretation = interpret_run(run_best, rmse, tag)
    println()
    println("  T_ood: ", T_ood, T_ood ? "  (COMMEMI hâlâ eğitim bandı dışında)" :
                                        "  (COMMEMI eğitim bandının içinde)")
    println("  x_ood: ", x_ood, "/", length(src_x),
            x_ood == 0 ? "  (COMMEMI eğitim x-bandının içinde)" :
                         "  (istasyon clamp)")
    @printf("  Grid RMSE %s=%.4f  v7=%.4f  Δ=%+.4f  (continue if %s < %.2f)\n",
            ver, rmse, V7_GRID_RMSE, rmse - V7_GRID_RMSE, ver, RMSE_CONTINUE)
    println("  Yorum: ", interpretation)
    println()

    table_path = joinpath(results_dir, "comparison.txt")
    row = table_row_label(tag)
    open(table_path, "w") do io
        println(io, "Eleven-way COMMEMI RMS comparison (1 chain, 200 ctrl, 100 iter, seed=20260308)")
        println(io, "checkpoint: $checkpoint")
        println(io, "VFSA run_dir:  $run_dir")
        println(io, "T_ood (COMMEMI ⊂ eğitim T): $T_ood")
        println(io, "x_ood (COMMEMI ⊂ eğitim x): $x_ood/$(length(src_x))")
        @printf(io, "params: this=%d  v4_layout=%d  v5_layout=%d  (×%.2f)\n",
                n_params, cap.n_old, cap.n_new, cap.ratio)
        @printf(io, "mesh: nz=%d nx=%d dx=%.1f m dz=%.1f m\n", mp.nz, mp.nx, mp.dx, mp.dz)
        println(io, "")
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "Homojen:", fmt(HOMO_INIT), fmt(HOMO_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (20/10ep):", fmt(UNET20_INIT), fmt(UNET20_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v1):", fmt(V1_INIT), fmt(V1_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v2, dar):", fmt(V2_INIT), fmt(V2_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v3, gT):", fmt(V3_INIT), fmt(V3_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (180/25ep, v4, gT+x):", fmt(V4_INIT), fmt(V4_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (400/35ep, v5, gT+x, büyük kapasite):", fmt(V5_INIT), fmt(V5_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (400/60ep, v6, v5 devam):", fmt(V6_INIT), fmt(V6_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "U-Net (1000/50ep, v7, senaryo+commemi_rms):", fmt(V7_INIT), fmt(V7_BEST))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                row, fmt(run_init), fmt(run_best))
        @printf(io, "  %-58s  initial=%-12s  best=%-12s\n",
                "Gerçek model (sanity):", fmt(TRUE_INIT), fmt(TRUE_BEST))
        println(io, "")
        @printf(io, "Grid RMSE (pred vs true, log10 Ω·m): %s=%.4f  v7=%.4f  Δ=%+.4f\n",
                ver, rmse, V7_GRID_RMSE, rmse - V7_GRID_RMSE)
        println(io, "")
        if run_best !== nothing
            @printf(io, "%s vs homojen (5.7354):  Δbest=%+.4f   %s\n",
                    ver, run_best - HOMO_BEST,
                    run_best < HOMO_BEST ? "HOMOJENİ GEÇTİ" : "homojeni GEÇMEDİ")
            @printf(io, "%s vs v7     (12.4032):  Δbest=%+.4f   %s\n",
                    ver, run_best - V7_BEST,
                    run_best < V7_BEST ? "v7'den iyi" : "v7'den kötü/eşit")
            @printf(io, "%s vs v5     (11.8539):  Δbest=%+.4f   %s\n",
                    ver, run_best - V5_BEST,
                    run_best < V5_BEST ? "v5'ten iyi" : "v5'ten kötü/eşit")
            println(io, "")
        end
        println(io, "v4+ mesh: T∈[1e-3,1e3] s, dx=160 m, nx=120 (same UNET_MESH)")
        println(io, data_footnote(tag))
        println(io, "Yorum: ", interpretation)
        println(io, "")
        println(io, "See also: results/KNOWN_LIMITATIONS.md")
        println(io, "Training curve: ", training_curve_relpath(tag))
    end
    println("  → $table_path")

    # ── Step 4: Predicted vs true heatmap ────────────────────────────────────
    println("\n[4/4] Plotting predicted prior vs resampled COMMEMI true model...")
    png_path = joinpath(results_dir, "predicted_vs_true.png")
    plot_predicted_vs_true(pred_logres, truth, png_path;
                           pred_title="U-Net $ver prior ($tag)")
    println("  → $png_path")

    println("\n", "═" ^ 72)
    println(" Decision: ", startswith(interpretation, "DEVAM ET") ? "DEVAM ET" : "DUR")
    println(" Done. Results in: $results_dir")
    println("═" ^ 72)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
