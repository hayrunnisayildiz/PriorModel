#!/usr/bin/env julia
#=
Mid-scale evaluation: train U-Net with 180 samples / 25 epochs,
generate COMMEMI prior, run VFSA2DMT, print 4-way RMS table,
and plot training loss curve.

Usage (from project root):
    julia --project=. scripts/evaluate_mid_scale.jl

Colab (cwd is /content — never `--project=.` from there):
    julia --project=/content/PriorModel /content/PriorModel/scripts/evaluate_mid_scale.jl
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf, Dates, Plots, DelimitedFiles, MTGeophysics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics: generate_prior, write_ini_prior, load_trained_model
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams: DEFAULT_MESH

# ─────────────────────────────────────────────────────────────────────────────
# RMS extraction (reused from sanity_check_mesh_transfer.jl)
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
# VFSA runner (identical settings to sanity_check for comparability)
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
# Training loss curve plot
# ─────────────────────────────────────────────────────────────────────────────

function plot_training_curve(csv_path::String, out_png::String)
    data = readdlm(csv_path, ','; header=true)
    vals = data[1]
    epochs     = Int.(vals[:, 1])
    train_loss = Float64.(vals[:, 2])
    val_loss   = Float64.(vals[:, 3])

    p = Plots.plot(epochs, train_loss;
        label="train_loss", lw=2, marker=:circle, ms=3,
        xlabel="Epoch", ylabel="Loss",
        title="Training Curve (180 samples, 80/20 split, 25 epochs)",
        legend=:topright,
        size=(700, 400), dpi=150)
    Plots.plot!(p, epochs, val_loss;
        label="val_loss", lw=2, marker=:square, ms=3, ls=:dash)
    if size(vals, 2) >= 4
        flags = Int.(vals[:, 4])
        best_i = findfirst(==(1), flags)
        if best_i !== nothing
            be = epochs[best_i]
            Plots.vline!(p, [be]; color=:gray, ls=:dot, lw=1.2,
                         label="best val (epoch $be)")
            Plots.scatter!(p, [be], [val_loss[best_i]];
                           marker=:star, ms=10, color=:orange, label="")
        end
    end
    Plots.savefig(p, out_png)
    @info "Training curve saved" path=out_png
    return out_png
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    checkpoint   = joinpath(ROOT, "models", "mid_scale_prior.jld2")
    vfsa_data    = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    csv_path     = joinpath(ROOT, "results", "training_log.csv")
    curve_png    = joinpath(ROOT, "results", "training_curve.png")
    results_dir  = joinpath(ROOT, "results")
    mkpath(results_dir)

    println("═" ^ 65)
    println(" Mid-scale Evaluation: U-Net (180 samples / 25 epochs)")
    println("═" ^ 65)

    # ── Step 1: Generate prior .ini from COMMEMI .obs (standardized internally)
    println("\n[1/3] Generating COMMEMI prior from mid-scale checkpoint...")
    prior_ini = joinpath(results_dir, "mid_scale_prior.ini")
    generate_prior(vfsa_data, checkpoint, prior_ini;
                   title="U-Net prior (180 samples, 25 epochs)")
    println("  → $prior_ini")

    # ── Step 2: Run VFSA2DMT ─────────────────────────────────────────────────
    println("\n[2/3] Running VFSA2DMT (1 chain, 200 ctrl, 100 iter, seed=20260308)...")
    result = run_vfsa_eval(prior_ini, vfsa_data)
    run_dir = result.run_info.run_dir
    @info "VFSA2DMT complete" dir=run_dir

    mid_init = extract_initial_rms(run_dir)
    mid_best = extract_best_rms(run_dir)

    # ── Step 3: Four-way RMS table ───────────────────────────────────────────
    println("\n[3/3] Four-way RMS comparison")
    println("─" ^ 65)

    fmt(x) = x === nothing ? "N/A" : @sprintf("%.4f", x)

    @printf("  %-35s  initial=%-12s  best=%-12s\n",
            "Homojen:", fmt(12.31), fmt(5.7354))
    @printf("  %-35s  initial=%-12s  best=%-12s\n",
            "U-Net (20 örnek/10ep):", fmt(61.50), fmt(8.3573))
    @printf("  %-35s  initial=%-12s  best=%-12s\n",
            "U-Net (180 örnek/25ep):", fmt(mid_init), fmt(mid_best))
    @printf("  %-35s  initial=%-12s  best=%-12s\n",
            "Gerçek model (sanity):", fmt(1.30), fmt(1.3002))
    println("─" ^ 65)

    # ── Step 4: Plot training curve ──────────────────────────────────────────
    if isfile(csv_path)
        println("\nPlotting training curve...")
        plot_training_curve(csv_path, curve_png)
        println("  → $curve_png")
    else
        @warn "Training log not found" path=csv_path
    end

    println("\n", "═" ^ 65)
    println(" Done.")
    println("═" ^ 65)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
