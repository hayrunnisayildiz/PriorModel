#!/usr/bin/env julia
#=
End-to-end example: neural MT prior → VFSA2DMT inversion → comparison plot.

Usage (from project root):
    julia --project=. examples/run_mt_prior_inversion.jl \
        --mt-obs   obs/site.h5 \
        --ckpt     models/best_mt_resistivity_prior.jld2 \
        --data     examples/0COMEMI2D-I/Comemi2D1.obs \
        --output   results/prior_inversion

Colab (cwd is /content — never `--project=.` from there):
    julia --project=/content/PriorModel \
        /content/PriorModel/examples/run_mt_prior_inversion.jl

Steps:
  1. Load MT observations, predict neural prior via MTResistivityUNet2D
  2. Write prior as .ini start-model for MTGeophysics.jl
  3. Run VFSA2DMT with the neural prior
  4. Run VFSA2DMT with a homogeneous halfspace (baseline)
  5. Plot side-by-side comparison
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Dates

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics: generate_prior, load_trained_model

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct RunConfig
    mt_obs_path::String      = joinpath(ROOT, "data", "synthetic", "train_pairs.h5")
    checkpoint_path::String  = joinpath(ROOT, "models", "best_mt_resistivity_prior.jld2")
    data_path::String        = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    output_dir::String       = joinpath(ROOT, "results", "prior_inversion")
    n_chains::Int            = 2
    max_iter::Int            = 3000
    n_trials::Int            = 1
    seed::Int                = 20260308
end

function parse_args(args::Vector{String})::RunConfig
    cfg = RunConfig()
    mt_obs = cfg.mt_obs_path
    ckpt   = cfg.checkpoint_path
    data   = cfg.data_path
    outdir = cfg.output_dir
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--mt-obs" && i + 1 <= length(args)
            mt_obs = args[i+1]; i += 2; continue
        elseif a == "--ckpt" && i + 1 <= length(args)
            ckpt = args[i+1]; i += 2; continue
        elseif a == "--data" && i + 1 <= length(args)
            data = args[i+1]; i += 2; continue
        elseif a == "--output" && i + 1 <= length(args)
            outdir = args[i+1]; i += 2; continue
        end
        i += 1
    end
    return RunConfig(; mt_obs_path=mt_obs, checkpoint_path=ckpt,
                     data_path=data, output_dir=outdir)
end

# ─────────────────────────────────────────────────────────────────────────────
# Inversion helpers
# ─────────────────────────────────────────────────────────────────────────────

function run_vfsa(; start_model_path::String, data_path::String,
                  script_path::String, cfg::RunConfig)
    @eval using MTGeophysics

    result = MTGeophysics.VFSA2DMT(
        MTGeophysics.VFSA2DMTParams(
            script_path      = script_path,
            start_model_path = start_model_path,
            data_path        = data_path,
            config = MTGeophysics.VFSA2DMTConfig(
                n_chains    = cfg.n_chains,
                n_ctrl      = 400,
                max_iter    = cfg.max_iter,
                n_trials    = cfg.n_trials,
                log_bounds  = (0.0, 4.0),
                seed        = cfg.seed,
                keep_models = true,
            ),
        ),
    )
    return result
end

"""Write a homogeneous halfspace .ini using the same mesh as the neural prior."""
function write_halfspace_ini(output_path::String, checkpoint_path::String;
                             background_resistivity::Float64=100.0)
    _, _, _, mp = load_trained_model(checkpoint_path)
    logres_grid = fill(Float32(log10(background_resistivity)), mp.nz, mp.nx)

    # Reuse the export module's writer
    ExportPriorToMTGeophysics.write_ini_prior(
        logres_grid, output_path, mp;
        title="Homogeneous halfspace $(background_resistivity) ohm-m",
    )
    return output_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Extract final misfit, iteration count, and per-iteration misfit curve from a VFSA result."""
function _extract_metrics(result)
    ri = result.run_info

    final_misfit = if hasproperty(ri, :best_rms)
        ri.best_rms
    elseif hasproperty(ri, :best_misfit)
        ri.best_misfit
    elseif hasproperty(result, :summary) && hasproperty(result.summary, :best_rms)
        result.summary.best_rms
    else
        nothing
    end

    n_iter = if hasproperty(ri, :iterations_used)
        ri.iterations_used
    elseif hasproperty(ri, :n_iterations)
        ri.n_iterations
    elseif hasproperty(ri, :total_iterations)
        ri.total_iterations
    else
        nothing
    end

    misfit_curve = if hasproperty(ri, :misfit_history)
        Float64.(ri.misfit_history)
    elseif hasproperty(ri, :rms_history)
        Float64.(ri.rms_history)
    elseif hasproperty(result, :chain_results)
        # Take the best chain's history
        best = argmin(cr -> something(
            hasproperty(cr, :best_rms) ? cr.best_rms : nothing,
            hasproperty(cr, :best_misfit) ? cr.best_misfit : nothing,
            Inf), result.chain_results)
        if hasproperty(best, :rms_history)
            Float64.(best.rms_history)
        elseif hasproperty(best, :misfit_history)
            Float64.(best.misfit_history)
        else
            nothing
        end
    else
        nothing
    end

    return (; final_misfit, n_iter, misfit_curve)
end

function print_inversion_summary(label::String, result)
    m = _extract_metrics(result)
    misfit_str = m.final_misfit === nothing ? "N/A" : @sprintf("%.4f", m.final_misfit)
    iter_str   = m.n_iter === nothing ? "N/A" : string(m.n_iter)
    @printf("  %-14s  misfit = %s  iterations = %s\n", label * ":", misfit_str, iter_str)
end

"""Compute how many times faster the prior reaches the halfspace's final misfit."""
function compute_speedup(halfspace_result, prior_result)
    mh = _extract_metrics(halfspace_result)
    mp = _extract_metrics(prior_result)

    mh.final_misfit === nothing && return nothing
    mp.misfit_curve === nothing && return nothing
    mh.n_iter === nothing && return nothing

    target = mh.final_misfit
    idx = findfirst(v -> v <= target, mp.misfit_curve)
    idx === nothing && return nothing

    return Float64(mh.n_iter) / Float64(idx)
end

# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

function plot_comparison(prior_result, halfspace_result, output_dir::String)
    try
        @eval using CairoMakie
    catch
        @warn "CairoMakie not available; skipping comparison plot"
        return nothing
    end

    fig = CairoMakie.Figure(size=(1200, 500))

    for (col, (result, label)) in enumerate([(prior_result, "Neural Prior"),
                                              (halfspace_result, "Halfspace")])
        ax = CairoMakie.Axis(fig[1, col];
            title=label,
            xlabel="Profile (m)",
            ylabel="Depth (m)",
            yreversed=true,
        )

        if hasproperty(result, :best_model) && result.best_model !== nothing
            model = result.best_model
            nz, ny = size(model)
            CairoMakie.heatmap!(ax, 1:ny, 1:nz, log10.(model');
                colormap=:turbo,
                colorrange=(0, 4),
            )
        end
    end

    CairoMakie.Colorbar(fig[1, 3]; colormap=:turbo, colorrange=(0, 4),
        label="log₁₀(ρ) [Ω·m]")

    plot_path = joinpath(output_dir, "comparison_prior_vs_halfspace.png")
    CairoMakie.save(plot_path, fig; px_per_unit=2)
    @info "Comparison plot saved" path=plot_path
    return plot_path
end

"""Plot misfit convergence curves for both runs overlaid."""
function plot_convergence(prior_result, halfspace_result, output_dir::String)
    try
        @eval using CairoMakie
    catch
        @warn "CairoMakie not available; skipping convergence plot"
        return nothing
    end

    mp = _extract_metrics(prior_result)
    mh = _extract_metrics(halfspace_result)
    (mp.misfit_curve === nothing && mh.misfit_curve === nothing) && return nothing

    fig = CairoMakie.Figure(size=(700, 400))
    ax = CairoMakie.Axis(fig[1, 1];
        xlabel="Iteration", ylabel="Misfit (RMS)",
        title="Convergence: Neural Prior vs Halfspace",
        yscale=log10,
    )

    if mh.misfit_curve !== nothing
        CairoMakie.lines!(ax, 1:length(mh.misfit_curve), mh.misfit_curve;
            color=:gray60, linewidth=2, label="Halfspace")
    end
    if mp.misfit_curve !== nothing
        CairoMakie.lines!(ax, 1:length(mp.misfit_curve), mp.misfit_curve;
            color=:dodgerblue, linewidth=2, label="Neural Prior")
    end

    CairoMakie.axislegend(ax; position=:rt)

    plot_path = joinpath(output_dir, "convergence_prior_vs_halfspace.png")
    CairoMakie.save(plot_path, fig; px_per_unit=2)
    @info "Convergence plot saved" path=plot_path
    return plot_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(cfg::RunConfig=RunConfig())
    mkpath(cfg.output_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    println("═" ^ 60)
    println(" MT Prior Inversion — Neural vs Halfspace")
    println("═" ^ 60)

    # Step 1: Generate neural prior .ini
    prior_ini = joinpath(cfg.output_dir, "neural_prior_$(timestamp).ini")
    println("\n[1/4] Generating neural prior start model...")
    generate_prior(cfg.mt_obs_path, cfg.checkpoint_path, prior_ini)
    println("  → $prior_ini")

    # Step 2: Generate halfspace .ini
    halfspace_ini = joinpath(cfg.output_dir, "halfspace_$(timestamp).ini")
    println("\n[2/4] Generating halfspace start model...")
    write_halfspace_ini(halfspace_ini, cfg.checkpoint_path)
    println("  → $halfspace_ini")

    # Step 3: Run VFSA with neural prior
    println("\n[3/4] Running VFSA2DMT with neural prior...")
    prior_result = run_vfsa(;
        start_model_path=prior_ini,
        data_path=cfg.data_path,
        script_path=@__FILE__,
        cfg=cfg,
    )
    @info "Neural prior inversion complete" dir=prior_result.run_info.run_dir

    # Step 4: Run VFSA with halfspace
    println("\n[4/4] Running VFSA2DMT with halfspace start...")
    halfspace_result = run_vfsa(;
        start_model_path=halfspace_ini,
        data_path=cfg.data_path,
        script_path=@__FILE__,
        cfg=cfg,
    )
    @info "Halfspace inversion complete" dir=halfspace_result.run_info.run_dir

    # Step 5: Misfit / iteration comparison
    println("\n", "─" ^ 60)
    println(" Misfit / Iteration Comparison")
    println("─" ^ 60)
    print_inversion_summary("Halfspace", halfspace_result)
    print_inversion_summary("Neural Prior", prior_result)
    println("─" ^ 60)

    speedup = compute_speedup(halfspace_result, prior_result)
    if speedup !== nothing
        @printf("Prior reached halfspace-equivalent misfit %.1f× faster\n", speedup)
    end

    # Step 6: Comparison plot
    println("\nGenerating comparison plot...")
    plot_comparison(prior_result, halfspace_result, cfg.output_dir)

    # Step 7: Convergence curve overlay
    println("Generating convergence plot...")
    plot_convergence(prior_result, halfspace_result, cfg.output_dir)

    println("\n", "═" ^ 60)
    println(" Done. Results in: $(cfg.output_dir)")
    println("═" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_args(ARGS))
end
