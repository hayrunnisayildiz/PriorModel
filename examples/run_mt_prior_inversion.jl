#!/usr/bin/env julia
#=
End-to-end example: neural MT prior → VFSA2DMT inversion → comparison plot.

Usage (from project root):
    julia --project=. examples/run_mt_prior_inversion.jl \
        --mt-obs   obs/site.h5 \
        --ckpt     models/best_mt_resistivity_prior.jld2 \
        --data     examples/0COMEMI2D-I/Comemi2D1.obs \
        --output   results/prior_inversion

Steps:
  1. Load MT observations, predict neural prior via MTResistivityUNet2D
  2. Write prior as .ini start-model for MTGeophysics.jl
  3. Run VFSA2DMT with the neural prior
  4. Run VFSA2DMT with a homogeneous halfspace (baseline)
  5. Plot side-by-side comparison
=#

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

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
# Comparison plot
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

    # Step 5: Comparison plot
    println("\nGenerating comparison plot...")
    plot_comparison(prior_result, halfspace_result, cfg.output_dir)

    println("\n", "═" ^ 60)
    println(" Done. Results in: $(cfg.output_dir)")
    println("═" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_args(ARGS))
end
