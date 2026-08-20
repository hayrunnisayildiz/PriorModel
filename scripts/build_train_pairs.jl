#!/usr/bin/env julia
#=
Generate train_pairs.h5: (X, Y) pairs where
  X = MT forward response  (n_stations, n_periods, 2, N)   [log10_rho_app, phase_deg]
  Y = log10 resistivity     (nz, nx, N)

This runs the full pipeline:
  1. Generate synthetic resistivity models (SyntheticGenerator)
  2. Forward-model each through MTGeophysics.jl TE/TM solver
  3. Pack into a single HDF5 with MeshParams attributes

Usage:
    julia --project=. scripts/build_train_pairs.jl --n 50
    julia --project=. scripts/build_train_pairs.jl --n 200 --out data/synthetic/train_pairs.h5

Colab (cwd is /content — never `--project=.` from there):
    julia --project=/content/PriorModel /content/PriorModel/scripts/build_train_pairs.jl --n 50
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Random
using Statistics
using HDF5

include(joinpath(ROOT, "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator

include(joinpath(ROOT, "src", "synthetic", "MeshParams.jl"))
using .MTMeshParams: MeshParams, UNET_MESH, n_periods, frequencies_hz

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

function parse_args(argv::Vector{String})
    opts = Dict{Symbol,Any}(
        :n => 50,
        :out => joinpath(ROOT, "data", "synthetic", "train_pairs.h5"),
        :seed => 42,
        :priors => "",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        need() = (i += 1; i <= length(argv) ? argv[i] : error("$(a) needs a value"))
        if a == "--n"
            opts[:n] = parse(Int, need())
        elseif a == "--out"
            opts[:out] = abspath(need())
        elseif a == "--seed"
            opts[:seed] = parse(Int, need())
        elseif a == "--priors"
            opts[:priors] = abspath(need())
        end
        i += 1
    end
    return opts
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(argv::Vector{String}=ARGS)
    opts = parse_args(argv)
    n_models = opts[:n]
    out_path = opts[:out]
    seed = opts[:seed]
    rng = MersenneTwister(seed)

    println("═" ^ 60)
    println(" Building train_pairs.h5 — $n_models model(s)")
    println("═" ^ 60)

    # Build mesh & priors
    priors_path = opts[:priors]
    if isempty(priors_path)
        default_priors = joinpath(ROOT, "config", "keivitsa_priors.json")
        priors_path = isfile(default_priors) ? default_priors : ""
    end

    cfg = GeneratorConfig()
    gmesh = build_generator_mesh(cfg)
    priors = if !isempty(priors_path) && isfile(priors_path)
        load_generator_priors(priors_path)
    else
        @warn "No priors file; using defaults"
        load_generator_priors()
    end

    nzt, nyc = target_size(gmesh)
    n_st = length(gmesh.receiver_positions)
    n_freq = length(gmesh.frequencies)
    n_comp = 2  # log10_rho_app, phase_deg

    @info "Grid" nz=nzt nx=nyc n_stations=n_st n_periods=n_freq

    # Allocate output arrays
    X_all = Array{Float32,4}(undef, n_st, n_freq, n_comp, n_models)
    Y_all = Array{Float32,3}(undef, nzt, nyc, n_models)

    t0 = time()
    for i in 1:n_models
        model = generate_model(gmesh, priors, cfg; rng=rng, index=i)
        Y_all[:, :, i] = Float32.(model.log10_rho)

        resp = forward_response(model, gmesh; mode=:TETM)

        # resp is MT2DResponse: rho_xy (n_freq, n_st), phase_xy (n_freq, n_st)
        # We need (n_stations, n_periods, 2) — transpose freq/station axes
        log10_rho_app = Float32.(log10.(resp.rho_xy'))   # (n_st, n_freq)
        phase_deg     = Float32.(resp.phase_xy')          # (n_st, n_freq)

        # Replace non-finite values
        log10_rho_app[.!isfinite.(log10_rho_app)] .= 0.0f0
        phase_deg[.!isfinite.(phase_deg)] .= 0.0f0

        X_all[:, :, 1, i] = log10_rho_app
        X_all[:, :, 2, i] = phase_deg

        if i % max(1, n_models ÷ 10) == 0 || i == n_models
            elapsed = time() - t0
            rate = i / elapsed
            eta = (n_models - i) / rate
            @printf("  [%d/%d]  %.1f models/s  ETA %.0fs\n", i, n_models, rate, eta)
        end
    end

    # Save HDF5
    mkpath(dirname(out_path))
    h5open(out_path, "w") do f
        f["X"] = X_all
        f["Y"] = Y_all

        # MeshParams attributes (for train_mt_resistivity.jl)
        a = HDF5.attributes(f)
        a["nx"] = nyc
        a["nz"] = nzt
        a["dx"] = Float64(cfg.y_core_cell)
        a["dz"] = Float64(cfg.target_dz)
        a["n_stations"] = n_st
        a["periods"] = Float64.(1.0 ./ gmesh.frequencies)
        a["n_models"] = n_models
        a["seed"] = seed
        a["schema"] = "train_pairs/v1"
    end

    @info "train_pairs.h5 written" path=out_path n_models size_X=size(X_all) size_Y=size(Y_all)
    println("\nDone: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
