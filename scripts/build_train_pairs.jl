#!/usr/bin/env julia
#=
Generate train_pairs.h5: (X, Y) pairs where
  X = MT forward response  (n_stations, n_periods, n_comp, N)
      default n_comp=2 TE: [log10_ρ, phase_deg]
      --tetm    n_comp=4:  [log10_ρ_TE, phase_TE, log10_ρ_TM, phase_TM]
  Y = log10 resistivity     (nz, nx, N)

Default remains TE-only so regenerating v5/v7-style files cannot silently
become 4-channel. Pass `--tetm` (and a separate `--out` path) for TE+TM.

This runs the full pipeline:
  1. Generate synthetic resistivity models (SyntheticGenerator)
  2. Forward-model each through MTGeophysics.jl TE/TM solver
  3. Pack into a single HDF5 with MeshParams attributes

Usage:
    julia --project=. scripts/build_train_pairs.jl --n 50
    julia --project=. scripts/build_train_pairs.jl --n 200 --out data/synthetic/train_pairs.h5
    julia --project=. scripts/build_train_pairs.jl --n 200 --tetm \
        --out data/synthetic/train_pairs_v8_tetm_n200.h5

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
using .MTMeshParams: n_components, MT_DATA_LAYOUT, MT_DATA_LAYOUT_TETM

include(joinpath(ROOT, "src", "synthetic", "MTInputStandardizer.jl"))
using .MTInputStandardizer: pack_te_response, pack_tetm_response

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

function parse_args(argv::Vector{String})
    opts = Dict{Symbol,Any}(
        :n => 50,
        :out => joinpath(ROOT, "data", "synthetic", "train_pairs.h5"),
        :seed => 42,
        :priors => "",
        :tetm => false,
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
        elseif a == "--tetm"
            opts[:tetm] = true
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
    tetm = Bool(opts[:tetm])
    rng = MersenneTwister(seed)
    n_comp = n_components(UNET_MESH; tetm=tetm)
    layout = tetm ? "TE+TM (4 ch)" : "TE-only (2 ch)"

    println("═" ^ 60)
    println(" Building train_pairs.h5 — $n_models model(s)  $layout")
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

    @info "Grid" nz=nzt nx=nyc n_stations=n_st n_periods=n_freq n_components=n_comp tetm=tetm
    @printf("  UNET_MESH: nx=%d  dx=%.1f m  nz=%d  dz=%.1f m  span=%.1f km  (v4–v7 contract: nx=120 dx=160 m)\n",
            UNET_MESH.nx, UNET_MESH.dx, UNET_MESH.nz, UNET_MESH.dz,
            UNET_MESH.nx * UNET_MESH.dx / 1000)
    (UNET_MESH.nx == 120 && UNET_MESH.dx == 160.0) ||
        error("UNET_MESH is not the v4–v7 contract: nx=$(UNET_MESH.nx) dx=$(UNET_MESH.dx) (expected nx=120 dx=160)")
    (nyc == UNET_MESH.nx && Float64(cfg.y_core_cell) == UNET_MESH.dx) ||
        error("generator mesh drifted from UNET_MESH: nx=$nyc dx=$(cfg.y_core_cell)")

    # Allocate output arrays
    X_all = Array{Float32,4}(undef, n_st, n_freq, n_comp, n_models)
    Y_all = Array{Float32,3}(undef, nzt, nyc, n_models)

    t0 = time()
    for i in 1:n_models
        model = generate_model(gmesh, priors, cfg; rng=rng, index=i)
        Y_all[:, :, i] = Float32.(model.log10_rho)

        resp = forward_response(model, gmesh; mode=:TETM)

        packed = if tetm
            # TM phase folded to [0,90] inside pack_tetm_response (solver stores raw atan2).
            pack_tetm_response(resp.rho_xy, resp.phase_xy, resp.rho_yx, resp.phase_yx)
        else
            pack_te_response(resp.rho_xy, resp.phase_xy)
        end
        packed[.!isfinite.(packed)] .= 0.0f0
        size(packed) == (n_st, n_freq, n_comp) ||
            error("packed MT tensor $(size(packed)) != ($n_st, $n_freq, $n_comp)")
        X_all[:, :, :, i] = packed

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
        a["n_components"] = n_comp
        a["tetm"] = tetm ? 1 : 0
        comps = tetm ? MT_DATA_LAYOUT_TETM.components : MT_DATA_LAYOUT.components
        a["components"] = join(comps, ",")
        a["schema"] = tetm ? "train_pairs/v2-tetm" : "train_pairs/v1"
    end

    @info "train_pairs.h5 written" path=out_path n_models size_X=size(X_all) size_Y=size(Y_all)
    println("\nDone: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
