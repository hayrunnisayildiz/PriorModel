#!/usr/bin/env julia
#=
Generate a synthetic 2-D MT resistivity training set calibrated by
config/keivitsa_priors.json, on a mesh that MTGeophysics.jl's TE/TM solver accepts.

Usage:
    julia --project=. scripts/build_synthetic_dataset.jl --n 2000
    julia --project=. scripts/build_synthetic_dataset.jl --verify
    julia --project=. scripts/build_synthetic_dataset.jl --n 50 --format jld2

Options:
    --n COUNT           number of models to generate                     [1000]
    --out PATH          output file  [data/processed/synthetic_mt2d.h5]
    --format FMT        hdf5 | jld2                                      [hdf5]
    --priors PATH       prior JSON              [config/keivitsa_priors.json]
    --seed N            base RNG seed                                 [config]
    --scenario NAME     force a single scenario for every model
    --verify            check mesh equality with BuildMesh2D and run the TE/TM
                        forward solver on one model per scenario (needs MTGeophysics)
    --no-generate       with --verify, skip dataset generation
=#

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator

const ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(argv::Vector{String})
    opts = Dict{Symbol,Any}(
        :n => 1000,
        :out => joinpath(ROOT, "data", "processed", "synthetic_mt2d.h5"),
        :format => :hdf5,
        :priors => "",
        :seed => nothing,
        :scenario => nothing,
        :verify => false,
        :generate => true,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        need() = (i += 1; i <= length(argv) ? argv[i] : error("$(a) needs a value"))
        if a == "--n"
            opts[:n] = parse(Int, need())
        elseif a == "--out"
            opts[:out] = abspath(need())
        elseif a == "--format"
            opts[:format] = Symbol(lowercase(need()))
        elseif a == "--priors"
            opts[:priors] = abspath(need())
        elseif a == "--seed"
            opts[:seed] = parse(Int, need())
        elseif a == "--scenario"
            opts[:scenario] = Symbol(need())
        elseif a == "--verify"
            opts[:verify] = true
        elseif a == "--no-generate"
            opts[:generate] = false
        elseif a in ("-h", "--help")
            src = read(@__FILE__, String)
            stop = findfirst("=#", src)
            println(stop === nothing ? src : src[1:first(stop)-1])
            exit(0)
        else
            error("Unknown argument: $(a)")
        end
        i += 1
    end
    return opts
end

"""Report the mesh, then one sample model per scenario with its value range."""
function describe_setup(mesh, priors, cfg)
    nzt, nyc = target_size(mesh)
    println("\n═══ Mesh ═══")
    println("  ", mesh)
    @printf("  solver array (n_z, n_y) = (%d, %d)   air rows = %d\n",
            n_z(mesh), n_y(mesh), mesh.n_air_cells)
    @printf("  target window (n_z, n_y) = (%d, %d)  uniform cells %g × %g m\n",
            nzt, nyc, cfg.target_dz, cfg.y_core_cell)
    zc = z_centers(mesh)[mesh.target_z]
    yc = y_centers(mesh)[mesh.core_y]
    @printf("  target extent: y %.0f…%.0f m,  z %.0f…%.0f m\n",
            yc[1], yc[end], zc[1], zc[end])
    @printf("  stations = %d,  frequencies = %d (%.3g…%.3g Hz)\n",
            length(mesh.receiver_positions), length(mesh.frequencies),
            minimum(mesh.frequencies), maximum(mesh.frequencies))

    scale = SyntheticGenerator.resolve_length_scale(cfg, priors)
    println("\n═══ Priors ═══")
    println("  source: ", priors.source)
    @printf("  length_scale = %.2f  (prior median layer %.1f m → %.0f m ≈ %.1f cells)\n",
            scale, exp(priors.layer_thickness.mu),
            exp(priors.layer_thickness.mu) * scale,
            exp(priors.layer_thickness.mu) * scale / cfg.target_dz)
    for p in priors.populations
        @printf("  %-16s log10 mu=%.2f sigma=%.2f w=%.3f range=[%.2f, %.2f]\n",
                p.name, p.mean, p.std, p.weight, p.lo, p.hi)
    end
    @printf("  aspect(thick/width)=%.3f  conductive_fraction=%.3f  corr z/y = %.1f/%.1f m\n",
            priors.aspect_ratio, priors.conductive_fraction,
            priors.corr_vertical, priors.corr_horizontal)
end

function sample_per_scenario(mesh, priors, cfg; seed::Int = 7)
    println("\n═══ One model per scenario ═══")
    rng = MersenneTwister(seed)
    models = Dict{Symbol,Any}()
    for s in SCENARIOS
        m = generate_model(mesh, priors, cfg; rng = rng, scenario = s)
        models[s] = m
        st = m.metadata["stats"]
        @printf("  %-14s log10rho %.2f / %.2f / %.2f (min/med/max)  sd=%.2f  cond=%.3f  mode=%s\n",
                s, st["log10_min"], st["log10_median"], st["log10_max"],
                st["log10_std"], st["conductive_cell_fraction"],
                m.metadata["prior_mode"])
    end
    return models
end

"""Prove mesh identity with BuildMesh2D and run the TE/TM solver on each scenario."""
function verify(mesh, priors, cfg)
    println("\n═══ Mesh identity vs MTGeophysics.BuildMesh2D ═══")
    v = verify_mesh(mesh)
    @printf("  ok=%s  max |Δy_nodes|=%.3g  max |Δz_nodes|=%.3g  max |Δreceivers|=%.3g\n",
            v.ok, v.max_y_error, v.max_z_error, v.max_receiver_error)
    @printf("  BuildMesh2D reports (n_z, n_y) = (%d, %d); generator has (%d, %d)\n",
            v.n_z, v.n_y, n_z(mesh), n_y(mesh))
    v.ok || error("mesh mismatch — the generator and the solver disagree on geometry")

    println("\n═══ TE/TM forward solve per scenario ═══")
    rng = MersenneTwister(11)
    for s in SCENARIOS
        m = generate_model(mesh, priors, cfg; rng = rng, scenario = s)
        ρ = solver_resistivity(m, mesh)
        size(ρ) == (n_z(mesh), n_y(mesh)) ||
            error("solver array is $(size(ρ)), expected $((n_z(mesh), n_y(mesh))))")
        t = @elapsed r = forward_response(m, mesh)
        okte = all(isfinite, r.rho_xy) && all(>(0), r.rho_xy)
        oktm = all(isfinite, r.rho_yx) && all(>(0), r.rho_yx)
        @printf("  %-14s TE rho_a %.1f…%.1f ohm.m  phase %.1f…%.1f deg | TM %.1f…%.1f | %s %s | %.1f s\n",
                s, minimum(r.rho_xy), maximum(r.rho_xy),
                minimum(r.phase_xy), maximum(r.phase_xy),
                minimum(r.rho_yx), maximum(r.rho_yx),
                okte ? "TE ok" : "TE BAD", oktm ? "TM ok" : "TM BAD", t)
        (okte && oktm) || error("non-physical response for scenario $(s)")
        size(r.rho_xy) == (length(mesh.frequencies), length(mesh.receiver_positions)) ||
            error("response shape mismatch for scenario $(s)")
    end
    println("  all scenarios produced finite, positive apparent resistivities")
end

function main()
    opts = parse_args(ARGS)
    cfg = GeneratorConfig()
    mesh = build_generator_mesh(cfg)
    priors = load_generator_priors(opts[:priors])

    describe_setup(mesh, priors, cfg)
    sample_per_scenario(mesh, priors, cfg)

    opts[:verify] && verify(mesh, priors, cfg)

    if opts[:generate]
        path = generate_dataset(opts[:out];
                                n_models = opts[:n], cfg = cfg, priors = priors,
                                mesh = mesh, seed = opts[:seed],
                                format = opts[:format])
        sz = round(filesize(path) / 1024^2; digits = 2)
        @printf("\nWrote %s (%.2f MB) — %d models\n", relpath(path, ROOT), sz, opts[:n])
    end
end

main()
