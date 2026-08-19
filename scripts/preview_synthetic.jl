#!/usr/bin/env julia
#=
Render synthetic sections with MTGeophysics.jl's own model plotter, so what you
see is exactly what the TE/TM solver sees.

Usage:
    julia --project=. scripts/preview_synthetic.jl
    julia --project=. scripts/preview_synthetic.jl --n 3 --out figures/synthetic
    julia --project=. scripts/preview_synthetic.jl --scenario dipping_fault --n 6

Options:
    --out DIR           output directory              [figures/synthetic_preview]
    --n COUNT           realisations per scenario                           [1]
    --scenario NAME     only this scenario (default: all of SCENARIOS)
    --seed N            base RNG seed                                      [42]
    --padding           include the lateral padding in the figure
    --depth KM          maximum plotted depth, km       [target zone bottom]
=#

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator

const ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(argv::Vector{String})
    o = Dict{Symbol,Any}(:out => joinpath(ROOT, "figures", "synthetic_preview"),
                         :n => 1, :scenario => nothing, :seed => 42,
                         :padding => false, :depth => nothing)
    i = 1
    while i <= length(argv)
        a = argv[i]
        need() = (i += 1; i <= length(argv) ? argv[i] : error("$(a) needs a value"))
        if a == "--out";          o[:out] = abspath(need())
        elseif a == "--n";        o[:n] = parse(Int, need())
        elseif a == "--scenario"; o[:scenario] = Symbol(need())
        elseif a == "--seed";     o[:seed] = parse(Int, need())
        elseif a == "--padding";  o[:padding] = true
        elseif a == "--depth";    o[:depth] = parse(Float64, need())
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
    return o
end

function main()
    o = parse_args(ARGS)
    cfg = GeneratorConfig()
    mesh = build_generator_mesh(cfg)
    priors = load_generator_priors()
    MTG = SyntheticGenerator.mtgeophysics()

    zc = z_centers(mesh)[mesh.target_z]
    depth_km = o[:depth] === nothing ? (zc[end] + cfg.target_dz) / 1000 : o[:depth]
    scenarios = o[:scenario] === nothing ? collect(SCENARIOS) : [o[:scenario]]
    mkpath(o[:out])

    for s in scenarios
        rng = MersenneTwister(o[:seed])
        for k in 1:o[:n]
            m = generate_model(mesh, priors, cfg; rng = rng, scenario = s)
            ρ = solver_resistivity(m, mesh)
            name = o[:n] == 1 ? "$(s).png" : @sprintf("%s_%02d.png", s, k)
            path = joinpath(o[:out], name)
            Base.invokelatest(MTG.plot_mt2d_model, SyntheticGenerator.to_mt2d_mesh(mesh), ρ;
                              output_path = path,
                              show_padding = o[:padding],
                              maximum_depth_km = depth_km,
                              resistivity_log10_range = cfg.log10_clip)
            st = m.metadata["stats"]
            @printf("%-14s seed=%-11d log10rho %.2f…%.2f  %s\n",
                    s, m.seed, st["log10_min"], st["log10_max"], relpath(path, ROOT))
        end
    end
    println("\nFigures in ", relpath(o[:out], ROOT))
end

main()
