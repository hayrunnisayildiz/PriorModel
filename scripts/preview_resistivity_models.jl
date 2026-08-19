#!/usr/bin/env julia
#=
Generate and preview MT-only resistivity models on MeshParams.

Usage:
    julia --project=. scripts/preview_resistivity_models.jl
    julia --project=. scripts/preview_resistivity_models.jl --n 5 --generic
    julia --project=. scripts/preview_resistivity_models.jl --out figures/resistivity_preview
=#

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "synthetic", "ResistivityModelGenerator.jl"))
using .ResistivityModelGenerator

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATASET_CONFIG = joinpath(ROOT, "config", "dataset_config.yaml")

function parse_args(argv::Vector{String})
    o = Dict{Symbol,Any}(
        :n => 5,
        :out => joinpath(ROOT, "figures", "resistivity_preview"),
        :data => joinpath(ROOT, "data", "synthetic", "resistivity_models"),
        :generic => false,
        :seed => 42,
        :diversity => 1.0,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        need() = (i += 1; i <= length(argv) ? argv[i] : error("$(a) needs a value"))
        if a == "--n"; o[:n] = parse(Int, need())
        elseif a == "--out"; o[:out] = abspath(need())
        elseif a == "--data"; o[:data] = abspath(need())
        elseif a == "--generic"; o[:generic] = true
        elseif a == "--seed"; o[:seed] = parse(Int, need())
        elseif a == "--diversity"; o[:diversity] = parse(Float64, need())
        elseif a in ("-h", "--help")
            println(read(@__FILE__, String)[1:findfirst("=#", read(@__FILE__, String))[1]-1])
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
    mp = DEFAULT_MESH
    calib = o[:generic] ? nothing : calibration_stats(DATASET_CONFIG)

    println("═══ ResistivityModelGenerator preview ═══")
    println("  mesh: ", mp)
    if calib !== nothing
        b = calib.combined_log10
        @printf("  calibration: n=%d  log10 p05/p50/p95 = %.2f / %.2f / %.2f\n",
                b.n, b.p05, b.p50, b.p95)
        @printf("  layer thickness p50 = %.2f m\n", calib.layer_thickness_m.p50)
    else
        println("  calibration: none (generic diversity mode)")
    end

    mkpath(o[:out])
    paths = generate_dataset(o[:n], o[:data], mp; calib=calib,
                             diversity_level=o[:diversity], seed=o[:seed])

    for (k, p) in enumerate(paths)
        model, mp_loaded = load_synthetic_model(p)
        size(model) == (mp.nz, mp.nx) ||
            error("model $(p) has size $(size(model)), expected ($(mp.nz), $(mp.nx))")
        fig = joinpath(o[:out], @sprintf("model_%04d.pgm", k))
        plot_resistivity_model(model, mp_loaded; path=fig,
                               title=@sprintf("sample %d  log10 rho", k))
        @printf("  [%d] log10 rho %.2f…%.2f  mean=%.2f  -> %s\n",
                k, minimum(model), maximum(model), mean(model), relpath(fig, ROOT))
    end

    println("\nModels: ", relpath(o[:data], ROOT))
    println("Figures: ", relpath(o[:out], ROOT))
end

main()
