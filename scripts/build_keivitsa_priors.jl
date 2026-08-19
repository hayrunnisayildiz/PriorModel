#!/usr/bin/env julia
#=
Build config/keivitsa_priors.json — the statistical "realism parameters" that the
synthetic resistivity model generator samples from.

Usage:
    julia --project=. scripts/build_keivitsa_priors.jl
    julia --project=. scripts/build_keivitsa_priors.jl --yaml
    julia --project=. scripts/build_keivitsa_priors.jl --populations 2 --cell 50

Options:
    --out PATH          JSON output path (default config/keivitsa_priors.json)
    --yaml [PATH]       also write a YAML mirror (default config/keivitsa_priors.yaml)
    --config PATH       dataset YAML (default config/dataset_config.yaml)
    --populations K     Gaussian mixture components fitted to log10(rho)  [3]
    --cell METRES       VLF raster cell for plan-view geometry   [auto: line spacing]
    --mad-sigma S       robust log10 outlier rejection, 0 disables         [6]
=#

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

include(joinpath(@__DIR__, "..", "src", "analysis", "keivitsa_stats.jl"))
using .KeivitsaStats

const ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_args(argv::Vector{String})
    kw = Dict{Symbol,Any}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        next() = (i += 1; i <= length(argv) ? argv[i] : error("$(a) needs a value"))
        if a == "--out"
            kw[:output_json] = abspath(next())
        elseif a == "--yaml"
            peek = i < length(argv) ? argv[i+1] : ""
            kw[:output_yaml] = startswith(peek, "--") || isempty(peek) ?
                               joinpath(ROOT, "config", "keivitsa_priors.yaml") :
                               abspath(next())
        elseif a == "--config"
            kw[:config_path] = abspath(next())
        elseif a == "--populations"
            kw[:n_populations] = parse(Int, next())
        elseif a == "--cell"
            kw[:raster_cell_m] = parse(Float64, next())
        elseif a == "--mad-sigma"
            kw[:mad_sigma] = parse(Float64, next())
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
    return kw
end

KeivitsaStats.run_stats(; parse_args(ARGS)...)
