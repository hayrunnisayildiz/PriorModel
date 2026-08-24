#!/usr/bin/env julia
#=
Critical-path tests for the 2-D MT prior pipeline.

Usage (from project root):
    julia --project=. test/runtests.jl
=#

using Test

# Headless: MTGeophysics may pull Plots/GR during load_data2d / load_model2d.
get!(ENV, "GKSwstype", "nul")

@testset "PriorModel critical contracts" begin
    include("test_mesh_params.jl")
    include("test_mt_input_standardizer.jl")
    include("test_ini_export.jl")
end
