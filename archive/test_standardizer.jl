#!/usr/bin/env julia
#=
MT input standardizer checks moved to test/test_mt_input_standardizer.jl.

Usage (from project root):
    julia --project=. test/runtests.jl
    julia --project=. scripts/test_standardizer.jl   # this wrapper
=#

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "test", "runtests.jl"))
