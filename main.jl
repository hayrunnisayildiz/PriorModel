#!/usr/bin/env julia
#=
2-D MT neural prior → VFSA2DMT orchestrator.

Thin wrapper: includes and calls `examples/run_mt_prior_inversion.jl`.
That example is the canonical E2E script — do not duplicate it here.

Usage (from project root):
    julia --project=. main.jl
    julia --project=. main.jl --ckpt models/production_prior_v7_commemi.jld2

Colab (cwd is /content — do not use --project=. from there):
    julia --project=/content/PriorModel /content/PriorModel/main.jl

The frozen 3-D Keivitsa U-Net line lives under archive/3d_keivitsa/;
see archive/3d_keivitsa/README_3D_ARCHIVE.md.
=#

include(joinpath(@__DIR__, "examples", "run_mt_prior_inversion.jl"))

main(parse_args(ARGS))
