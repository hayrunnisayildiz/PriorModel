#!/usr/bin/env julia
#=
Smart Prior Generator & Physics-Constrained 3D Gravity Inversion — orchestrator.

Usage (from project root):
    julia --project=. main.jl

Pipeline:
  1. Data fusion        → 10-channel tensor (X, Y, Z, C)
  2. Smart prior        → m0, bounds (m_min, m_max)  [g/cm³]
  3. Physics inversion  → m_final under prior + box constraints
  4. HDF5 export        → data/processed/density_model_results.h5
=#

using Pkg
Pkg.activate(@__DIR__)

using Random
using Statistics
using Printf
using HDF5
using Lux

# ── Project modules ─────────────────────────────────────────────────────────
# Include order: GridSpecs first, then consumers. Voxelizer / ForwardGravity
# each `include` their own GridSpecs copy (Julia modules are not singletons),
# so fusion and inversion GridSpec types are distinct — see aliases below.
const ROOT = @__DIR__

include(joinpath(ROOT, "src", "fusion", "GridSpec.jl"))                 # GridSpecs
include(joinpath(ROOT, "src", "fusion", "Voxelizer.jl"))                # Voxelizer (+ GridSpecs)
include(joinpath(ROOT, "src", "neural_prior", "PriorNet3D.jl"))         # PriorNet3D (+ CBAM)
include(joinpath(ROOT, "src", "neural_prior", "Losses.jl"))              # Losses
include(joinpath(ROOT, "src", "physics_inversion", "ForwardGravity.jl")) # ForwardGravity
include(joinpath(ROOT, "src", "physics_inversion", "Misfit.jl"))         # Misfit
include(joinpath(ROOT, "src", "physics_inversion", "InversionSolver.jl")) # InversionSolver

# Selective `using` avoids export clashes (`total_variation` lives in both
# Losses and Misfit). Modules themselves are already in Main after `include`.
using .Voxelizer: build_fusion_tensor, fusion_channel_names
using .PriorNet3D
using .InversionSolver

# Fusion GridSpec (Voxelizer) vs inversion GridSpec (ForwardGravity nest).
const VoxGridSpecs = Voxelizer.GridSpecs
const InvGridSpecs = InversionSolver.Misfit.ForwardGravity.GridSpecs

# ── Paths & geological constants ────────────────────────────────────────────

const CONFIG_PATH = joinpath(ROOT, "config", "config.yaml")
const OUTPUT_H5   = joinpath(ROOT, "data", "processed", "density_model_results.h5")

"""1-based fusion-tensor index of gravimetric channel (`tensor_channels` YAML id: 0)."""
const GRAVITY_CHANNEL::Int = 1

"""Geological density sanity window for mafic / ultramafic + sulfide hosts (g/cm³)."""
const RHO_GEO_MIN::Float32 = 1.8f0
const RHO_GEO_MAX::Float32 = 4.0f0

# Full deposit grid is ~3×10⁶ voxels; finite-difference L-BFGS cannot invert
# that in one workstation pass. Coarsen the model and cap observation count.
const INVERSION_COARSEN::Int = 8
const MAX_OBS_POINTS::Int = 64
const INVERSION_MAXITERS::Int = 25

# ── Helpers ─────────────────────────────────────────────────────────────────

"""
    extract_gravity_observations(tensor, grid; channel=1)
        -> (d_obs, obs_x, obs_y, obs_z)

Pull surface gravity from fusion channel 1 (`gravity`, mGal).

Uses the top slice `k = nz` at cell centres; station `z` is `grid.zmax`
(surface RL envelope). Non-finite voxels are skipped.
"""
function extract_gravity_observations(tensor::Array{Float32,4},
                                      grid::VoxGridSpecs.GridSpec;
                                      channel::Int=GRAVITY_CHANNEL)
    nxx, nyy, nzz, nc = size(tensor)
    channel <= nc || throw(ArgumentError("gravity channel $channel > n_channels $nc"))

    xc = VoxGridSpecs.x_centers(grid)
    yc = VoxGridSpecs.y_centers(grid)
    k_top = nzz
    z_surf = grid.zmax

    obs_x = Float32[]
    obs_y = Float32[]
    obs_z = Float32[]
    d_obs = Float32[]

    @inbounds for j in 1:nyy, i in 1:nxx
        v = tensor[i, j, k_top, channel]
        if isfinite(v)
            push!(obs_x, xc[i])
            push!(obs_y, yc[j])
            push!(obs_z, z_surf)
            push!(d_obs, v)
        end
    end
    return d_obs, obs_x, obs_y, obs_z
end

"""Deterministic subsample when the station count exceeds `max_n`."""
function subsample_observations(d_obs::Vector{Float32},
                                obs_x::Vector{Float32},
                                obs_y::Vector{Float32},
                                obs_z::Vector{Float32};
                                max_n::Int=MAX_OBS_POINTS,
                                rng::AbstractRNG=Random.MersenneTwister(42))
    n = length(d_obs)
    n <= max_n && return d_obs, obs_x, obs_y, obs_z
    idx = sort(randperm(rng, n)[1:max_n])
    return d_obs[idx], obs_x[idx], obs_y[idx], obs_z[idx]
end

"""
    coarsen_volume(m, grid, factor) -> (m_coarse, grid_coarse)

Strided decimation every `factor` voxels along X, Y, Z, with a matching
`GridSpec` whose `dx, dy, dz` are scaled by `factor`.
"""
function coarsen_volume(m::Array{Float32,3},
                        grid::InvGridSpecs.GridSpec,
                        factor::Int)
    factor >= 1 || throw(ArgumentError("coarsen factor must be ≥ 1"))
    factor == 1 && return m, grid

    mc = m[1:factor:end, 1:factor:end, 1:factor:end]
    nxx, nyy, nzz = size(mc)
    f = Float32(factor)
    gc = InvGridSpecs.GridSpec(
        grid.xmin,
        grid.xmin + Float32(nxx) * grid.dx * f,
        grid.ymin,
        grid.ymin + Float32(nyy) * grid.dy * f,
        grid.zmin,
        grid.zmin + Float32(nzz) * grid.dz * f,
        grid.dx * f,
        grid.dy * f,
        grid.dz * f,
        grid.epsg_code,
    )
    return mc, gc
end

"""Squeeze a prior head `(X,Y,Z,1[,B])` down to `(X, Y, Z)`."""
function prior_to_3d(m0::AbstractArray{Float32})::Array{Float32,3}
    nd = ndims(m0)
    nd == 3 && return Array{Float32,3}(m0)
    nd == 4 && size(m0, 4) == 1 && return Array{Float32,3}(dropdims(m0; dims=4))
    nd == 5 && return Array{Float32,3}(m0[:, :, :, 1, 1])
    throw(ArgumentError("unexpected prior shape $(size(m0))"))
end

"""Split bounds tensor `(X,Y,Z,2[,B])` into `(m_min, m_max)` volumes."""
function bounds_to_3d(bounds::AbstractArray{Float32})::Tuple{Array{Float32,3}, Array{Float32,3}}
    nd = ndims(bounds)
    if nd == 4
        return Array{Float32,3}(bounds[:, :, :, 1]),
               Array{Float32,3}(bounds[:, :, :, 2])
    elseif nd == 5
        return Array{Float32,3}(bounds[:, :, :, 1, 1]),
               Array{Float32,3}(bounds[:, :, :, 2, 1])
    end
    throw(ArgumentError("unexpected bounds shape $(size(bounds))"))
end

"""Print m0 extrema / mean and test the geological window [1.8, 4.0] g/cm³."""
function check_geological_range(m0::Array{Float32,3};
                                lo::Float32=RHO_GEO_MIN,
                                hi::Float32=RHO_GEO_MAX)
    mn, mx = extrema(m0)
    μ = mean(m0)
    println("  m0 statistics: min=$(round(mn; digits=4))  max=$(round(mx; digits=4))  mean=$(round(μ; digits=4)) g/cm³")
    if mn >= lo && mx <= hi
        println("  ✓ m0 lies within geological range [$(lo), $(hi)] g/cm³")
    else
        println("  ⚠ m0 extends outside [$(lo), $(hi)] g/cm³ — review prior network / training")
    end
    return mn, mx, μ
end

"""Print the inversion misfit trajectory (data / prior / TV / bounds / total)."""
function print_misfit_history(history::InversionSolver.InversionHistory; max_lines::Int=30)
    n = length(history.total)
    println("── Inversion misfit history ($n records) ──")
    println("  step     data(mGal²)   prior      tv        bounds    total")
    idxs = n <= max_lines ? (1:n) : vcat(1:5, (n - max_lines + 6):n)
    prev = 0
    for i in idxs
        if prev > 0 && i - prev > 1
            println("  ...")
        end
        @printf("  %4d  %12.6g  %8.4g  %8.4g  %8.4g  %8.4g\n",
                i,
                history.data[i],
                history.prior[i],
                history.tv[i],
                history.bounds[i],
                history.total[i])
        prev = i
    end
end

"""
Write m0, bounds (m_min / m_max) and m_final to HDF5.

Units: density g/cm³; misfit history mGal² (data term).
"""
function save_results_h5(path::AbstractString;
                         m0::Array{Float32,3},
                         m_min::Array{Float32,3},
                         m_max::Array{Float32,3},
                         m_final::Array{Float32,3},
                         m0_inv::Array{Float32,3},
                         history::InversionSolver.InversionHistory,
                         grid_full::InvGridSpecs.GridSpec,
                         grid_inv::InvGridSpecs.GridSpec,
                         coarsen_factor::Int)
    mkpath(dirname(path))
    h5open(path, "w") do f
        write(f, "m0", m0)
        write(f, "m_min", m_min)
        write(f, "m_max", m_max)
        write(f, "bounds", cat(m_min, m_max; dims=4))   # (X, Y, Z, 2)
        write(f, "m_final", m_final)
        write(f, "m0_inversion_grid", m0_inv)
        write(f, "history/total", history.total)
        write(f, "history/data", history.data)
        write(f, "history/prior", history.prior)
        write(f, "history/tv", history.tv)
        write(f, "history/bounds", history.bounds)
        attrs = attributes(f)
        attrs["units/density"] = "g/cm3"
        attrs["units/gravity_misfit"] = "mGal2"
        attrs["inversion/coarsen_factor"] = coarsen_factor
        attrs["grid_full/xmin"] = grid_full.xmin
        attrs["grid_full/dx"] = grid_full.dx
        attrs["grid_full/epsg"] = grid_full.epsg_code
        attrs["grid_inv/xmin"] = grid_inv.xmin
        attrs["grid_inv/dx"] = grid_inv.dx
    end
    println("  Saved: $path")
end

# ── Pipeline ────────────────────────────────────────────────────────────────

function run_pipeline()
    println("═══════════════════════════════════════════════════════════════")
    println(" Smart Prior + Physics Inversion Pipeline")
    println(" Config: $CONFIG_PATH")
    println("═══════════════════════════════════════════════════════════════")

    # ── Step 1: Data fusion & tensor construction ─────────────────────────
    # load_gridspec → GridSpec; build_fusion_tensor → (X, Y, Z, C=11) Float32.
    # Voxelizer.GridSpec is the type build_fusion_tensor accepts (not Main.GridSpecs).
    println("\n[Step 1/4] Data fusion & tensor construction")
    local grid_vox::VoxGridSpecs.GridSpec
    local grid_inv_full::InvGridSpecs.GridSpec
    local tensor::Array{Float32,4}
    @time try
        grid_vox = Voxelizer.load_gridspec(CONFIG_PATH)
        println("  GridSpec: ", grid_vox)
        tensor = build_fusion_tensor(grid_vox, CONFIG_PATH)
        nxx, nyy, nzz, nc = size(tensor)
        println("  Fusion tensor: ($nxx, $nyy, $nzz, $nc)  Float32  (X, Y, Z, C)")
        ch_names = fusion_channel_names(CONFIG_PATH)
        println("  Channels: ", join(ch_names, ", "))
        grid_inv_full = InvGridSpecs.load_gridspec(CONFIG_PATH)
    catch err
        println("  ✗ Step 1 failed — fusion tensor:")
        showerror(stdout, err)
        println()
        rethrow()
    end

    # ── Step 2: Smart prior generation ────────────────────────────────────
    # SmartPriorNet3D(ps, st) maps the 10-channel volume onto m0 and
    # [m_min, m_max] (g/cm³). NaN empty cells are zero-filled inside the net.
    println("\n[Step 2/4] Smart prior generation (SmartPriorNet3D + CBAM)")
    local m0_full::Array{Float32,3}
    local m_min_full::Array{Float32,3}
    local m_max_full::Array{Float32,3}
    @time try
        rng = Random.default_rng()
        n_channels = size(tensor, 4)
        model = PriorNet3D.SmartPriorNet3D(in_channels=n_channels)
        x_in = PriorNet3D.add_batch_dim(PriorNet3D.replace_nan(tensor))
        ps, st = Lux.setup(rng, model)
        println("  Model: ", model)

        m0_out, bounds_out, st = PriorNet3D.generate_prior(model, x_in, ps, st)
        m0_full = prior_to_3d(m0_out)
        m_min_full, m_max_full = bounds_to_3d(bounds_out)
        println("  Prior shapes: m0=$(size(m0_full))  bounds=$(size(m_min_full))")
        check_geological_range(m0_full)

        # Optional: how many borehole density voxels could supervise training.
        _, dens_mask = Losses.extract_density_target(tensor)
        println("  Borehole density voxels (supervision mask): ", count(dens_mask))
    catch err
        println("  ✗ Step 2 failed — smart prior:")
        showerror(stdout, err)
        println()
        rethrow()
    end

    # ── Step 3: Physics-constrained inversion ─────────────────────────────
    # d_obs from gravity channel 1; run_physics_inversion under m0 + bounds.
    println("\n[Step 3/4] Physics-constrained gravity inversion")
    local m_final::Array{Float32,3}
    local m0_inv::Array{Float32,3}
    local history::InversionSolver.InversionHistory
    local grid_inv::InvGridSpecs.GridSpec
    @time try
        d_obs, obs_x, obs_y, obs_z = extract_gravity_observations(tensor, grid_vox)
        n_raw = length(d_obs)
        println("  Gravity stations from tensor channel $GRAVITY_CHANNEL (top slice): $n_raw")
        n_raw == 0 && error("no finite gravity values in fusion tensor channel $GRAVITY_CHANNEL")

        d_obs, obs_x, obs_y, obs_z = subsample_observations(
            d_obs, obs_x, obs_y, obs_z; max_n=MAX_OBS_POINTS)
        println("  Using $(length(d_obs)) observation points (max=$MAX_OBS_POINTS)")

        m0, grid_inv = coarsen_volume(m0_full, grid_inv_full, INVERSION_COARSEN)
        m0_inv = m0
        m_min, _ = coarsen_volume(m_min_full, grid_inv_full, INVERSION_COARSEN)
        m_max, _ = coarsen_volume(m_max_full, grid_inv_full, INVERSION_COARSEN)
        println("  Inversion grid coarsen×$(INVERSION_COARSEN): ", grid_inv,
                "  model size=$(size(m0))")
        println("  Gravity is linear in density — assembling G once, then L-BFGS with analytic ∇Φ")

        result = InversionSolver.run_physics_inversion(
            m0, m_min, m_max, d_obs, grid_inv, obs_x, obs_y, obs_z;
            maxiters=INVERSION_MAXITERS,
            λ_prior=1.0f-2,
            λ_tv=1.0f-3,
            verbose=true,
        )
        m_final = result.m_final
        history = result.history
        println("  Optimizer retcode: ", result.sol.retcode)
        println("  m_final: min=$(minimum(m_final))  max=$(maximum(m_final))  mean=$(mean(m_final)) g/cm³")
        print_misfit_history(history)
    catch err
        println("  ✗ Step 3 failed — inversion:")
        showerror(stdout, err)
        println()
        rethrow()
    end

    # ── Step 4: HDF5 export ───────────────────────────────────────────────
    println("\n[Step 4/4] Saving results to HDF5")
    @time try
        save_results_h5(OUTPUT_H5;
                        m0=m0_full,
                        m_min=m_min_full,
                        m_max=m_max_full,
                        m_final=m_final,
                        m0_inv=m0_inv,
                        history=history,
                        grid_full=grid_inv_full,
                        grid_inv=grid_inv,
                        coarsen_factor=INVERSION_COARSEN)
    catch err
        println("  ✗ Step 4 failed — HDF5 export:")
        showerror(stdout, err)
        println()
        rethrow()
    end

    println("\n═══════════════════════════════════════════════════════════════")
    println(" Pipeline complete.")
    println(" Output: $OUTPUT_H5")
    println("═══════════════════════════════════════════════════════════════")
    return (; grid_vox, grid_inv_full, tensor, m0_full, m_min_full, m_max_full, m_final, history)
end

# ── Entry point ─────────────────────────────────────────────────────────────

function main()
    try
        return run_pipeline()
    catch err
        println("\n✗ Pipeline aborted.")
        showerror(stdout, err)
        println()
        rethrow()
    end
end

main()
