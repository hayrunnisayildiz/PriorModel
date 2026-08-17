#!/usr/bin/env julia
#=
Smart Prior Generator & Physics-Constrained 3D Gravity Inversion — orchestrator.

Usage (from project root):
    julia --project=. main.jl

Prerequisites:
    julia --project=. src/training/train_prior.jl   # produces models/best_prior_model.jld2

Pipeline:
  1. Load preprocessed volume  → keivitsa_preprocessed.h5
  2. Tiled prior inference     → m0, m_min, m_max  [g/cm³]  (UnifiedPriorUNet3D)
  3. Physics inversion         → m_final under prior + box constraints
  4. HDF5 export               → data/processed/density_model_results.h5
=#

using Pkg
Pkg.activate(@__DIR__)

using Random
using Statistics
using Printf
using HDF5
using JLD2
using Lux

const ROOT = @__DIR__

include(joinpath(ROOT, "src", "fusion", "GridSpec.jl"))
include(joinpath(ROOT, "src", "data", "patch_loader.jl"))
include(joinpath(ROOT, "src", "networks", "prior_unet3d.jl"))
include(joinpath(ROOT, "src", "physics_inversion", "ForwardGravity.jl"))
include(joinpath(ROOT, "src", "physics_inversion", "Misfit.jl"))
include(joinpath(ROOT, "src", "physics_inversion", "InversionSolver.jl"))

using .PriorUNet3DLayers: UnifiedPriorUNet3D, replace_nan, OUT_M0, OUT_MMIN, OUT_MMAX
using .InversionSolver

const VoxGridSpecs = GridSpecs
const InvGridSpecs = InversionSolver.Misfit.ForwardGravity.GridSpecs

# ── Paths & constants ───────────────────────────────────────────────────────

const PREPROCESSED_H5 = joinpath(ROOT, "data", "processed", "keivitsa_preprocessed.h5")
const MODEL_CHECKPOINT = joinpath(ROOT, "models", "best_prior_model.jld2")
const OUTPUT_H5 = joinpath(ROOT, "data", "processed", "density_model_results.h5")

const GRAVITY_CHANNEL::Int = 1
const RHO_GEO_MIN::Float32 = 1.8f0
const RHO_GEO_MAX::Float32 = 4.0f0

const INVERSION_COARSEN::Int = 8
const MAX_OBS_POINTS::Int = 64
const INVERSION_MAXITERS::Int = 25
const INVERSION_LAMBDA_PRIOR::Float32 = 1.0f-2
const INVERSION_LAMBDA_TV::Float32 = 1.0f-3
const INVERSION_LAMBDA_BOUNDS::Float32 = 1.0f2

# ── GPU (optional) ──────────────────────────────────────────────────────────

function cuda_functional()::Bool
    try
        @eval import CUDA
        return CUDA.functional()
    catch
        return false
    end
end

function to_device(x, ::Val{true})
    @eval import CUDA
    return Lux.recursive_map(CUDA.cu, x)
end
to_device(x, ::Val{false}) = x

# ── Dataset helpers ─────────────────────────────────────────────────────────

function load_grid_from_h5(path::AbstractString)::InvGridSpecs.GridSpec
    h5open(path, "r") do f
        a = attributes(f)
        return InvGridSpecs.GridSpec(
            Float32(read(a["xmin"])), Float32(read(a["xmax"])),
            Float32(read(a["ymin"])), Float32(read(a["ymax"])),
            Float32(read(a["zmin"])), Float32(read(a["zmax"])),
            Float32(read(a["dx"])), Float32(read(a["dy"])), Float32(read(a["dz"])),
            Int32(read(a["epsg"])),
        )
    end
end

"""Pick a 4-D volume dataset key (`fusion` → `Y` → first 4-D array)."""
function resolve_dataset_key(path::AbstractString;
                             preferred::AbstractString="fusion")::Tuple{String,NTuple{4,Int}}
    h5open(path, "r") do f
        if haskey(f, preferred) && f[preferred] isa HDF5.Dataset
            d = f[preferred]
            return (string(preferred), (Int.(size(d))...,)[1:4])
        end
        for key in ("fusion", "tensor", "Y", "F")
            key == preferred && continue
            if haskey(f, key) && f[key] isa HDF5.Dataset
                sz = size(f[key])
                length(sz) == 4 || continue
                @info "Dataset fallback" preferred=preferred dataset_key=key shape=sz
                return (key, (Int(sz[1]), Int(sz[2]), Int(sz[3]), Int(sz[4])))
            end
        end
        for key in keys(f)
            d = f[key]
            if d isa HDF5.Dataset && ndims(d) == 4
                @info "Dataset fallback" preferred=preferred dataset_key=key shape=size(d)
                return (string(key), (Int.(size(d))...,)[1:4])
            end
        end
        error("No 4-D volume dataset found in $(path)")
    end
end

"""Non-overlapping tile origins; trailing partial voxels get one final aligned tile."""
function tile_origins(n::Int, patch::Int)::Vector{Int}
    n <= patch && return [1]
    starts = collect(1:patch:(n - patch + 1))
    last = n - patch + 1
    if starts[end] != last
        push!(starts, last)
    end
    return starts
end

# ── Checkpoint loading ────────────────────────────────────────────────────────

"""
    load_trained_prior(checkpoint_path) -> (model, ps, st, meta)

Load `UnifiedPriorUNet3D` weights saved by `train_prior.jl`.
"""
function load_trained_prior(checkpoint_path::AbstractString)
    isfile(checkpoint_path) || error("""
Trained prior checkpoint not found: $(abspath(checkpoint_path))

Run training first:
    julia --project=. src/training/train_prior.jl
""")

    data = jldopen(checkpoint_path, "r") do f
        (
            model = read(f, "model"),
            ps = read(f, "ps"),
            st = read(f, "st"),
            cfg = haskey(f, "cfg") ? read(f, "cfg") : nothing,
            in_channels = haskey(f, "in_channels") ? read(f, "in_channels") : nothing,
            dataset_key = haskey(f, "dataset_key") ? read(f, "dataset_key") : "fusion",
            epoch = haskey(f, "epoch") ? read(f, "epoch") : -1,
            best_loss = haskey(f, "best_loss") ? read(f, "best_loss") : NaN32,
        )
    end

    data.model isa UnifiedPriorUNet3D ||
        error("Checkpoint model is $(typeof(data.model)); expected UnifiedPriorUNet3D")

    meta = (
        cfg=data.cfg,
        in_channels=data.in_channels,
        dataset_key=data.dataset_key,
        epoch=data.epoch,
        best_loss=data.best_loss,
    )
    return data.model, data.ps, data.st, meta
end

# ── Patch inference ───────────────────────────────────────────────────────────

"""Read a possibly edge-truncated patch, zero-pad to `(px,py,pz,nc)` and append mask."""
function _read_patch_input!(dataset::HDF5.Dataset,
                            i0::Int, j0::Int, k0::Int,
                            px::Int, py::Int, pz::Int, nc::Int,
                            raw::Array{Float32,4},
                            out::Array{Float32,4})
    nx, ny, nz, _ = size(dataset)
    i1 = min(i0 + px - 1, nx)
    j1 = min(j0 + py - 1, ny)
    k1 = min(k0 + pz - 1, nz)
    fill!(raw, NaN32)
    slab = dataset[i0:i1, j0:j1, k0:k1, 1:nc]
    copyto!(view(raw, 1:(i1-i0+1), 1:(j1-j0+1), 1:(k1-k0+1), :), slab)

    @inbounds for k in 1:pz, j in 1:py, i in 1:px
        valid = false
        for c in 1:nc
            v = raw[i, j, k, c]
            if isfinite(v)
                out[i, j, k, c] = v
                valid = true
            else
                out[i, j, k, c] = 0.0f0
            end
        end
        out[i, j, k, nc + 1] = valid ? 1.0f0 : 0.0f0
    end
    return (i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)
end

"""
    infer_prior_tiled(model, ps, st, h5path, dataset_key;
                      px, py, pz, in_channels, use_gpu)

Memory-friendly full-grid inference via non-overlapping (with edge-aligned) patches.
Returns `(m0, m_min, m_max)` `(nx, ny, nz)` `Float32` volumes in ``g/cm^3``.
"""
function infer_prior_tiled(model::UnifiedPriorUNet3D, ps, st,
                           h5path::AbstractString, dataset_key::AbstractString;
                           px::Int=32, py::Int=32, pz::Int=16,
                           in_channels::Int=12,
                           use_gpu::Bool=false)
    h5open(h5path, "r") do f
        dataset = f[dataset_key]
        nx, ny, nz, nc = size(dataset)
        nc + 1 == in_channels ||
            error("in_channels=$in_channels but dataset has $nc channels (+1 mask expected)")

        m0 = zeros(Float32, nx, ny, nz)
        m_min = zeros(Float32, nx, ny, nz)
        m_max = zeros(Float32, nx, ny, nz)
        covered = falses(nx, ny, nz)

        raw = Array{Float32,4}(undef, px, py, pz, nc)
        x_patch = Array{Float32,4}(undef, px, py, pz, in_channels)
        x5 = Array{Float32,5}(undef, px, py, pz, in_channels, 1)

        i_starts = tile_origins(nx, px)
        j_starts = tile_origins(ny, py)
        k_starts = tile_origins(nz, pz)
        n_tiles = length(i_starts) * length(j_starts) * length(k_starts)
        println("  Tiled inference: $(nx)×$(ny)×$(nz)  patch=($px,$py,$pz)  tiles=$n_tiles")

        tile = 0
        t_start = time()
        for i0 in i_starts, j0 in j_starts, k0 in k_starts
            tile += 1
            di, dj, dk = _read_patch_input!(dataset, i0, j0, k0, px, py, pz, nc, raw, x_patch)
            x5[:, :, :, :, 1] = x_patch
            x_in = use_gpu ? to_device(x5, Val(true)) : x5
            vol, st = model(x_in, ps, st)

            vol_cpu = use_gpu ? Array(vol) : vol
            i1 = i0 + di - 1
            j1 = j0 + dj - 1
            k1 = k0 + dk - 1

            m0[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_M0, 1]
            m_min[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_MMIN, 1]
            m_max[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_MMAX, 1]
            covered[i0:i1, j0:j1, k0:k1] .= true

            tile % max(1, n_tiles ÷ 10) == 0 &&
                @printf("    tile %d/%d  origin=(%d,%d,%d)  elapsed=%.1fs\n",
                        tile, n_tiles, i0, j0, k0, time() - t_start)
        end

        n_cov = count(covered)
        n_tot = nx * ny * nz
        n_cov == n_tot ||
            @warn "Incomplete volume coverage" covered=n_cov total=n_tot
        @printf("  Inference finished in %.1fs  coverage=%d/%d voxels\n",
                time() - t_start, n_cov, n_tot)
        return m0, m_min, m_max
    end
end

# ── Gravity observations & coarsening ───────────────────────────────────────

"""
    extract_gravity_from_surface(X, grid; channel=1)
        -> (d_obs, obs_x, obs_y, obs_z)

Surface gravity (mGal) from preprocessed 2-D `X` `(nx, ny, C)`.
"""
function extract_gravity_from_surface(X::Array{Float32,3},
                                      grid::InvGridSpecs.GridSpec;
                                      channel::Int=GRAVITY_CHANNEL)
    nxx, nyy, nc = size(X)
    channel <= nc || throw(ArgumentError("gravity channel $channel > n_channels $nc"))

    xc = InvGridSpecs.x_centers(grid)
    yc = InvGridSpecs.y_centers(grid)
    z_surf = grid.zmax

    obs_x = Float32[]
    obs_y = Float32[]
    obs_z = Float32[]
    d_obs = Float32[]

    @inbounds for j in 1:nyy, i in 1:nxx
        v = X[i, j, channel]
        isfinite(v) || continue
        push!(obs_x, xc[i])
        push!(obs_y, yc[j])
        push!(obs_z, z_surf)
        push!(d_obs, v)
    end
    return d_obs, obs_x, obs_y, obs_z
end

function subsample_observations(d_obs, obs_x, obs_y, obs_z;
                                max_n::Int=MAX_OBS_POINTS,
                                rng::AbstractRNG=Random.MersenneTwister(42))
    n = length(d_obs)
    n <= max_n && return d_obs, obs_x, obs_y, obs_z
    idx = sort(randperm(rng, n)[1:max_n])
    return d_obs[idx], obs_x[idx], obs_y[idx], obs_z[idx]
end

function coarsen_volume(m::Array{Float32,3},
                        grid::InvGridSpecs.GridSpec,
                        factor::Int)
    factor >= 1 || throw(ArgumentError("coarsen factor must be ≥ 1"))
    factor == 1 && return m, grid

    mc = m[1:factor:end, 1:factor:end, 1:factor:end]
    nxx, nyy, nzz = size(mc)
    f = Float32(factor)
    gc = InvGridSpecs.GridSpec(
        grid.xmin, grid.xmin + Float32(nxx) * grid.dx * f,
        grid.ymin, grid.ymin + Float32(nyy) * grid.dy * f,
        grid.zmin, grid.zmin + Float32(nzz) * grid.dz * f,
        grid.dx * f, grid.dy * f, grid.dz * f,
        grid.epsg_code,
    )
    return mc, gc
end

function check_geological_range(m0::Array{Float32,3};
                                lo::Float32=RHO_GEO_MIN,
                                hi::Float32=RHO_GEO_MAX)
    mn, mx = extrema(m0)
    μ = mean(m0)
    println("  m0 statistics: min=$(round(mn; digits=4))  max=$(round(mx; digits=4))  mean=$(round(μ; digits=4)) g/cm³")
    if mn >= lo && mx <= hi
        println("  ✓ m0 within geological range [$(lo), $(hi)] g/cm³")
    else
        println("  ⚠ m0 extends outside [$(lo), $(hi)] g/cm³")
    end
    return mn, mx, μ
end

function print_misfit_history(history::InversionSolver.InversionHistory; max_lines::Int=30)
    n = length(history.total)
    println("── Inversion misfit history ($n records) ──")
    println("  step     data(mGal²)   prior      tv        bounds    total")
    idxs = n <= max_lines ? (1:n) : vcat(1:5, (n - max_lines + 6):n)
    prev = 0
    for i in idxs
        prev > 0 && i - prev > 1 && println("  ...")
        @printf("  %4d  %12.6g  %8.4g  %8.4g  %8.4g  %8.4g\n",
                i, history.data[i], history.prior[i],
                history.tv[i], history.bounds[i], history.total[i])
        prev = i
    end
end

function save_results_h5(path::AbstractString;
                         m_prior::Array{Float32,3},
                         m_min::Array{Float32,3},
                         m_max::Array{Float32,3},
                         m_final::Array{Float32,3},
                         m_prior_inv::Array{Float32,3},
                         history::InversionSolver.InversionHistory,
                         grid_full::InvGridSpecs.GridSpec,
                         grid_inv::InvGridSpecs.GridSpec,
                         run_params::NamedTuple)
    mkpath(dirname(path))
    h5open(path, "w") do f
        write(f, "m_prior", m_prior)
        write(f, "m0", m_prior)                    # legacy alias
        write(f, "m_min", m_min)
        write(f, "m_max", m_max)
        write(f, "bounds", cat(m_min, m_max; dims=4))
        write(f, "m_final", m_final)
        write(f, "m0_inversion_grid", m_prior_inv)
        write(f, "history/total", history.total)
        write(f, "history/data", history.data)
        write(f, "history/prior", history.prior)
        write(f, "history/tv", history.tv)
        write(f, "history/bounds", history.bounds)
        attrs = attributes(f)
        attrs["units/density"] = "g/cm3"
        attrs["units/gravity_misfit"] = "mGal2"
        for (k, v) in pairs(run_params)
            attrs["run/" * string(k)] = v
        end
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
    pipeline_start = time()
    use_gpu = cuda_functional()

    println("═══════════════════════════════════════════════════════════════")
    println(" Smart Prior + Physics Inversion Pipeline")
    println(" Preprocessed data: $PREPROCESSED_H5")
    println(" Model checkpoint:    $MODEL_CHECKPOINT")
    println(use_gpu ? "  Device: CUDA GPU" : "  Device: CPU")
    println("═══════════════════════════════════════════════════════════════")

    isfile(PREPROCESSED_H5) ||
        error("Preprocessed dataset not found: $(PREPROCESSED_H5)")

    # ── Step 1: Load trained prior ───────────────────────────────────────────
    println("\n[Step 1/4] Loading trained UnifiedPriorUNet3D checkpoint")
    local model::UnifiedPriorUNet3D
    local ps
    local st
    local meta
    step1_t = @elapsed begin
        model, ps, st, meta = load_trained_prior(MODEL_CHECKPOINT)
        use_gpu && (ps = to_device(ps, Val(true)))
        println("  Model: ", model)
        meta.epoch >= 0 &&
            @printf("  Checkpoint epoch=%d  best_loss=%.6e\n", meta.epoch, meta.best_loss)
        meta.in_channels !== nothing &&
            println("  Trained in_channels: ", meta.in_channels)
    end
    @printf("  Step 1 done in %.1fs\n", step1_t)

    # ── Step 2: Tiled full-volume prior inference ────────────────────────────
    println("\n[Step 2/4] Tiled prior inference (memory-safe full grid)")
    local m0_full::Array{Float32,3}
    local m_min_full::Array{Float32,3}
    local m_max_full::Array{Float32,3}
    local grid_full::InvGridSpecs.GridSpec
    local dataset_key::String
    local in_channels::Int

    step2_t = @elapsed begin
        grid_full = load_grid_from_h5(PREPROCESSED_H5)
        dataset_key, vol_shape = resolve_dataset_key(PREPROCESSED_H5;
                                                     preferred=something(meta.dataset_key, "fusion"))
        nc = vol_shape[4]
        in_channels = something(meta.in_channels, nc + 1)
        px = meta.cfg !== nothing ? meta.cfg.px : 32
        py = meta.cfg !== nothing ? meta.cfg.py : 32
        pz = meta.cfg !== nothing ? meta.cfg.pz : 16

        println("  Grid: ", grid_full)
        println("  Volume dataset: $dataset_key  shape=$vol_shape  in_channels=$in_channels")

        m0_full, m_min_full, m_max_full = infer_prior_tiled(
            model, ps, st, PREPROCESSED_H5, dataset_key;
            px=Int(px), py=Int(py), pz=Int(pz),
            in_channels=Int(in_channels), use_gpu=use_gpu,
        )
        println("  Prior shapes: m0=$(size(m0_full))  m_min=$(size(m_min_full))")
        check_geological_range(m0_full)
        @printf("  Bounds: m_min∈[%.3f, %.3f]  m_max∈[%.3f, %.3f] g/cm³\n",
                minimum(m_min_full), maximum(m_min_full),
                minimum(m_max_full), maximum(m_max_full))
    end
    @printf("  Step 2 done in %.1fs\n", step2_t)

    # ── Step 3: Physics-constrained inversion ────────────────────────────────
    println("\n[Step 3/4] Physics-constrained gravity inversion (L-BFGS)")
    local m_final::Array{Float32,3}
    local m0_inv::Array{Float32,3}
    local history::InversionSolver.InversionHistory
    local grid_inv::InvGridSpecs.GridSpec

    step3_t = @elapsed begin
        X_surf = h5open(PREPROCESSED_H5, "r") do f
            haskey(f, "X") ? read(f, "X") : nothing
        end
        X_surf === nothing && error("Surface gravity `X` not found in $(PREPROCESSED_H5)")

        d_obs, obs_x, obs_y, obs_z = extract_gravity_from_surface(X_surf, grid_full)
        n_raw = length(d_obs)
        println("  Gravity stations from surface X (channel $GRAVITY_CHANNEL): $n_raw")
        n_raw == 0 && error("no finite gravity values in surface X")

        d_obs, obs_x, obs_y, obs_z = subsample_observations(
            d_obs, obs_x, obs_y, obs_z; max_n=MAX_OBS_POINTS)
        println("  Using $(length(d_obs)) observation points (max=$MAX_OBS_POINTS)")

        m0, grid_inv = coarsen_volume(m0_full, grid_full, INVERSION_COARSEN)
        m0_inv = m0
        m_min, _ = coarsen_volume(m_min_full, grid_full, INVERSION_COARSEN)
        m_max, _ = coarsen_volume(m_max_full, grid_full, INVERSION_COARSEN)
        println("  Inversion grid coarsen×$(INVERSION_COARSEN): ", grid_inv,
                "  model size=$(size(m0))")

        result = InversionSolver.run_physics_inversion(
            m0, m_min, m_max, d_obs, grid_inv, obs_x, obs_y, obs_z;
            maxiters=INVERSION_MAXITERS,
            λ_prior=INVERSION_LAMBDA_PRIOR,
            λ_tv=INVERSION_LAMBDA_TV,
            λ_bounds=INVERSION_LAMBDA_BOUNDS,
            verbose=true,
        )
        m_final = result.m_final
        history = result.history

        println("  Optimizer retcode: ", result.sol.retcode)
        @printf("  m_final: min=%.4f  max=%.4f  mean=%.4f g/cm³\n",
                minimum(m_final), maximum(m_final), mean(m_final))
        n_hist = length(history.total)
        n_hist >= 1 && @printf("  Final misfit: data=%.6g  total=%.6g (mGal² scale on data term)\n",
                               history.data[end], history.total[end])
        print_misfit_history(history)
    end
    @printf("  Step 3 done in %.1fs\n", step3_t)

    # ── Step 4: HDF5 export ─────────────────────────────────────────────────
    println("\n[Step 4/4] Saving results to HDF5")
    step4_t = @elapsed begin
        run_params = (
            checkpoint=basename(MODEL_CHECKPOINT),
            dataset_key=dataset_key,
            in_channels=in_channels,
            patch_px=meta.cfg !== nothing ? meta.cfg.px : 32,
            patch_py=meta.cfg !== nothing ? meta.cfg.py : 32,
            patch_pz=meta.cfg !== nothing ? meta.cfg.pz : 16,
            coarsen_factor=INVERSION_COARSEN,
            max_obs=MAX_OBS_POINTS,
            maxiters=INVERSION_MAXITERS,
            lambda_prior=INVERSION_LAMBDA_PRIOR,
            lambda_tv=INVERSION_LAMBDA_TV,
            lambda_bounds=INVERSION_LAMBDA_BOUNDS,
            train_epoch=meta.epoch,
            train_best_loss=meta.best_loss,
        )
        save_results_h5(OUTPUT_H5;
                        m_prior=m0_full,
                        m_min=m_min_full,
                        m_max=m_max_full,
                        m_final=m_final,
                        m_prior_inv=m0_inv,
                        history=history,
                        grid_full=grid_full,
                        grid_inv=grid_inv,
                        run_params=run_params)
    end
    @printf("  Step 4 done in %.1fs\n", step4_t)

    total_t = time() - pipeline_start
    println("\n═══════════════════════════════════════════════════════════════")
    @printf(" Pipeline complete in %.1fs\n", total_t)
    println(" Output: $OUTPUT_H5")
    println("═══════════════════════════════════════════════════════════════")

    return (; grid_full, m0_full, m_min_full, m_max_full, m_final, history)
end

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
