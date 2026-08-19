#!/usr/bin/env julia
#=
Smart Prior Generator — EM / IP / resistivity U-Net orchestrator.

Usage (from project root):
    julia --project=. main.jl

Colab (cwd is /content — do not use --project=. from there):
    julia --project=/content/PriorModel /content/PriorModel/main.jl

Prerequisites:
    julia --project=. src/training/train_prior.jl   # produces models/best_prior_model.jld2

Pipeline:
  1. Load preprocessed EM volume → keivitsa_preprocessed.h5
  2. Tiled prior inference       → m0, m_min, m_max  [ohm·m]  (UnifiedPriorUNet3D)
  3. HDF5 export                 → data/processed/resistivity_prior_results.h5
=#

using Pkg
include(joinpath(@__DIR__, "src", "pkg_setup.jl"))
activate_project!(@__DIR__)

# Load LuxCUDA (CUDA + cuDNN) before Lux network includes (Julia 1.12 world-age).
try
    using LuxCUDA
catch err
    @info "LuxCUDA unavailable; inference will use CPU" exception=err project=Base.active_project()
end

using Random
using Statistics
using Printf
using HDF5
using JLD2
using Lux

const ROOT = @__DIR__

include(joinpath(ROOT, "src", "gpu_utils.jl"))
GPUUtils.init_cuda!()
include(joinpath(ROOT, "src", "fusion", "GridSpec.jl"))
include(joinpath(ROOT, "src", "data", "patch_loader.jl"))
include(joinpath(ROOT, "src", "networks", "prior_unet3d.jl"))

using .GPUUtils: cuda_available, to_device, device_label, cuda_diagnostics, gpu_forward
using .PriorUNet3DLayers: UnifiedPriorUNet3D, replace_nan, OUT_M0, OUT_MMIN, OUT_MMAX

const VoxGridSpecs = GridSpecs

# ── Paths & constants ───────────────────────────────────────────────────────

const PREPROCESSED_H5 = joinpath(ROOT, "data", "processed", "keivitsa_preprocessed.h5")
const MODEL_CHECKPOINT = joinpath(ROOT, "models", "best_prior_model.jld2")
const OUTPUT_H5 = joinpath(ROOT, "data", "processed", "resistivity_prior_results.h5")

const RES_GEO_MIN::Float32 = 0.1f0      # ohm·m
const RES_GEO_MAX::Float32 = 1.0f5      # ohm·m
const INFER_BATCH_SIZE::Int = 4

# ── Dataset helpers ─────────────────────────────────────────────────────────

function load_grid_from_h5(path::AbstractString)::VoxGridSpecs.GridSpec
    h5open(path, "r") do f
        a = HDF5.attributes(f)
        return VoxGridSpecs.GridSpec(
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
                            out::AbstractArray{Float32,4})
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
Returns `(m0, m_min, m_max)` `(nx, ny, nz)` `Float32` volumes in ohm·m.
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
        batch_size = use_gpu ? INFER_BATCH_SIZE : 1
        x5_batch = Array{Float32,5}(undef, px, py, pz, in_channels, batch_size)
        batch_meta = NTuple{6,Int}[]

        i_starts = tile_origins(nx, px)
        j_starts = tile_origins(ny, py)
        k_starts = tile_origins(nz, pz)
        n_tiles = length(i_starts) * length(j_starts) * length(k_starts)
        println("  Tiled inference: $(nx)×$(ny)×$(nz)  patch=($px,$py,$pz)  tiles=$n_tiles  batch=$batch_size")

        function flush_batch!(n_in_batch::Int)
            n_in_batch == 0 && return
            x_slice = n_in_batch == batch_size ? x5_batch :
                      x5_batch[:, :, :, :, 1:n_in_batch]
            x_in = use_gpu ? to_device(x_slice, Val(true)) : x_slice
            vol, st_out = gpu_forward(use_gpu) do
                model(x_in, ps, st)
            end
            st = st_out
            vol_cpu = use_gpu ? Array(vol) : vol

            for b in 1:n_in_batch
                i0, j0, k0, di, dj, dk = batch_meta[b]
                i1 = i0 + di - 1
                j1 = j0 + dj - 1
                k1 = k0 + dk - 1
                m0[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_M0, b]
                m_min[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_MMIN, b]
                m_max[i0:i1, j0:j1, k0:k1] .= vol_cpu[1:di, 1:dj, 1:dk, OUT_MMAX, b]
                covered[i0:i1, j0:j1, k0:k1] .= true
            end
            empty!(batch_meta)
        end

        tile = 0
        batch_n = 0
        t_start = time()
        for i0 in i_starts, j0 in j_starts, k0 in k_starts
            tile += 1
            batch_n += 1
            x_patch = view(x5_batch, :, :, :, :, batch_n)
            di, dj, dk = _read_patch_input!(dataset, i0, j0, k0, px, py, pz, nc, raw, x_patch)
            push!(batch_meta, (i0, j0, k0, di, dj, dk))

            if batch_n == batch_size
                flush_batch!(batch_n)
                batch_n = 0
            end

            tile % max(1, n_tiles ÷ 10) == 0 &&
                @printf("    tile %d/%d  origin=(%d,%d,%d)  elapsed=%.1fs\n",
                        tile, n_tiles, i0, j0, k0, time() - t_start)
        end
        flush_batch!(batch_n)

        n_cov = count(covered)
        n_tot = nx * ny * nz
        n_cov == n_tot ||
            @warn "Incomplete volume coverage" covered=n_cov total=n_tot
        @printf("  Inference finished in %.1fs  coverage=%d/%d voxels\n",
                time() - t_start, n_cov, n_tot)
        return m0, m_min, m_max
    end
end

# ── Range check & persistence ───────────────────────────────────────────────

function check_geological_range(m0::Array{Float32,3};
                                lo::Float32=RES_GEO_MIN,
                                hi::Float32=RES_GEO_MAX)
    mn, mx = extrema(m0)
    μ = mean(m0)
    println("  m0 statistics: min=$(round(mn; digits=4))  max=$(round(mx; digits=4))  mean=$(round(μ; digits=4)) Ω·m")
    if mn >= lo && mx <= hi
        println("  ✓ m0 within geological range [$(lo), $(hi)] Ω·m")
    else
        println("  ⚠ m0 extends outside [$(lo), $(hi)] Ω·m")
    end
    return mn, mx, μ
end

function save_results_h5(path::AbstractString;
                         m_prior::Array{Float32,3},
                         m_min::Array{Float32,3},
                         m_max::Array{Float32,3},
                         grid_full::VoxGridSpecs.GridSpec,
                         run_params::NamedTuple)
    mkpath(dirname(path))
    h5open(path, "w") do f
        write(f, "m_prior", m_prior)
        write(f, "m0", m_prior)
        write(f, "m_min", m_min)
        write(f, "m_max", m_max)
        write(f, "bounds", cat(m_min, m_max; dims=4))
        attrs = HDF5.attributes(f)
        attrs["units/resistivity"] = "ohm.m"
        attrs["prior_domain"] = "resistivity"
        attrs["channels"] = "resistivity,vlf_resistivity,slingram_real,ip_chargeability"
        for (k, v) in pairs(run_params)
            attrs["run/" * string(k)] = v
        end
        attrs["grid_full/xmin"] = grid_full.xmin
        attrs["grid_full/xmax"] = grid_full.xmax
        attrs["grid_full/ymin"] = grid_full.ymin
        attrs["grid_full/ymax"] = grid_full.ymax
        attrs["grid_full/zmin"] = grid_full.zmin
        attrs["grid_full/zmax"] = grid_full.zmax
        attrs["grid_full/dx"] = grid_full.dx
        attrs["grid_full/dy"] = grid_full.dy
        attrs["grid_full/dz"] = grid_full.dz
        attrs["grid_full/epsg"] = grid_full.epsg_code
    end
    println("  Saved: $path")
end

# ── Pipeline ────────────────────────────────────────────────────────────────

function run_pipeline()
    pipeline_start = time()
    use_gpu = cuda_available(verbose=true)
    if !use_gpu
        cuda_diagnostics()
    end

    println("═══════════════════════════════════════════════════════════════")
    println(" Smart Prior — EM / IP / resistivity")
    println(" Preprocessed data: $PREPROCESSED_H5")
    println(" Model checkpoint:    $MODEL_CHECKPOINT")
    println("  Device: ", device_label(use_gpu))
    println("═══════════════════════════════════════════════════════════════")

    isfile(PREPROCESSED_H5) ||
        error("Preprocessed dataset not found: $(PREPROCESSED_H5)")

    # ── Step 1: Load trained prior ───────────────────────────────────────────
    println("\n[Step 1/3] Loading trained UnifiedPriorUNet3D checkpoint")
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
    println("\n[Step 2/3] Tiled resistivity prior inference (memory-safe full grid)")
    local m0_full::Array{Float32,3}
    local m_min_full::Array{Float32,3}
    local m_max_full::Array{Float32,3}
    local grid_full::VoxGridSpecs.GridSpec
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
        @printf("  Bounds: m_min∈[%.4g, %.4g]  m_max∈[%.4g, %.4g] Ω·m\n",
                minimum(m_min_full), maximum(m_min_full),
                minimum(m_max_full), maximum(m_max_full))
    end
    @printf("  Step 2 done in %.1fs\n", step2_t)

    # ── Step 3: HDF5 export ─────────────────────────────────────────────────
    println("\n[Step 3/3] Saving resistivity prior to HDF5")
    step3_t = @elapsed begin
        run_params = (
            checkpoint=basename(MODEL_CHECKPOINT),
            dataset_key=dataset_key,
            in_channels=in_channels,
            patch_px=meta.cfg !== nothing ? meta.cfg.px : 32,
            patch_py=meta.cfg !== nothing ? meta.cfg.py : 32,
            patch_pz=meta.cfg !== nothing ? meta.cfg.pz : 16,
            train_epoch=meta.epoch,
            train_best_loss=meta.best_loss,
        )
        save_results_h5(OUTPUT_H5;
                        m_prior=m0_full,
                        m_min=m_min_full,
                        m_max=m_max_full,
                        grid_full=grid_full,
                        run_params=run_params)
    end
    @printf("  Step 3 done in %.1fs\n", step3_t)

    total_t = time() - pipeline_start
    println("\n═══════════════════════════════════════════════════════════════")
    @printf(" Pipeline complete in %.1fs\n", total_t)
    println(" Output: $OUTPUT_H5")
    println("═══════════════════════════════════════════════════════════════")

    return (; grid_full, m0_full, m_min_full, m_max_full)
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
