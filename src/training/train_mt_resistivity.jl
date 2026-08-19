#!/usr/bin/env julia
#=
2-D MT resistivity prior training — MTResistivityUNet2D on synthetic pairs.

Loss:
    L_total = λ_data · L1(predicted, true_log10ρ)
            + λ_tv  · TotalVariation(predicted)

Usage (from project root):
    julia --project=. src/training/train_mt_resistivity.jl
    julia --project=. src/training/train_mt_resistivity.jl --dataset data/synthetic/train_pairs.h5 --epochs 80
=#

using Pkg
include(joinpath(@__DIR__, "..", "..", "src", "pkg_setup.jl"))
activate_project!(joinpath(@__DIR__, "..", ".."))

try
    using LuxCUDA
catch err
    @info "LuxCUDA unavailable; training will use CPU" exception=err
end

using Random
using Printf
using Statistics
using HDF5
using JLD2
using Lux
using Zygote
using Optimisers

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "gpu_utils.jl"))
GPUUtils.init_cuda!()
include(joinpath(ROOT, "src", "synthetic", "MeshParams.jl"))
include(joinpath(ROOT, "src", "networks", "mt_resistivity_unet2d.jl"))
include(joinpath(ROOT, "src", "training", "MTPriorLoss.jl"))

using .GPUUtils: cuda_available, to_device, device_label, cuda_diagnostics, gpu_forward
using .MTMeshParams: MeshParams, DEFAULT_MESH, n_periods, load_mesh_params, validate_mesh_params
using .MTResistivityUNet2DLayers: MTResistivityUNet2D, replace_nan
using .MTPriorLoss: mt_prior_loss_terms

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_DATASET = joinpath(ROOT, "data", "synthetic", "train_pairs.h5")
const DEFAULT_MODEL_OUT = joinpath(ROOT, "models", "best_mt_resistivity_prior.jld2")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct MTTrainConfig
    dataset::String       = DEFAULT_DATASET
    output::String        = DEFAULT_MODEL_OUT
    epochs::Int           = 50
    lr::Float64           = 1.0e-3
    batch_size::Int       = 8
    lambda_data::Float32  = 1.0f0
    lambda_tv::Float32    = 1.0f-3
    base_channels::Int    = 16
    seed::Int             = 42
end

function parse_cli_args(args::Vector{String})::MTTrainConfig
    cfg = MTTrainConfig()
    dataset = cfg.dataset
    output  = cfg.output
    epochs  = cfg.epochs
    lr      = cfg.lr
    batch_size = cfg.batch_size
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--dataset" && i + 1 <= length(args)
            dataset = args[i+1]; i += 2; continue
        elseif arg == "--output" && i + 1 <= length(args)
            output = args[i+1]; i += 2; continue
        elseif arg == "--epochs" && i + 1 <= length(args)
            epochs = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--lr" && i + 1 <= length(args)
            lr = parse(Float64, args[i+1]); i += 2; continue
        elseif arg == "--batch-size" && i + 1 <= length(args)
            batch_size = parse(Int, args[i+1]); i += 2; continue
        end
        i += 1
    end
    return MTTrainConfig(; dataset, output, epochs, lr, batch_size)
end

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

"""
Load `train_pairs.h5` and return `(X, Y, mesh_params)`.

- `X`: `(n_stations, n_periods, n_components, N)` Float32 MT responses
- `Y`: `(nz, nx, N)` Float32 log10-resistivity ground truth
- `mesh_params`: [`MeshParams`](@ref) stored as HDF5 attributes
"""
function load_dataset(path::AbstractString)
    isfile(path) || error("Dataset not found: $path")
    h5open(path, "r") do f
        haskey(f, "X") || error("Dataset missing 'X' key")
        haskey(f, "Y") || error("Dataset missing 'Y' key")

        X = Float32.(read(f["X"]))
        Y = Float32.(read(f["Y"]))

        a = HDF5.attributes(f)
        if haskey(a, "mesh_params_json")
            mp_json = read(a["mesh_params_json"])
            # Deserialize via MeshParams constructor
            mp = _parse_mesh_attrs(a)
        elseif haskey(a, "nx")
            mp = _parse_mesh_attrs(a)
        else
            @warn "No MeshParams in dataset; using DEFAULT_MESH"
            mp = DEFAULT_MESH
        end
        return X, Y, mp
    end
end

function _parse_mesh_attrs(a)
    nx = haskey(a, "nx") ? Int(read(a["nx"])) : DEFAULT_MESH.nx
    nz = haskey(a, "nz") ? Int(read(a["nz"])) : DEFAULT_MESH.nz
    dx = haskey(a, "dx") ? Float64(read(a["dx"])) : DEFAULT_MESH.dx
    dz = haskey(a, "dz") ? Float64(read(a["dz"])) : DEFAULT_MESH.dz
    n_stations = haskey(a, "n_stations") ? Int(read(a["n_stations"])) : DEFAULT_MESH.n_stations
    periods = if haskey(a, "periods")
        Float64.(read(a["periods"]))
    else
        DEFAULT_MESH.periods
    end
    return MeshParams(nx, nz, dx, dz, n_stations, periods)
end

"""Sample a random mini-batch from the dataset arrays."""
function sample_batch(rng::AbstractRNG, X::AbstractArray{Float32,4},
                      Y::AbstractArray{Float32,3}, batch_size::Int)
    N = size(X, 4)
    idx = rand(rng, 1:N, batch_size)
    return X[:, :, :, idx], Y[:, :, idx]
end

# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

function train_step!(model::MTResistivityUNet2D, ps, st, opt_state,
                     x_batch::AbstractArray{Float32,4},
                     y_batch::AbstractArray{Float32,3},
                     use_gpu::Bool;
                     λ_data::Float32, λ_tv::Float32)
    x = to_device(x_batch, Val(use_gpu))
    y = to_device(y_batch, Val(use_gpu))

    loss_and_grad = Zygote.withgradient(ps) do p
        gpu_forward(use_gpu) do
            pred, _ = model(x, p, st)
            mt_prior_loss_terms(pred, y; λ_data=λ_data, λ_tv=λ_tv).total
        end
    end

    pred, st_new = gpu_forward(use_gpu) do
        model(x, ps, st)
    end
    terms = mt_prior_loss_terms(pred, y; λ_data=λ_data, λ_tv=λ_tv)
    Optimisers.update!(opt_state, ps, loss_and_grad.grad[1])
    return terms, st_new
end

# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

function save_checkpoint(path::AbstractString, model, ps, st,
                         cfg::MTTrainConfig, mesh_params::MeshParams,
                         epoch::Int, best_loss::Float32)
    mkpath(dirname(abspath(path)))
    jldsave(path; model, ps, st, cfg, mesh_params, epoch, best_loss)
end

# ─────────────────────────────────────────────────────────────────────────────
# COMMEMI benchmark evaluation
# ─────────────────────────────────────────────────────────────────────────────

"""
    evaluate_on_commemi(checkpoint_path; verbose=true) -> NamedTuple

Load a trained `MTResistivityUNet2D` checkpoint and evaluate on the COMMEMI 2-D
benchmark from `MTGeophysics.jl` (`helpers/benchmarks_2D.jl`).

Returns `(; rmse, ssim)` of predicted vs reference log10-resistivity.
"""
function evaluate_on_commemi(checkpoint_path::AbstractString; verbose::Bool=true)
    isfile(checkpoint_path) || error("Checkpoint not found: $checkpoint_path")
    ckpt = load(checkpoint_path)
    model = ckpt["model"]
    ps    = ckpt["ps"]
    st    = ckpt["st"]
    mesh  = ckpt["mesh_params"]

    # Try loading MTGeophysics benchmark
    local bench_module
    try
        @eval using MTGeophysics
        bench_module = MTGeophysics
    catch
        @warn "MTGeophysics.jl not available; attempting local helpers path"
        bench_path = joinpath(ROOT, "deps", "MTGeophysics", "helpers", "benchmarks_2D.jl")
        if isfile(bench_path)
            include(bench_path)
        else
            error("Cannot locate MTGeophysics benchmarks_2D.jl")
        end
    end

    # Generate COMMEMI benchmark MT data and reference model
    bench_data = if isdefined(bench_module, :commemi_2d_benchmark)
        bench_module.commemi_2d_benchmark(; mesh=mesh)
    else
        error("commemi_2d_benchmark not found in MTGeophysics")
    end

    x_mt   = Float32.(bench_data.mt_data)       # (n_stations, n_periods, 2)
    y_true = Float32.(bench_data.log10_rho_true) # (nz, nx)

    pred, _ = model(x_mt, ps, st)
    pred_f = Float32.(pred)

    rmse_val = sqrt(mean((pred_f .- y_true) .^ 2))
    ssim_val = _ssim_2d(pred_f, y_true)

    if verbose
        @printf("COMMEMI benchmark evaluation:\n")
        @printf("  RMSE (log10 Ω·m): %.4f\n", rmse_val)
        @printf("  SSIM:             %.4f\n", ssim_val)
    end
    return (; rmse=rmse_val, ssim=ssim_val)
end

"""Simplified SSIM for 2-D grids (Wang et al. 2004, single-scale, L=dynamic range)."""
function _ssim_2d(x::AbstractArray{Float32}, y::AbstractArray{Float32};
                  k1::Float32=0.01f0, k2::Float32=0.03f0)
    L = max(maximum(y) - minimum(y), 1.0f-6)
    c1 = (k1 * L)^2
    c2 = (k2 * L)^2
    μx = mean(x)
    μy = mean(y)
    σx² = mean((x .- μx) .^ 2)
    σy² = mean((y .- μy) .^ 2)
    σxy = mean((x .- μx) .* (y .- μy))
    return (2μx * μy + c1) * (2σxy + c2) / ((μx^2 + μy^2 + c1) * (σx² + σy² + c2))
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(cfg::MTTrainConfig=MTTrainConfig())
    rng = MersenneTwister(cfg.seed)
    use_gpu = cuda_available(verbose=true)
    !use_gpu && cuda_diagnostics()

    println("═" ^ 60)
    println(" MTResistivityUNet2D training — synthetic MT pairs")
    println("═" ^ 60)
    println("  Device: ", device_label(use_gpu))

    X, Y, mesh = load_dataset(cfg.dataset)
    mesh = validate_mesh_params(mesh)
    N = size(X, 4)
    n_st, n_per, n_comp = size(X, 1), size(X, 2), size(X, 3)
    @info "Dataset loaded" path=cfg.dataset N=N stations=n_st periods=n_per components=n_comp
    @info "Target grid" nz=mesh.nz nx=mesh.nx

    batches_per_epoch = max(1, N ÷ cfg.batch_size)

    model = MTResistivityUNet2D(;
        in_channels=n_comp,
        base_channels=cfg.base_channels,
        mesh=mesh,
    )
    ps, st = Lux.setup(rng, model)
    use_gpu && (ps = to_device(ps, Val(true)))

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps)

    λ_data = cfg.lambda_data
    λ_tv   = cfg.lambda_tv

    best_loss = Inf32
    best_epoch = 0

    println("\nStarting training for $(cfg.epochs) epochs  (N=$N, batch=$(cfg.batch_size))")
    @printf("  λ_data=%.2e  λ_tv=%.2e  lr=%.2e\n", λ_data, λ_tv, cfg.lr)

    for epoch in 1:cfg.epochs
        epoch_total = 0.0f0
        epoch_data  = 0.0f0
        epoch_tv    = 0.0f0
        n_seen = 0

        for _ in 1:batches_per_epoch
            xb, yb = sample_batch(rng, X, Y, cfg.batch_size)
            terms, st = train_step!(model, ps, st, opt_state, xb, yb, use_gpu;
                                    λ_data=λ_data, λ_tv=λ_tv)
            epoch_total += terms.total
            epoch_data  += terms.data
            epoch_tv    += terms.tv
            n_seen += 1
        end

        n_seen > 0 || error("no batches consumed")
        avg_total = epoch_total / n_seen
        @printf("Epoch %3d/%d  total=%.6e  L_data=%.6e  L_TV=%.6e\n",
                epoch, cfg.epochs,
                avg_total,
                epoch_data / n_seen,
                epoch_tv / n_seen)

        if avg_total < best_loss
            best_loss = avg_total
            best_epoch = epoch
            save_checkpoint(cfg.output, model, ps, st, cfg, mesh, epoch, best_loss)
            @info "New best model saved" path=cfg.output loss=best_loss epoch=epoch
        end
    end

    println("\nTraining complete.")
    @printf("Best epoch: %d  loss=%.6e  →  %s\n", best_epoch, best_loss, cfg.output)

    # COMMEMI benchmark evaluation
    println("\n── COMMEMI benchmark evaluation ──")
    try
        evaluate_on_commemi(cfg.output; verbose=true)
    catch err
        @warn "COMMEMI evaluation skipped" exception=err
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_cli_args(ARGS))
end
