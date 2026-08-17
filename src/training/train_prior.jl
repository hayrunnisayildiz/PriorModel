#!/usr/bin/env julia
#=
Patch-based UnifiedPriorUNet3D training (Prompt 3).

Combined loss:
    L_total = λ_well L_sondaj + λ_grav L_gravity + λ_TV L_TV + λ_bounds L_bounds

Usage (from project root):
    julia --project=. src/training/train_prior.jl
    julia --project=. src/training/train_prior.jl --epochs 50 --dataset data/processed/keivitsa_preprocessed.h5
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Random
using Printf
using Statistics
using HDF5
using JLD2
using Lux
using Zygote
using Optimisers

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "fusion", "GridSpec.jl"))
include(joinpath(ROOT, "src", "data", "patch_loader.jl"))
include(joinpath(ROOT, "src", "networks", "prior_unet3d.jl"))
include(joinpath(ROOT, "src", "physics_inversion", "ForwardGravity.jl"))
include(joinpath(ROOT, "src", "neural_prior", "Losses.jl"))
include(joinpath(ROOT, "src", "training", "PriorTrainingLoss.jl"))

using .PatchLoader: PatchSampler, PatchBatch, close
using .PriorUNet3DLayers: UnifiedPriorUNet3D
using .ForwardGravity: gravity_kernel_matrix
using .PriorTrainingLoss: PatchTrainContext, SupervisionChannels, patch_combined_loss_terms
using .Losses: DENSITY_CHANNEL

const GridSpec = GridSpecs.GridSpec
const FwdGridSpec = ForwardGravity.GridSpecs.GridSpec

"""Convert fusion `GridSpec` to the forward-operator copy."""
to_forward_grid(g::GridSpec)::FwdGridSpec =
    FwdGridSpec(g.xmin, g.xmax, g.ymin, g.ymax, g.zmin, g.zmax,
                g.dx, g.dy, g.dz, g.epsg_code)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_DATASET = joinpath(ROOT, "data", "processed", "keivitsa_preprocessed.h5")
const DEFAULT_MODEL_OUT = joinpath(ROOT, "models", "best_prior_model.jld2")

# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers (optional CUDA)
# ─────────────────────────────────────────────────────────────────────────────

"""Return `true` when CUDA is loaded and a functional device is available."""
function cuda_functional()::Bool
    try
        @eval import CUDA
        return CUDA.functional()
    catch
        return false
    end
end

"""Move arrays (and nested parameter trees) to GPU when requested."""
function to_device(x, ::Val{true})
    @eval import CUDA
    return Lux.recursive_map(CUDA.cu, x)
end
to_device(x, ::Val{false}) = x

# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

"""Load grid metadata stored as HDF5 root attributes."""
function load_grid_from_h5(path::AbstractString)::GridSpec
    h5open(path, "r") do f
        a = attributes(f)
        return GridSpec(
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

"""Load 2-D surface gravity `(nx, ny, 1)` from preprocessed `X` if present."""
function load_surface_gravity(path::AbstractString;
                              gravity_ch::Int=1)::Union{Array{Float32,3},Nothing}
    h5open(path, "r") do f
        haskey(f, "X") || return nothing
        X = read(f, "X")
        ndims(X) == 3 || return nothing
        gravity_ch <= size(X, 3) || return nothing
        return reshape(X[:, :, gravity_ch], size(X, 1), size(X, 2), 1)
    end
end

"""Resolve supervision channel indices from dataset key and HDF5 metadata."""
function supervision_channels(path::AbstractString, dataset_key::AbstractString,
                              nc::Int)::SupervisionChannels
    if dataset_key == "Y"
        return SupervisionChannels(1, 3, nothing, :kg_m3)
    elseif dataset_key == "fusion"
        return SupervisionChannels(DENSITY_CHANNEL, DENSITY_CHANNEL, 1, :kg_m3)
    end
    h5open(path, "r") do f
        if haskey(f, "Y") || occursin("target", lowercase(dataset_key))
            return SupervisionChannels(min(1, nc), min(3, nc), nothing, :kg_m3)
        end
    end
    return SupervisionChannels(min(1, nc), min(1, nc), 1, :kg_m3)
end

"""Build a fixed patch-local Nagy prism kernel (one centre-top station)."""
function build_patch_gravity_operator(px::Int, py::Int, pz::Int,
                                      dx::Float32, dy::Float32, dz::Float32)
    # Local frame: surface at z = 0, subsurface negative (metres, z up).
    grid = to_forward_grid(GridSpec(0.0f0, px * dx, 0.0f0, py * dy, -pz * dz, 0.0f0,
                                    dx, dy, dz, 0))
    ox = Float32[px * dx / 2]
    oy = Float32[py * dy / 2]
    oz = Float32[0.0f0]
    G = gravity_kernel_matrix(grid, ox, oy, oz)
    return G
end

function build_patch_train_context(path::AbstractString, dataset_key::AbstractString, nc::Int,
                                   px::Int, py::Int, pz::Int)::PatchTrainContext
    grid = load_grid_from_h5(path)
    ch = supervision_channels(path, dataset_key, nc)
    sg = load_surface_gravity(path; gravity_ch= something(ch.gravity_surface, 1))
    G = build_patch_gravity_operator(px, py, pz, grid.dx, grid.dy, grid.dz)
    return PatchTrainContext(G, sg, ch)
end

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

"""Training hyper-parameters."""
Base.@kwdef struct TrainConfig
    dataset::String = DEFAULT_DATASET
    output::String = DEFAULT_MODEL_OUT
    dataset_key::String = "fusion"
    px::Int = 32
    py::Int = 32
    pz::Int = 16
    batch_size::Int = 4
    batches_per_epoch::Int = 32
    epochs::Int = 30
    lr::Float64 = 1.0e-3
    lambda_well::Float32 = 1.0f0
    lambda_grav::Float32 = 1.0f-2
    lambda_tv::Float32 = 1.0f-3
    lambda_bounds::Float32 = 1.0f-4
    base_channels::Int = 16
    seed::Int = 42
end

function parse_cli_args(args::Vector{String})::TrainConfig
    cfg = TrainConfig()
    dataset = cfg.dataset
    output = cfg.output
    epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    batches_per_epoch = cfg.batches_per_epoch
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
        elseif arg == "--batches-per-epoch" && i + 1 <= length(args)
            batches_per_epoch = parse(Int, args[i+1]); i += 2; continue
        end
        i += 1
    end
    return TrainConfig(; dataset, output, epochs, lr, batch_size, batches_per_epoch)
end

# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

"""One mini-batch forward + backward pass."""
function train_step!(model::UnifiedPriorUNet3D, ps, st, opt_state,
                     batch::PatchBatch, ctx::PatchTrainContext, use_gpu::Bool;
                     λ_well::Float32, λ_grav::Float32, λ_tv::Float32, λ_bounds::Float32)
    x = to_device(batch.data, Val(use_gpu))

    loss_and_grad = Zygote.withgradient(ps) do p
        vol, _ = model(x, p, st)
        patch_combined_loss_terms(vol, x, batch.origins, ctx;
                                  λ_well=λ_well, λ_grav=λ_grav,
                                  λ_tv=λ_tv, λ_bounds=λ_bounds).total
    end

    vol, st_new = model(x, ps, st)
    terms = patch_combined_loss_terms(vol, x, batch.origins, ctx;
                                      λ_well=λ_well, λ_grav=λ_grav,
                                      λ_tv=λ_tv, λ_bounds=λ_bounds)
    Optimisers.update!(opt_state, ps, loss_and_grad.grad[1])
    return terms, st_new
end

# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

function save_checkpoint(path::AbstractString, model, ps, st, cfg::TrainConfig,
                         dataset_key::AbstractString, in_channels::Int,
                         epoch::Int, best_loss::Float32)
    mkpath(dirname(abspath(path)))
    jldsave(path; model, ps, st, cfg, dataset_key, in_channels, epoch, best_loss)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(cfg::TrainConfig=TrainConfig())
    rng = MersenneTwister(cfg.seed)
    use_gpu = cuda_functional()

    println("═" ^ 60)
    println(" Patch-based UnifiedPriorUNet3D training")
    println("═" ^ 60)
    println(use_gpu ? "  Device: CUDA GPU" : "  Device: CPU")

    dataset_key, vol_shape = resolve_dataset_key(cfg.dataset; preferred=cfg.dataset_key)
    nc = vol_shape[4]
    in_channels = nc + 1   # PatchLoader appends validity mask channel
    @info "Dataset resolved" path=cfg.dataset key=dataset_key volume=vol_shape in_channels

    ctx = build_patch_train_context(cfg.dataset, dataset_key, nc,
                                    cfg.px, cfg.py, cfg.pz)
    ctx.surface_gravity === nothing &&
        @warn "No surface gravity map (X); L_gravity will be zero"

    sampler = PatchSampler(cfg.dataset;
                           dataset_key=dataset_key,
                           px=cfg.px, py=cfg.py, pz=cfg.pz,
                           batch_size=cfg.batch_size,
                           n_batches=cfg.batches_per_epoch,
                           rng=rng)

    model = UnifiedPriorUNet3D(; in_channels=in_channels, base_channels=cfg.base_channels)
    ps, st = Lux.setup(rng, model)
    use_gpu && (ps = to_device(ps, Val(true)))

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps)

    λ_well    = cfg.lambda_well
    λ_grav    = cfg.lambda_grav
    λ_tv      = cfg.lambda_tv
    λ_bounds  = cfg.lambda_bounds

    best_loss = Inf32
    best_epoch = 0

    println("\nStarting training for $(cfg.epochs) epochs")
    @printf("  patch=(%d,%d,%d)  batch=%d  batches/epoch=%d\n",
            cfg.px, cfg.py, cfg.pz, cfg.batch_size, cfg.batches_per_epoch)
    @printf("  λ_well=%.2e  λ_grav=%.2e  λ_tv=%.2e  λ_bounds=%.2e  lr=%.2e\n",
            λ_well, λ_grav, λ_tv, λ_bounds, cfg.lr)

    try
        for epoch in 1:cfg.epochs
            epoch_total = 0.0f0
            epoch_well = 0.0f0
            epoch_grav = 0.0f0
            epoch_tv = 0.0f0
            epoch_bounds = 0.0f0
            n_seen = 0

            for batch in sampler
                terms, st = train_step!(model, ps, st, opt_state, batch, ctx, use_gpu;
                                        λ_well=λ_well, λ_grav=λ_grav,
                                        λ_tv=λ_tv, λ_bounds=λ_bounds)
                epoch_total += terms.total
                epoch_well += terms.well
                epoch_grav += terms.gravity
                epoch_tv += terms.tv
                epoch_bounds += terms.bounds
                n_seen += 1
            end

            n_seen > 0 || error("no batches consumed — check PatchSampler / dataset")
            avg_total = epoch_total / n_seen
            @printf("Epoch %3d/%d  total=%.6e  L_sondaj=%.6e  L_gravity=%.6e  L_TV=%.6e  L_bounds=%.6e\n",
                    epoch, cfg.epochs,
                    avg_total,
                    epoch_well / n_seen,
                    epoch_grav / n_seen,
                    epoch_tv / n_seen,
                    epoch_bounds / n_seen)

            if avg_total < best_loss
                best_loss = avg_total
                best_epoch = epoch
                save_checkpoint(cfg.output, model, ps, st, cfg, dataset_key, in_channels,
                                epoch, best_loss)
                @info "New best model saved" path=cfg.output loss=best_loss epoch=epoch
            end
        end
    finally
        close(sampler)
    end

    println("\nTraining complete.")
    @printf("Best epoch: %d  loss=%.6e  →  %s\n", best_epoch, best_loss, cfg.output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_cli_args(ARGS))
end
