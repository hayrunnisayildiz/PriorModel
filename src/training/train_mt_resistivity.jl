#!/usr/bin/env julia
#=
2-D MT resistivity prior training — MTResistivityUNet2D on synthetic pairs.

Loss:
    L_total = λ_data · L1(predicted, true_log10ρ)
            + λ_tv  · TotalVariation(predicted)

Usage (from project root):
    julia --project=. src/training/train_mt_resistivity.jl
    julia --project=. src/training/train_mt_resistivity.jl --dataset data/synthetic/train_pairs.h5 --epochs 80
    julia --project=. src/training/train_mt_resistivity.jl --resume-from models/mid_scale_prior_v5.jld2 --epochs 60
    julia --project=. src/training/train_mt_resistivity.jl --dataset data/synthetic/train_pairs_v7.h5 --epochs 50 --output models/production_prior_v7.jld2

Colab (cwd is /content — never `--project=.` from there):
    julia --project=/content/PriorModel /content/PriorModel/src/training/train_mt_resistivity.jl \
        --dataset /content/PriorModel/data/synthetic/train_pairs.h5 \
        --commemi-every 0 --no-plot

Checkpoint selection: COMMEMI short-VFSA `commemi_rms` is primary (probed every
`--commemi-every` epochs; default 10, `0` disables the probe). Synthetic
`val_loss` is the fallback / tie-breaker. `--no-plot` skips the training-curve
PNG (Plots.jl subprocess; not GLMakie). The probe loads MTGeophysics lazily
and stubs GLMakie so OpenGL is never required (CairoMakie is the VFSA backend).
=#

include(joinpath(@__DIR__, "..", "pkg_setup.jl"))

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
using JSON3
using Lux
using Zygote
using Optimisers
using DelimitedFiles

include(joinpath(ROOT, "src", "gpu_utils.jl"))
GPUUtils.init_cuda!()
include(joinpath(ROOT, "src", "networks", "mt_resistivity_unet2d.jl"))
include(joinpath(ROOT, "src", "training", "MTPriorLoss.jl"))
include(joinpath(ROOT, "src", "synthetic", "MTInputStandardizer.jl"))
include(joinpath(ROOT, "src", "training", "commemi_probe.jl"))

using .GPUUtils: cuda_available, to_device, device_label, cuda_diagnostics, gpu_forward
using .MTResistivityUNet2DLayers: MTResistivityUNet2D, replace_nan, count_parameters, report_capacity_change
using .MTResistivityUNet2DLayers.MTMeshParams: MeshParams, DEFAULT_MESH, n_periods, load_mesh_params, validate_mesh_params
using .MTPriorLoss: mt_prior_loss_terms
using .MTInputStandardizer: standardize_mt_input
using .CommemiProbe: load_commemi_mt, probe_commemi_rms,
                     should_save_checkpoint, should_probe_epoch, checkpoint_reason

"""Colab `%%bash` block-buffers stdout; flush so epoch lines appear immediately."""
function _say(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (`ROOT` comes from src/pkg_setup.jl)
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
    lr::Float64           = 2.0e-4          # was 1e-3; 1/5 for stability
    batch_size::Int       = 16              # 144 train samples → 9 full batches
    clip_norm::Float64    = 1.0
    train_frac::Float64   = 0.8
    lambda_data::Float32  = 1.0f0
    lambda_tv::Float32    = 1.0f-3
    base_channels::Int    = 32
    n_down::Int           = 3
    seed::Int             = 42
    training_log::String  = joinpath(ROOT, "results", "training_log.csv")
    split_json::String    = joinpath(ROOT, "results", "train_val_split.json")
    curve_png::String     = joinpath(ROOT, "results", "training_curve.png")
    resume_from::String   = ""              # JLD2 checkpoint to continue from
    resume_log::String    = ""              # previous CSV prepended to the combined curve
    schedule_epochs::Int  = 0               # cosine horizon; 0 → epochs (or ckpt cfg.epochs on resume)
    commemi_every::Int    = 10              # probe COMMEMI VFSA every N epochs (0 = off)
    commemi_iter::Int     = 25              # short VFSA iterations (full eval uses 100)
    commemi_obs::String   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    no_plot::Bool         = false           # skip training-curve PNG (Colab / headless)
end

function parse_cli_args(args::Vector{String})::MTTrainConfig
    cfg = MTTrainConfig()
    dataset = cfg.dataset
    output  = cfg.output
    epochs  = cfg.epochs
    lr      = cfg.lr
    batch_size = cfg.batch_size
    clip_norm = cfg.clip_norm
    base_channels = cfg.base_channels
    n_down = cfg.n_down
    training_log = cfg.training_log
    split_json = cfg.split_json
    curve_png = cfg.curve_png
    resume_from = cfg.resume_from
    resume_log = cfg.resume_log
    schedule_epochs = cfg.schedule_epochs
    commemi_every = cfg.commemi_every
    commemi_iter = cfg.commemi_iter
    commemi_obs = cfg.commemi_obs
    no_plot = cfg.no_plot
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
        elseif arg == "--clip-norm" && i + 1 <= length(args)
            clip_norm = parse(Float64, args[i+1]); i += 2; continue
        elseif arg == "--base-channels" && i + 1 <= length(args)
            base_channels = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--n-down" && i + 1 <= length(args)
            n_down = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--training-log" && i + 1 <= length(args)
            training_log = args[i+1]; i += 2; continue
        elseif arg == "--split-json" && i + 1 <= length(args)
            split_json = args[i+1]; i += 2; continue
        elseif arg == "--curve-png" && i + 1 <= length(args)
            curve_png = args[i+1]; i += 2; continue
        elseif arg == "--resume-from" && i + 1 <= length(args)
            resume_from = args[i+1]; i += 2; continue
        elseif arg == "--resume-log" && i + 1 <= length(args)
            resume_log = args[i+1]; i += 2; continue
        elseif arg == "--schedule-epochs" && i + 1 <= length(args)
            schedule_epochs = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--commemi-every" && i + 1 <= length(args)
            commemi_every = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--commemi-iter" && i + 1 <= length(args)
            commemi_iter = parse(Int, args[i+1]); i += 2; continue
        elseif arg == "--commemi-obs" && i + 1 <= length(args)
            commemi_obs = args[i+1]; i += 2; continue
        elseif arg == "--no-plot"
            no_plot = true; i += 1; continue
        end
        i += 1
    end
    return MTTrainConfig(; dataset, output, epochs, lr, batch_size, clip_norm,
                         base_channels, n_down, training_log, split_json, curve_png,
                         resume_from, resume_log, schedule_epochs,
                         commemi_every, commemi_iter, commemi_obs, no_plot)
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

"""
Random but reproducible 80/20 train/validation split.

Returns sorted-by-assignment `train_idx` / `val_idx` in shuffle order so the
first few entries are the first assigned samples (must be disjoint).
"""
function split_train_val(n::Int, rng::AbstractRNG; train_frac::Float64=0.8)
    n >= 2 || error("need at least 2 samples to split, got N=$n")
    0.0 < train_frac < 1.0 || error("train_frac must be in (0,1), got $train_frac")
    perm = shuffle(rng, collect(1:n))
    n_train = round(Int, train_frac * n)
    n_train = clamp(n_train, 1, n - 1)
    train_idx = perm[1:n_train]
    val_idx   = perm[(n_train + 1):end]
    overlap = intersect(Set(train_idx), Set(val_idx))
    isempty(overlap) || error("train/val overlap: $overlap")
    return train_idx, val_idx
end

function save_train_val_split(path::AbstractString, seed::Int,
                              train_idx::Vector{Int}, val_idx::Vector{Int})
    mkpath(dirname(abspath(path)))
    doc = Dict{String,Any}(
        "seed"      => seed,
        "n_total"   => length(train_idx) + length(val_idx),
        "n_train"   => length(train_idx),
        "n_val"     => length(val_idx),
        "train_frac"=> length(train_idx) / (length(train_idx) + length(val_idx)),
        "train_idx" => train_idx,
        "val_idx"   => val_idx,
    )
    open(path, "w") do io
        JSON3.pretty(io, doc)
    end
    return path
end

function load_train_val_split(path::AbstractString)
    isfile(path) || error("train/val split not found: $path")
    doc = JSON3.read(read(path, String))
    train_idx = Int.(collect(doc["train_idx"]))
    val_idx   = Int.(collect(doc["val_idx"]))
    overlap = intersect(Set(train_idx), Set(val_idx))
    isempty(overlap) || error("loaded train/val overlap: $overlap")
    return train_idx, val_idx
end

"""Gather a mini-batch by explicit sample indices (1-based, dataset axis 4 / 3)."""
function sample_batch(X::AbstractArray{Float32,4}, Y::AbstractArray{Float32,3},
                      idx::AbstractVector{Int})
    return X[:, :, :, idx], Y[:, :, idx]
end

"""Shuffle `indices` and split into mini-batches (last batch may be shorter)."""
function epoch_minibatches(rng::AbstractRNG, indices::Vector{Int}, batch_size::Int)
    shuffled = shuffle(rng, indices)
    batches = Vector{Vector{Int}}()
    for i in 1:batch_size:length(shuffled)
        push!(batches, shuffled[i:min(i + batch_size - 1, length(shuffled))])
    end
    return batches
end

"""Cosine annealing from `lr_max` (epoch 1) to `lr_min` (final epoch).

`t` is clamped to `[0, 1]` so resuming past the original horizon stays at
`lr_min` instead of restarting the cosine.
"""
function cosine_annealing_lr(epoch::Int, n_epochs::Int, lr_max::Float64;
                             lr_min::Float64=lr_max * 0.05)
    n_epochs < 1 && error("n_epochs must be ≥ 1")
    t = clamp((epoch - 1) / max(n_epochs - 1, 1), 0.0, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t))
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
    st_train = Lux.trainmode(st)

    loss_and_grad = Zygote.withgradient(ps) do p
        gpu_forward(use_gpu) do
            pred, _ = model(x, p, st_train)
            mt_prior_loss_terms(pred, y; λ_data=λ_data, λ_tv=λ_tv).total
        end
    end

    pred, st_new = gpu_forward(use_gpu) do
        model(x, ps, st_train)
    end
    terms = mt_prior_loss_terms(pred, y; λ_data=λ_data, λ_tv=λ_tv)
    Optimisers.update!(opt_state, ps, loss_and_grad.grad[1])
    return terms, st_new
end

"""
Forward-only loss on a held-out index set. Never used for gradient updates.

Uses `Lux.testmode` so dropout / batch-norm (if present) run in eval mode.
"""
function eval_split_loss(model::MTResistivityUNet2D, ps, st,
                         X::AbstractArray{Float32,4}, Y::AbstractArray{Float32,3},
                         indices::Vector{Int}, batch_size::Int, use_gpu::Bool;
                         λ_data::Float32, λ_tv::Float32)::Float32
    st_eval = Lux.testmode(st)
    acc = 0.0f0
    n = 0
    i = 1
    n_idx = length(indices)
    while i <= n_idx
        j = min(i + batch_size - 1, n_idx)
        idx = indices[i:j]
        xb, yb = sample_batch(X, Y, idx)
        x = to_device(xb, Val(use_gpu))
        y = to_device(yb, Val(use_gpu))
        pred, _ = gpu_forward(use_gpu) do
            model(x, ps, st_eval)
        end
        terms = mt_prior_loss_terms(pred, y; λ_data=λ_data, λ_tv=λ_tv)
        bsz = length(idx)
        acc += Float32(terms.total) * Float32(bsz)
        n += bsz
        i = j + 1
    end
    n > 0 || error("eval_split_loss received an empty index set")
    return acc / Float32(n)
end

function write_training_log(path::AbstractString, epochs::Vector{Int},
                            train_losses::Vector{Float32}, val_losses::Vector{Float32},
                            commemi_rms::Vector{Float64}, best_epoch::Int)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "epoch,train_loss,val_loss,commemi_rms,is_best")
        for i in eachindex(epochs)
            flag = epochs[i] == best_epoch ? 1 : 0
            r = commemi_rms[i]
            if isfinite(r)
                @printf(io, "%d,%.8e,%.8e,%.8e,%d\n",
                        epochs[i], train_losses[i], val_losses[i], r, flag)
            else
                @printf(io, "%d,%.8e,%.8e,NaN,%d\n",
                        epochs[i], train_losses[i], val_losses[i], flag)
            end
        end
    end
    return path
end

function load_training_log(path::AbstractString)
    isfile(path) || error("training log not found: $path")
    data = readdlm(path, ','; header=true)
    vals = data[1]
    size(vals, 1) >= 1 || error("empty training log: $path")
    epochs = Int.(vals[:, 1])
    train  = Float32.(vals[:, 2])
    val    = Float32.(vals[:, 3])
    ncols = size(vals, 2)
    rms = if ncols >= 5
        Float64.(vals[:, 4])
    else
        fill(NaN, length(epochs))
    end
    flags = if ncols >= 5
        Int.(vals[:, 5])
    elseif ncols >= 4
        Int.(vals[:, 4])
    else
        zeros(Int, length(epochs))
    end
    best_i = findfirst(==(1), flags)
    best_epoch = best_i === nothing ? 0 : epochs[best_i]
    return epochs, train, val, rms, best_epoch
end

function plot_training_curve(csv_path::AbstractString, png_path::AbstractString;
                             resume_epoch::Int=0)
    isfile(csv_path) || error("training log not found: $csv_path")
    mkpath(dirname(abspath(png_path)))

    plot_code = raw"""
        using DelimitedFiles, Plots
        csv_path = ARGS[1]
        png_path = ARGS[2]
        resume_epoch = parse(Int, ARGS[3])
        data = readdlm(csv_path, ','; header=true)
        vals = data[1]
        epochs     = Int.(vals[:, 1])
        train_loss = Float64.(vals[:, 2])
        val_loss   = Float64.(vals[:, 3])
        ncols = size(vals, 2)
        rms = ncols >= 5 ? Float64.(vals[:, 4]) : fill(NaN, length(epochs))
        flags = if ncols >= 5
            Int.(vals[:, 5])
        elseif ncols >= 4
            Int.(vals[:, 4])
        else
            zeros(Int, length(epochs))
        end

        p = plot(epochs, train_loss;
            label="train_loss", lw=2, marker=:circle, ms=3, color=:steelblue,
            xlabel="Epoch", ylabel="Loss",
            title="Training curve (val_loss + COMMEMI RMS)",
            legend=:topleft, size=(960, 460), dpi=150,
            left_margin=8Plots.mm, right_margin=12Plots.mm)
        plot!(p, epochs, val_loss;
            label="val_loss", lw=2, marker=:square, ms=3, ls=:dash, color=:darkorange)

        ok = findall(isfinite, rms)
        if !isempty(ok)
            p2 = twinx(p)
            plot!(p2, epochs[ok], rms[ok];
                label="commemi_rms", lw=2, marker=:diamond, ms=5,
                color=:firebrick, ylabel="COMMEMI VFSA best RMS")
            Plots.plot!(p2; legend=:topright)
        end

        if resume_epoch > 0
            vline!(p, [resume_epoch]; color=:purple, ls=:dash, lw=1.8,
                   label="resume (epoch $(resume_epoch))")
        end
        best_i = findfirst(==(1), flags)
        if best_i !== nothing
            be = epochs[best_i]
            vline!(p, [be]; color=:gray, ls=:dot, lw=1.2,
                   label="best ckpt (epoch $(be))")
            scatter!(p, [be], [val_loss[best_i]];
                     marker=:star, ms=10, color=:orange, label="")
        end
        savefig(p, png_path)
        println(png_path)
    """

    plot_env = copy(ENV)
    delete!(plot_env, "JULIA_PROJECT")
    try
        run(setenv(`julia --startup-file=no -e $plot_code $csv_path $png_path $(string(resume_epoch))`,
                   plot_env))
        @info "Training curve saved" path=png_path
        return png_path
    catch err
        @warn "Could not render training_curve.png" exception=err
        return nothing
    end
end

"""Linear slope of `y` vs 1-based index; used for late-training saturation checks."""
function _finite_slope(y::AbstractVector{<:Real})
    n = length(y)
    n < 2 && return 0.0
    x_mean = (n + 1) / 2
    y_mean = sum(Float64.(y)) / n
    num = 0.0
    den = 0.0
    for i in 1:n
        dx = i - x_mean
        num += dx * (Float64(y[i]) - y_mean)
        den += dx * dx
    end
    return num / max(den, 1e-12)
end

"""
    report_saturation(epochs, val_losses; resume_epoch, window=5) -> NamedTuple

Compare the last `window` val-loss points with the window ending at `resume_epoch`
to decide whether the curve is still descending or has plateaued.
"""
function report_saturation(epochs::Vector{Int}, val_losses::Vector{Float32};
                           resume_epoch::Int=0, window::Int=5)
    n = length(val_losses)
    n == 0 && return (; plateau=true, still_dropping=false, slope_late=0.0,
                        slope_pre=0.0, val_resume=NaN, val_final=NaN,
                        rel_drop=0.0, new_best_after_resume=false, verdict="")
    w = min(window, n)
    slope_late = _finite_slope(val_losses[(n - w + 1):n])
    val_final = Float64(val_losses[end])
    val_resume = if resume_epoch > 0
        i = findfirst(==(resume_epoch), epochs)
        i === nothing ? Float64(val_losses[1]) : Float64(val_losses[i])
    else
        Float64(val_losses[1])
    end
    slope_pre = if resume_epoch > 0
        i = findfirst(==(resume_epoch), epochs)
        if i === nothing || i < 2
            0.0
        else
            i0 = max(1, i - w + 1)
            _finite_slope(val_losses[i0:i])
        end
    else
        0.0
    end
    rel_drop = val_resume == 0 ? 0.0 : (val_resume - val_final) / abs(val_resume)
    # Plateau: late slope not clearly negative, or post-resume drop < 1%.
    still_dropping = slope_late < -1.0e-4 && rel_drop > 0.01
    plateau = !still_dropping
    best_i = argmin(val_losses)
    new_best_after_resume = resume_epoch > 0 && epochs[best_i] > resume_epoch
    verdict = if still_dropping
        "Val loss epoch $(epochs[end])'te hâlâ düşüyor — daha da uzatmaya değer."
    else
        "Val loss plato yaptı — daha fazla epoch beklenen kazancı vermez."
    end
    return (; plateau, still_dropping, slope_late, slope_pre, val_resume, val_final,
            rel_drop, new_best_after_resume, verdict)
end

# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

function save_checkpoint(path::AbstractString, model, ps, st,
                         cfg::MTTrainConfig, mesh_params::MeshParams,
                         epoch::Int, best_loss::Float32;
                         opt_state=nothing, schedule_epochs::Int=cfg.epochs,
                         best_commemi_rms::Float64=NaN)
    mkpath(dirname(abspath(path)))
    if opt_state === nothing
        jldsave(path; model, ps, st, cfg, mesh_params, epoch, best_loss,
                schedule_epochs, best_commemi_rms)
    else
        jldsave(path; model, ps, st, cfg, mesh_params, epoch, best_loss,
                opt_state, schedule_epochs, best_commemi_rms)
    end
end

"""
    load_resume_checkpoint(path) -> NamedTuple

Load weights, RNG-independent training state, and optional Adam moments from a
JLD2 file written by [`save_checkpoint`](@ref). Missing `opt_state` (v5 and
earlier) is returned as `nothing` so the caller can rebuild Adam.
"""
function load_resume_checkpoint(path::AbstractString)
    isfile(path) || error("Resume checkpoint not found: $path")
    ckpt = load(path)
    haskey(ckpt, "ps") || error("checkpoint missing parameters 'ps': $path")
    haskey(ckpt, "st") || error("checkpoint missing state 'st': $path")
    epoch = Int(get(ckpt, "epoch", 0))
    best_loss = Float32(get(ckpt, "best_loss", Inf32))
    model = get(ckpt, "model", nothing)
    mesh  = get(ckpt, "mesh_params", nothing)
    cfg_ck = get(ckpt, "cfg", nothing)
    opt_state = get(ckpt, "opt_state", nothing)
    schedule_epochs = Int(get(ckpt, "schedule_epochs", 0))
    best_commemi_rms = Float64(get(ckpt, "best_commemi_rms", NaN))
    if schedule_epochs <= 0 && cfg_ck !== nothing
        schedule_epochs = try
            Int(cfg_ck.epochs)
        catch
            0
        end
    end
    return (; model, ps=ckpt["ps"], st=ckpt["st"], mesh_params=mesh,
            epoch, best_loss, cfg=cfg_ck, opt_state, schedule_epochs,
            best_commemi_rms)
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

"""Forward pass → CPU `(nz, nx)` log10-ρ grid for the COMMEMI probe."""
function predict_prior_grid(model::MTResistivityUNet2D, ps, st,
                            mt_data::Array{Float32,3}, use_gpu::Bool)::Matrix{Float32}
    st_eval = Lux.testmode(st)
    x = to_device(mt_data, Val(use_gpu))
    pred, _ = gpu_forward(use_gpu) do
        model(x, ps, st_eval)
    end
    p = Array(pred)
    ndims(p) == 3 && (p = dropdims(p; dims=3))
    return Float32.(p)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(cfg::MTTrainConfig=MTTrainConfig())
    rng = MersenneTwister(cfg.seed)
    use_gpu = cuda_available(verbose=true)
    !use_gpu && cuda_diagnostics()

    resume_path = strip(cfg.resume_from)
    resuming = !isempty(resume_path)
    resume_ckpt = resuming ? load_resume_checkpoint(resume_path) : nothing
    start_epoch = resuming ? resume_ckpt.epoch + 1 : 1
    resume_epoch = resuming ? resume_ckpt.epoch : 0
    start_epoch <= cfg.epochs || error(
        "nothing to do: resume starts at epoch $start_epoch but --epochs=$(cfg.epochs)")

    _say("═" ^ 60)
    _say(" MTResistivityUNet2D training — synthetic MT pairs")
    _say("═" ^ 60)
    _say("  Device: ", device_label(use_gpu))
    _say("  First epoch compiles CUDA/Zygote kernels; Colab can look frozen for 10–20 min.")
    if resuming
        println("  Resume: ", abspath(resume_path),
                "  (epoch ", resume_ckpt.epoch, " → ", cfg.epochs, ")")
    end

    X, Y, mesh = load_dataset(cfg.dataset)
    mesh = validate_mesh_params(mesh)
    N = size(X, 4)
    n_st, n_per, n_comp = size(X, 1), size(X, 2), size(X, 3)
    @info "Dataset loaded" path=cfg.dataset N=N stations=n_st periods=n_per components=n_comp
    @info "Target grid" nz=mesh.nz nx=mesh.nx

    reuse_split = resuming && isfile(cfg.split_json)
    train_idx, val_idx = if reuse_split
        load_train_val_split(cfg.split_json)
    else
        split_train_val(N, rng; train_frac=cfg.train_frac)
    end
    reuse_split || save_train_val_split(cfg.split_json, cfg.seed, train_idx, val_idx)
    println("\n── Train/val split ──")
    @printf("  N=%d  train=%d (%.0f%%)  val=%d (%.0f%%)  seed=%d%s\n",
            N, length(train_idx), 100 * length(train_idx) / N,
            length(val_idx), 100 * length(val_idx) / N, cfg.seed,
            reuse_split ? "  [loaded from split JSON]" : "")
    println("  first 3 train idx: ", train_idx[1:min(3, length(train_idx))])
    println("  first 3 val   idx: ", val_idx[1:min(3, length(val_idx))])
    @info "Split indices" path=cfg.split_json reused=reuse_split

    base_ch = cfg.base_channels
    n_down  = cfg.n_down
    if resuming && resume_ckpt.model !== nothing
        try
            base_ch = Int(resume_ckpt.model.base_channels)
            n_down  = Int(resume_ckpt.model.n_down)
        catch
        end
    end

    model = MTResistivityUNet2D(;
        in_channels=n_comp,
        base_channels=base_ch,
        n_down=n_down,
        mesh=mesh,
    )
    ps, st = Lux.setup(rng, model)
    if resuming
        ps = resume_ckpt.ps
        st = resume_ckpt.st
        println("  Loaded weights from epoch ", resume_ckpt.epoch,
                "  best_val=", resume_ckpt.best_loss)
    end
    n_params = count_parameters(ps)
    cap = report_capacity_change(mesh)
    println("\n── Model capacity ──")
    @printf("  this run:  base=%d  n_down=%d  params=%d\n",
            base_ch, n_down, n_params)
    @printf("  v4 layout: base=16 n_down=2  params=%d\n", cap.n_old)
    @printf("  v5 layout: base=32 n_down=3  params=%d  (×%.2f vs v4)\n",
            cap.n_new, cap.ratio)
    use_gpu && (ps = to_device(ps, Val(true)))

    opt = Optimisers.OptimiserChain(
        Optimisers.ClipNorm(cfg.clip_norm),
        Optimisers.Adam(cfg.lr),
    )
    opt_state = if resuming && resume_ckpt.opt_state !== nothing
        println("  Restored optimizer state from checkpoint")
        resume_ckpt.opt_state
    else
        resuming && println("  Optimizer state missing in checkpoint — fresh Adam (weights kept)")
        Optimisers.setup(opt, ps)
    end

    # Cosine horizon: on resume, keep the original schedule so LR continues
    # from where it left off (t clamped → lr_min after the original last epoch)
    # rather than restarting at lr_max. Override with --schedule-epochs.
    cosine_n = if cfg.schedule_epochs > 0
        cfg.schedule_epochs
    elseif resuming && resume_ckpt.schedule_epochs > 0
        resume_ckpt.schedule_epochs
    else
        cfg.epochs
    end

    λ_data = cfg.lambda_data
    λ_tv   = cfg.lambda_tv

    best_val_loss = resuming ? Float32(resume_ckpt.best_loss) : Inf32
    best_commemi_rms = resuming ? Float64(resume_ckpt.best_commemi_rms) : NaN
    best_epoch = resuming ? resume_ckpt.epoch : 0
    log_epochs = Int[]
    log_train  = Float32[]
    log_val    = Float32[]
    log_rms    = Float64[]

    prev_log = strip(cfg.resume_log)
    if resuming && !isempty(prev_log) && isfile(prev_log)
        log_epochs, log_train, log_val, log_rms, logged_best = load_training_log(prev_log)
        if best_epoch == 0
            best_epoch = logged_best
        end
        if !isfinite(best_commemi_rms)
            measured = filter(isfinite, log_rms)
            !isempty(measured) && (best_commemi_rms = minimum(measured))
        end
        println("  Prepended training log: ", prev_log,
                "  (", length(log_epochs), " epochs)")
    end

    mkpath(dirname(abspath(cfg.training_log)))
    mkpath(dirname(abspath(cfg.output)))
    if resuming && abspath(resume_path) != abspath(cfg.output)
        cp(resume_path, cfg.output; force=true)
        println("  Copied resume checkpoint → ", cfg.output, "  (kept until a better ckpt)")
    end

    commemi_mt = nothing
    probe_dir = joinpath(ROOT, "results", "commemi_probe")
    if cfg.commemi_every > 0
        if isfile(cfg.commemi_obs)
            # Do not swallow pack/shape errors: a C=2 tensor here used to load
            # "successfully", then DimensionMismatch was caught at the first
            # probe epoch and --commemi-every became a silent no-op.
            commemi_mt = load_commemi_mt(cfg.commemi_obs, mesh, standardize_mt_input)
            C_probe = size(commemi_mt, 3)
            C_probe == n_comp ||
                error("COMMEMI probe tensor $(size(commemi_mt)) has C=$C_probe " *
                      "but training C_in=$n_comp; refusing to disable " *
                      "--commemi-every silently")
            mkpath(probe_dir)
            @printf("  COMMEMI probe: every %d ep, %d VFSA iter  obs=%s  tensor=%s  C_in=%d\n",
                    cfg.commemi_every, cfg.commemi_iter, cfg.commemi_obs,
                    size(commemi_mt), n_comp)
        else
            @warn "COMMEMI obs missing; probe disabled" path=cfg.commemi_obs
        end
    end

    _say("\nStarting training for epochs ", start_epoch, "–", cfg.epochs,
         "  (cosine horizon=", cosine_n, ")")
    @printf("  N=%d  train=%d  val=%d  batch=%d  clip_norm=%.2f\n",
            N, length(train_idx), length(val_idx), cfg.batch_size, cfg.clip_norm)
    @printf("  λ_data=%.2e  λ_tv=%.2e  lr_max=%.2e  (cosine annealing)\n",
            λ_data, λ_tv, cfg.lr)
    println("  checkpoint selection: commemi_rms (primary) / val_loss (fallback) → ",
            cfg.output)
    flush(stdout)
    if resuming
        η_prev = cosine_annealing_lr(resume_epoch, cosine_n, cfg.lr)
        η_next = cosine_annealing_lr(start_epoch, cosine_n, cfg.lr)
        @printf("  LR continue: epoch %d → %.3e   epoch %d → %.3e\n",
                resume_epoch, η_prev, start_epoch, η_next)
    end

    first_step_logged = false
    for epoch in start_epoch:cfg.epochs
        η = cosine_annealing_lr(epoch, cosine_n, cfg.lr)
        Optimisers.adjust!(opt_state, η)
        _say(@sprintf("  epoch %d/%d starting (lr=%.3e) …", epoch, cfg.epochs, η))

        epoch_total = 0.0f0
        epoch_data  = 0.0f0
        epoch_tv    = 0.0f0
        n_seen = 0

        for idx in epoch_minibatches(rng, train_idx, cfg.batch_size)
            xb, yb = sample_batch(X, Y, idx)
            terms, st = train_step!(model, ps, st, opt_state, xb, yb, use_gpu;
                                    λ_data=λ_data, λ_tv=λ_tv)
            if !first_step_logged
                _say("  first GPU train step finished (compile done)")
                first_step_logged = true
            end
            bsz = length(idx)
            epoch_total += Float32(terms.total) * Float32(bsz)
            epoch_data  += Float32(terms.data)  * Float32(bsz)
            epoch_tv    += Float32(terms.tv)    * Float32(bsz)
            n_seen += bsz
        end

        n_seen > 0 || error("no train batches consumed")
        avg_train = epoch_total / Float32(n_seen)
        avg_data  = epoch_data  / Float32(n_seen)
        avg_tv    = epoch_tv    / Float32(n_seen)

        avg_val = eval_split_loss(model, ps, st, X, Y, val_idx, cfg.batch_size, use_gpu;
                                  λ_data=λ_data, λ_tv=λ_tv)

        if avg_train == avg_val
            @warn "train_loss == val_loss exactly; check that split indices differ" epoch=epoch avg_train=avg_train
        end

        epoch_rms = NaN
        do_probe = commemi_mt !== nothing &&
                   should_probe_epoch(epoch, cfg.epochs, cfg.commemi_every)
        if do_probe
            println("  ── COMMEMI probe (short VFSA, ", cfg.commemi_iter, " iter) ──")
            try
                logres = predict_prior_grid(model, ps, st, commemi_mt, use_gpu)
                epoch_rms = probe_commemi_rms(logres, mesh, cfg.commemi_obs;
                                              max_iter=cfg.commemi_iter,
                                              work_dir=probe_dir, epoch=epoch)
            catch err
                @warn "COMMEMI probe failed" epoch=epoch exception=err
                epoch_rms = NaN
            end
        end

        if isfinite(epoch_rms)
            @printf("Epoch %3d/%d  lr=%.3e  train=%.6e  val=%.6e  commemi_rms=%.4f  L_data=%.6e  L_TV=%.6e\n",
                    epoch, cfg.epochs, η, avg_train, avg_val, epoch_rms, avg_data, avg_tv)
        else
            @printf("Epoch %3d/%d  lr=%.3e  train=%.6e  val=%.6e  L_data=%.6e  L_TV=%.6e\n",
                    epoch, cfg.epochs, η, avg_train, avg_val, avg_data, avg_tv)
        end
        flush(stdout)
        flush(stderr)

        push!(log_epochs, epoch)
        push!(log_train, avg_train)
        push!(log_val, avg_val)
        push!(log_rms, epoch_rms)

        reason = checkpoint_reason(avg_val, epoch_rms, best_val_loss, best_commemi_rms)
        println("  checkpoint: ", reason)
        if should_save_checkpoint(avg_val, epoch_rms, best_val_loss, best_commemi_rms)
            best_val_loss = avg_val
            best_epoch = epoch
            if isfinite(epoch_rms)
                best_commemi_rms = epoch_rms
            end
            save_checkpoint(cfg.output, model, ps, st, cfg, mesh, epoch, best_val_loss;
                            opt_state=opt_state, schedule_epochs=cosine_n,
                            best_commemi_rms=best_commemi_rms)
            if isfinite(epoch_rms)
                @info "New best model saved (commemi_rms primary)" path=cfg.output val_loss=best_val_loss commemi_rms=best_commemi_rms epoch=epoch
            else
                @info "New best model saved (val_loss fallback)" path=cfg.output val_loss=best_val_loss epoch=epoch
            end
        end

        write_training_log(cfg.training_log, log_epochs, log_train, log_val, log_rms, best_epoch)
    end

    @info "Training log saved" path=cfg.training_log

    println("\nTraining complete.")
    if isfinite(best_commemi_rms)
        @printf("Best epoch: %d  val_loss=%.6e  commemi_rms=%.4f  →  %s\n",
                best_epoch, best_val_loss, best_commemi_rms, cfg.output)
    else
        @printf("Best epoch: %d  val_loss=%.6e  →  %s\n",
                best_epoch, best_val_loss, cfg.output)
    end

    sat = report_saturation(log_epochs, log_val; resume_epoch=resume_epoch)
    println("\n── Saturation check ──")
    @printf("  val @ resume (epoch %d): %.6e\n", resume_epoch, sat.val_resume)
    @printf("  val @ final  (epoch %d): %.6e\n", log_epochs[end], sat.val_final)
    @printf("  relative drop after resume: %.2f%%\n", 100 * sat.rel_drop)
    @printf("  slope last-5: %.3e   slope pre-resume last-5: %.3e\n",
            sat.slope_late, sat.slope_pre)
    println("  new best after resume: ", sat.new_best_after_resume,
            sat.new_best_after_resume ? " (epoch $best_epoch)" : "")
    println("  ", sat.verdict)

    measured_rms = [(e, r) for (e, r) in zip(log_epochs, log_rms) if isfinite(r)]
    if !isempty(measured_rms)
        println("\n── COMMEMI probe history ──")
        for (e, r) in measured_rms
            mark = e == best_epoch ? "  ← best ckpt" : ""
            @printf("  epoch %3d  commemi_rms=%.4f%s\n", e, r, mark)
        end
    end

    if cfg.no_plot
        println("  training curve skipped (--no-plot)")
    else
        curve = plot_training_curve(cfg.training_log, cfg.curve_png; resume_epoch=resume_epoch)
        curve !== nothing && println("  training curve → ", curve)
    end

    println("\n── Hyperparameter summary ──")
    println("  Öncesi  - checkpoint_selection: val_loss only")
    @printf("  Sonrası - checkpoint_selection: commemi_rms primary, val_loss fallback  (probe every %d ep, %d iter)\n",
            cfg.commemi_every, cfg.commemi_iter)
    println("  Train/val split doğrulaması: ",
            train_idx[1:min(3, length(train_idx))],
            " vs ",
            val_idx[1:min(3, length(val_idx))])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_cli_args(ARGS))
end
