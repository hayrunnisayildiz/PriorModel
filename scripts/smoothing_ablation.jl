#!/usr/bin/env julia
#=
smoothing_ablation.jl — Hypothesis test: does the neural prior underperform
the homogeneous VFSA start because it is genuinely too coarse (Hypothesis A),
or because it contains structured artifacts (e.g. periodic/checkerboard
shadows) that actively mislead the VFSA search (Hypothesis B)?

Method: take the EXISTING trained checkpoint (no retraining), predict the
COMMEMI 2-D-I prior once, then apply increasing Gaussian smoothing to that
same prediction and re-run the *same* short VFSA on each smoothed version.

Interpretation:
  - If RMS improves markedly as smoothing increases (and gets closer to /
    beats the homogeneous baseline) -> Hypothesis B: the raw prior has
    harmful structured artifacts; fix = stronger TV/smoothness regularization
    during training, not necessarily more data.
  - If smoothing barely changes RMS, or the best RMS at any smoothing level
    still trails the homogeneous baseline -> Hypothesis A: the prior's
    underlying structure is simply too inaccurate; the fix is more/better
    training data (larger n, closer to the paper's 400k/50k, or a
    survey-conditioned scenario library), not post-hoc smoothing.

Usage (from PriorModel-master/, i.e. project root):
    julia --project=. scripts/smoothing_ablation.jl
    julia --project=. scripts/smoothing_ablation.jl \
        --ckpt models/mid_scale_prior_v5.jld2 \
        --data examples/0COMEMI2D-I/Comemi2D1.obs \
        --sigmas 0,0.5,1,2,4 \
        --max-iter 100 --n-ctrl 200

Reference numbers (KNOWN_LIMITATIONS.md, same VFSA settings: 1 chain,
n_ctrl=200, max_iter=100, seed=20260308):
    Homogeneous start : init 12.31  -> best 5.7354
    True model start  : init 1.30   -> best 1.3002
    U-Net v5 (best so far, real window): init 39.86 -> best 11.8539
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics:
    load_trained_model, predict_prior, standardize_mt_input,
    pack_te_response, pack_tetm_response

include(joinpath(ROOT, "src", "training", "commemi_probe.jl"))
using .CommemiProbe: write_probe_ini, run_short_vfsa

# Reference numbers, same VFSA settings (KNOWN_LIMITATIONS.md)
const HOMO_INIT = 12.31
const HOMO_BEST = 5.7354
const TRUE_INIT = 1.30
const TRUE_BEST = 1.3002

# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

Base.@kwdef struct Config
    ckpt_path::String   = joinpath(ROOT, "models", "mid_scale_prior_v5.jld2")
    data_path::String   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    out_dir::String      = joinpath(ROOT, "results", "smoothing_ablation")
    sigmas::Vector{Float64} = [0.0, 0.5, 1.0, 2.0, 4.0]
    max_iter::Int        = 100
    n_ctrl::Int           = 200
    seed::Int             = 20260308
end

function parse_args(args::Vector{String})::Config
    cfg = Config()
    ckpt, data, outdir = cfg.ckpt_path, cfg.data_path, cfg.out_dir
    sigmas, max_iter, n_ctrl, seed = cfg.sigmas, cfg.max_iter, cfg.n_ctrl, cfg.seed
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--ckpt" && i + 1 <= length(args)
            ckpt = args[i+1]; i += 2; continue
        elseif a == "--data" && i + 1 <= length(args)
            data = args[i+1]; i += 2; continue
        elseif a == "--out" && i + 1 <= length(args)
            outdir = args[i+1]; i += 2; continue
        elseif a == "--sigmas" && i + 1 <= length(args)
            sigmas = parse.(Float64, split(args[i+1], ","))
            i += 2; continue
        elseif a == "--max-iter" && i + 1 <= length(args)
            max_iter = parse(Int, args[i+1]); i += 2; continue
        elseif a == "--n-ctrl" && i + 1 <= length(args)
            n_ctrl = parse(Int, args[i+1]); i += 2; continue
        elseif a == "--seed" && i + 1 <= length(args)
            seed = parse(Int, args[i+1]); i += 2; continue
        end
        i += 1
    end
    return Config(; ckpt_path=ckpt, data_path=data, out_dir=outdir,
                  sigmas=sigmas, max_iter=max_iter, n_ctrl=n_ctrl, seed=seed)
end

# ─────────────────────────────────────────────────────────────────────────
# Manual separable Gaussian smoothing (no extra deps: reflect-pad + 1-D conv)
# ─────────────────────────────────────────────────────────────────────────

function gaussian_kernel1d(sigma::Float64)::Vector{Float64}
    sigma <= 0 && return [1.0]
    radius = max(1, ceil(Int, 3 * sigma))
    xs = -radius:radius
    k = [exp(-(x^2) / (2 * sigma^2)) for x in xs]
    return k ./ sum(k)
end

function conv1d_reflect(v::AbstractVector{<:Real}, k::Vector{Float64})::Vector{Float64}
    n = length(v)
    r = length(k) ÷ 2
    out = zeros(Float64, n)
    for i in 1:n
        acc = 0.0
        for (j, kj) in enumerate(k)
            offset = j - r - 1
            src = i + offset
            # reflect padding at boundaries
            if src < 1
                src = 2 - src
            elseif src > n
                src = 2n - src
            end
            src = clamp(src, 1, n)
            acc += kj * v[src]
        end
        out[i] = acc
    end
    return out
end

"""Separable 2-D Gaussian smoothing of a (nz, nx) grid. sigma in grid cells."""
function gaussian_smooth(grid::AbstractMatrix{<:Real}, sigma::Float64)::Matrix{Float32}
    sigma <= 0 && return Float32.(grid)
    k = gaussian_kernel1d(sigma)
    nz, nx = size(grid)
    tmp = zeros(Float64, nz, nx)
    for iz in 1:nz
        tmp[iz, :] = conv1d_reflect(view(grid, iz, :), k)
    end
    out = zeros(Float64, nz, nx)
    for ix in 1:nx
        out[:, ix] = conv1d_reflect(view(tmp, :, ix), k)
    end
    return Float32.(out)
end

# ─────────────────────────────────────────────────────────────────────────
# COMMEMI MT tensor (mirrors CommemiProbe's loader; TE or TE+TM by C_in)
# ─────────────────────────────────────────────────────────────────────────

function commemi_tensor(obs_path::String, mp, n_channels::Int)
    data = MTGeophysics.load_data2d(obs_path)
    y = Float64.(collect(data.receivers))
    periods = Float64.(collect(data.periods))
    raw = if n_channels == 4
        pack_tetm_response(data.rho_xy, data.phase_xy, data.rho_yx, data.phase_yx)
    elseif n_channels == 2
        pack_te_response(data.rho_xy, data.phase_xy)
    else
        error("Unsupported n_channels=$n_channels (expected 2 or 4)")
    end
    return standardize_mt_input(y, periods, raw; mp=mp, method=:bilinear)
end

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

function main(cfg::Config)
    @info "Loading checkpoint" cfg.ckpt_path
    model, ps, st, mp = load_trained_model(cfg.ckpt_path)

    @info "Checkpoint identity" base_channels=model.base_channels n_down=model.n_down in_channels=model.in_channels nx=mp.nx nz=mp.nz dx=mp.dx n_periods=length(mp.periods)

    @eval using MTGeophysics
    mt_data = Base.invokelatest(commemi_tensor, cfg.data_path, mp, Int(model.in_channels))
    @info "COMMEMI tensor" size=size(mt_data)

    logres_raw = predict_prior(model, ps, st, mt_data)
    @info "Raw prior stats" min=minimum(logres_raw) max=maximum(logres_raw) mean=mean(logres_raw) std=std(logres_raw)

    mkpath(cfg.out_dir)
    rows = NamedTuple[]

    for sigma in cfg.sigmas
        logres = gaussian_smooth(logres_raw, sigma)
        tag = @sprintf("sigma_%.2f", sigma)
        ini_path = joinpath(cfg.out_dir, "prior_$(tag).ini")
        write_probe_ini(logres, ini_path, mp; title="smoothing ablation $(tag)")

        rms, run_dir = run_short_vfsa(ini_path, cfg.data_path;
                                      max_iter=cfg.max_iter, n_ctrl=cfg.n_ctrl,
                                      seed=cfg.seed, work_dir=cfg.out_dir)
        @info "VFSA result" sigma rms run_dir
        push!(rows, (sigma=sigma, rms=rms, ini=ini_path))
    end

    # ── Report ──────────────────────────────────────────────────────────
    report_path = joinpath(cfg.out_dir, "comparison.txt")
    open(report_path, "w") do io
        println(io, "Smoothing ablation — checkpoint: $(cfg.ckpt_path)")
        println(io, "VFSA: 1 chain, n_ctrl=$(cfg.n_ctrl), max_iter=$(cfg.max_iter), seed=$(cfg.seed)")
        println(io, "")
        println(io, @sprintf("%-12s %10s", "start model", "best RMS"))
        println(io, @sprintf("%-12s %10.4f", "homogeneous", HOMO_BEST))
        println(io, @sprintf("%-12s %10.4f", "true model", TRUE_BEST))
        for row in rows
            println(io, @sprintf("%-12s %10.4f", "sigma=$(row.sigma)", row.rms))
        end
        println(io, "")
        best = rows[argmin([r.rms for r in rows])]
        println(io, "Best smoothed result: sigma=$(best.sigma), rms=$(round(best.rms; digits=4))")
        println(io, "")
        if best.rms < HOMO_BEST
            println(io, "=> Smoothing BEATS homogeneous baseline: structured artifacts")
            println(io, "   were likely misleading VFSA (Hypothesis B). Increase TV/")
            println(io, "   smoothness weight during training; retraining on more data")
            println(io, "   is not the first priority.")
        elseif rows[1].rms - best.rms > 0.5 * (rows[1].rms - HOMO_BEST + 1e-9)
            println(io, "=> Smoothing helps meaningfully but does not fully close the")
            println(io, "   gap to homogeneous: likely a mix of both hypotheses —")
            println(io, "   strengthen smoothness regularization AND grow training data.")
        else
            println(io, "=> Smoothing does NOT meaningfully change the result: the raw")
            println(io, "   prior's structure is simply too inaccurate (Hypothesis A).")
            println(io, "   Fix = more / better-distributed training data (see the")
            println(io, "   400k/50k scale used by Wang et al. 2023), not post-hoc")
            println(io, "   filtering or regularization tweaks alone.")
        end
    end

    println(read(report_path, String))
    @info "Report written" report_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_args(ARGS))
end
