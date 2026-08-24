#!/usr/bin/env julia
#=
Faz 1 smoke: COMMEMI checkpoint path, not a v8 training run.

Proves:
  1. `--commemi-every 10` schedule (`should_probe_epoch`)
  2. Checkpoint uses `commemi_rms`, not `val_loss` (v5→v6 trap)
  3. `probe_commemi_rms` runs a short VFSA and returns finite RMS
     without loading real GLMakie (no OpenGL)

Usage (repo root):
    julia --project=. scripts/smoke_commemi_checkpoint.jl
    julia --project=. scripts/smoke_commemi_checkpoint.jl --max-iter 2
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Dates
using HDF5

include(joinpath(ROOT, "src", "synthetic", "MeshParams.jl"))
include(joinpath(ROOT, "src", "training", "commemi_probe.jl"))

using .MTMeshParams: MeshParams, DEFAULT_MESH
using .CommemiProbe: should_probe_epoch, should_save_checkpoint, checkpoint_reason,
                     probe_commemi_rms

const DATASET = joinpath(ROOT, "data", "synthetic", "train_pairs_v7.h5")
const OBS     = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
const OUTDIR  = joinpath(ROOT, "results", "smoke_commemi_checkpoint")
const LOG     = joinpath(OUTDIR, "smoke_commemi_checkpoint.log")

function parse_max_iter(args::Vector{String})::Int
    i = findfirst(==("--max-iter"), args)
    i === nothing && return 2
    i < length(args) || error("--max-iter needs an integer")
    return parse(Int, args[i + 1])
end

function read_mesh(path::AbstractString)::MeshParams
    isfile(path) || error("dataset not found: $path")
    h5open(path, "r") do f
        a = HDF5.attributes(f)
        nx = haskey(a, "nx") ? Int(read(a["nx"])) : DEFAULT_MESH.nx
        nz = haskey(a, "nz") ? Int(read(a["nz"])) : DEFAULT_MESH.nz
        dx = haskey(a, "dx") ? Float64(read(a["dx"])) : DEFAULT_MESH.dx
        dz = haskey(a, "dz") ? Float64(read(a["dz"])) : DEFAULT_MESH.dz
        n_stations = haskey(a, "n_stations") ? Int(read(a["n_stations"])) : DEFAULT_MESH.n_stations
        periods = haskey(a, "periods") ? Float64.(read(a["periods"])) : copy(DEFAULT_MESH.periods)
        return MeshParams(nx, nz, dx, dz, n_stations, periods)
    end
end

function assert_ok(cond::Bool, msg::AbstractString)
    cond || error("FAIL: $msg")
    println("  PASS  ", msg)
    return nothing
end

function glmakie_status()::String
    pkgid = Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie")
    haskey(Base.loaded_modules, pkgid) || return "not loaded"
    m = Base.loaded_modules[pkgid]
    n = length(names(m; all=true))
    return n <= 2 ? "STUB (empty module, OpenGL skipped)" :
                    "REAL package loaded ($n names)"
end

function run_policy_tests()
    println("\n══ 1. should_probe_epoch (commemi-every = 10) ══")
    n, every = 50, 10
    probed = [e for e in 1:n if should_probe_epoch(e, n, every)]
    println("  probed epochs: ", probed)
    assert_ok(probed == [10, 20, 30, 40, 50], "probe on 10,20,30,40 and final epoch")
    assert_ok(!should_probe_epoch(1, n, every), "epoch 1 is not probed")
    assert_ok(!should_probe_epoch(11, n, every), "epoch 11 is not probed")
    assert_ok(!should_probe_epoch(10, n, 0), "--commemi-every 0 disables all probes")
    assert_ok(should_probe_epoch(3, 3, 10), "final epoch is always probed when every > 0")

    println("\n══ 2. checkpoint policy — v5→v6 trap ══")
    assert_ok(should_save_checkpoint(0.457, 11.85, Inf, NaN),
              "first measured commemi_rms always saves")
    assert_ok(!should_save_checkpoint(0.438, 13.32, 0.457, 11.85),
              "better val_loss must NOT overwrite worse commemi_rms (v5→v6 trap)")
    assert_ok(should_save_checkpoint(0.500, 10.00, 0.457, 11.85),
              "worse val_loss still saves if commemi_rms improves")
    assert_ok(!should_save_checkpoint(0.300, NaN, 0.457, 11.85),
              "after first probe, val_loss-only epochs never save")
    assert_ok(should_save_checkpoint(0.400, NaN, 0.457, NaN),
              "before first probe, val_loss fallback saves")
    println("  v5→v6 reason: ", checkpoint_reason(0.438, 13.32, 0.457, 11.85))
end

function run_mock_loop(; rms10::Float64, rms20::Float64)
    println("\n══ 3. mock training loop (20 ep, --commemi-every 10) ══")
    println("  val_loss improves every epoch (the trap). commemi_rms decides saves.")
    n_epochs, every = 20, 10
    best_val = Inf
    best_rms = NaN
    best_ep = 0
    saved = Int[]
    for epoch in 1:n_epochs
        val = 0.80 - 0.02 * epoch
        do_probe = should_probe_epoch(epoch, n_epochs, every)
        epoch_rms = !do_probe ? NaN : (epoch == 10 ? rms10 : rms20)
        reason = checkpoint_reason(val, epoch_rms, best_val, best_rms)
        save = should_save_checkpoint(val, epoch_rms, best_val, best_rms)
        mark = do_probe ? @sprintf("commemi_rms=%.4f", epoch_rms) : "no probe"
        @printf("  epoch %2d  val=%.4f  %-22s  %s\n", epoch, val, mark, reason)
        if save
            best_val = val
            best_ep = epoch
            isfinite(epoch_rms) && (best_rms = epoch_rms)
            push!(saved, epoch)
        end
    end
    println("  saved epochs: ", saved, "  best_epoch=", best_ep,
            "  best_commemi_rms=", best_rms)
    assert_ok(10 in saved, "epoch 10 (first probe) saved")
    assert_ok(!(20 in saved) || rms20 < rms10 - 1e-8,
              "epoch 20 saves only if commemi_rms improved (val_loss ignored)")
    assert_ok(best_ep == (rms20 < rms10 - 1e-8 ? 20 : 10),
              "best checkpoint follows commemi_rms, not the last/best val_loss")
    return best_ep, best_rms, saved
end

function main(args::Vector{String}=ARGS)
    max_iter = parse_max_iter(args)
    mkpath(OUTDIR)

    println("═"^64)
    println(" COMMEMI checkpoint smoke-run  ", Dates.now())
    println("═"^64)
    println("  dataset:  ", DATASET, "  exists=", isfile(DATASET))
    println("  obs:      ", OBS, "  exists=", isfile(OBS))
    println("  max_iter: ", max_iter, "  (short VFSA; training uses 25)")
    println("  log:      ", LOG)
    flush(stdout)

    run_policy_tests()

    isfile(DATASET) || error("missing $DATASET")
    isfile(OBS)     || error("missing $OBS")
    mp = read_mesh(DATASET)
    println("\n══ 4. live short VFSA (homogeneous start, mesh from v7 h5) ══")
    println("  mesh: ", mp)
    println("  GLMakie before VFSA: ", glmakie_status())
    flush(stdout)

    log100 = fill(Float32(2.0), mp.nz, mp.nx)  # 100 Ω·m
    log10  = fill(Float32(1.0), mp.nz, mp.nx)  # 10 Ω·m

    println("  running probe A (100 Ω·m, max_iter=$max_iter) …")
    flush(stdout)
    rms_a = probe_commemi_rms(log100, mp, OBS; max_iter=max_iter,
                              work_dir=OUTDIR, epoch=10)
    println("  GLMakie after VFSA:  ", glmakie_status())
    @printf("  probe A  epoch 10  commemi_rms = %.4f\n", rms_a)
    assert_ok(isfinite(rms_a) && rms_a > 0, "probe A returned a finite positive RMS")

    println("  running probe B (10 Ω·m, max_iter=$max_iter) …")
    flush(stdout)
    rms_b = probe_commemi_rms(log10, mp, OBS; max_iter=max_iter,
                              work_dir=OUTDIR, epoch=20)
    @printf("  probe B  epoch 20  commemi_rms = %.4f\n", rms_b)
    assert_ok(isfinite(rms_b) && rms_b > 0, "probe B returned a finite positive RMS")

    gl = glmakie_status()
    println("  GLMakie status: ", gl)
    assert_ok(startswith(gl, "STUB") || gl == "not loaded",
              "real GLMakie was not loaded (OpenGL skipped)")

    best_ep, best_rms, saved = run_mock_loop(; rms10=rms_a, rms20=rms_b)

    open(LOG, "w") do io
        println(io, "COMMEMI checkpoint smoke  ", Dates.now())
        println(io, "probe_A_rms=", rms_a, "  probe_B_rms=", rms_b)
        println(io, "best_epoch=", best_ep, "  best_commemi_rms=", best_rms)
        println(io, "saved=", saved)
        println(io, "glmakie=", gl)
        println(io, "ACCEPT: probe ran at epochs 10 and 20; checkpoint used commemi_rms")
    end

    println("\n══ ACCEPT ══")
    println("  commemi_probe ran a real short VFSA at probe epochs 10 and 20.")
    println("  Checkpoint rule used commemi_rms; improving val_loss did not win.")
    @printf("  best_epoch=%d  best_commemi_rms=%.4f  saved=%s\n",
            best_ep, best_rms, string(saved))
    println("  log → ", LOG)
    println()
    println("Full v7-hyperparameter training is NOT started. Proposed command:")
    println()
    println("  julia --project=. src/training/train_mt_resistivity.jl \\")
    println("    --dataset data/synthetic/train_pairs_v7.h5 \\")
    println("    --epochs 50 \\")
    println("    --output models/production_prior_v7_commemi.jld2 \\")
    println("    --training-log results/training_log_v7_commemi.csv \\")
    println("    --split-json results/train_val_split_v7.json \\")
    println("    --commemi-every 10 --commemi-iter 25 --no-plot")
    println()
    println("Ask before running that: it needs GPU / Colab time.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
