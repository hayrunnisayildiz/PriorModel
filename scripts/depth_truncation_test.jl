#!/usr/bin/env julia
#=
depth_truncation_test.jl — Does the 1200 m target-zone depth (UNET_MESH:
nz=48 x dz=25 m) alone explain a large part of the gap between U-Net priors
and the homogeneous baseline, independent of any neural network or training
data quality?

Background: COMMEMI 2-D-I's true conductive body spans depth 200-3600 m
(~3400 m thick). The U-Net's predicted "target zone" only covers the top
1200 m; everything below is scaffolding that PriorModel's own
`solver_resistivity()` builds by mechanically continuing the bottom target
row downward (see synthetic_generator.jl, "the deep zone below the target
window continues the bottom target row ... never training targets").
~70% of the true body's thickness sits below what any prior — however
well-trained — can ever represent.

This script tests that specific hypothesis directly, with NO neural network:
  1. Load COMMEMI's own true model (examples/0COMEMI2D-I/Comemi2D1.true).
  2. Truncate it below 1200 m using PriorModel's own convention (continue
     the row at the 1200 m boundary downward) — i.e. give the "prior" a
     PERFECT match to the true model in the shallow zone it can represent,
     and only scaffold the part it structurally cannot.
  3. Run the standard VFSA probe on this truncated model and compare its
     RMS to the known reference numbers.

Interpretation:
  - If the truncated-but-otherwise-perfect model's RMS is already close to
    the U-Net priors' RMS (~10-11), depth truncation alone is a first-order
    ceiling on achievable accuracy — independent of training data quality
    or scenario diversity. Fix = extend the target zone (n_target/target_dz)
    or explicitly accept/document the ceiling; more data will NOT close
    this gap.
  - If the truncated model's RMS is close to the true-model RMS (1.30),
    depth truncation is NOT the bottleneck, and Hypothesis A (data
    diversity/quality) remains the right thing to keep working on.

Usage (from project root):
    julia --project=. scripts/depth_truncation_test.jl
    julia --project=. scripts/depth_truncation_test.jl --target-depth 1200 --max-iter 100

Reference numbers (KNOWN_LIMITATIONS.md / prior sessions, same VFSA
settings: 1 chain, n_ctrl=200, max_iter=100, seed=20260308):
    True model         : init 1.30   -> best 1.3002
    Homogeneous         : init 12.31  -> best 5.7354
    U-Net v5             : init 39.86  -> best 11.8539
    U-Net v8 (TE+TM dx160): init ~?    -> best 10.7045
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Dates

# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

Base.@kwdef struct Config
    true_path::String   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
    data_path::String   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    out_dir::String       = joinpath(ROOT, "results", "depth_truncation_test")
    target_depth_m::Float64 = 1200.0   # UNET_MESH: nz=48 * dz=25 m
    max_iter::Int          = 100
    n_ctrl::Int             = 200
    seed::Int               = 20260308
end

function parse_args(args::Vector{String})::Config
    cfg = Config()
    true_path, data, outdir = cfg.true_path, cfg.data_path, cfg.out_dir
    target_depth, max_iter, n_ctrl, seed = cfg.target_depth_m, cfg.max_iter, cfg.n_ctrl, cfg.seed
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--true" && i + 1 <= length(args)
            true_path = args[i+1]; i += 2; continue
        elseif a == "--data" && i + 1 <= length(args)
            data = args[i+1]; i += 2; continue
        elseif a == "--out" && i + 1 <= length(args)
            outdir = args[i+1]; i += 2; continue
        elseif a == "--target-depth" && i + 1 <= length(args)
            target_depth = parse(Float64, args[i+1]); i += 2; continue
        elseif a == "--max-iter" && i + 1 <= length(args)
            max_iter = parse(Int, args[i+1]); i += 2; continue
        elseif a == "--n-ctrl" && i + 1 <= length(args)
            n_ctrl = parse(Int, args[i+1]); i += 2; continue
        elseif a == "--seed" && i + 1 <= length(args)
            seed = parse(Int, args[i+1]); i += 2; continue
        end
        i += 1
    end
    return Config(; true_path=true_path, data_path=data, out_dir=outdir,
                  target_depth_m=target_depth, max_iter=max_iter, n_ctrl=n_ctrl, seed=seed)
end

# ─────────────────────────────────────────────────────────────────────────
# Minimal .ini (write_model2d / LOGE format) reader + writer
# (self-contained: no dependency on MTGeophysics-internal parsing helpers)
# ─────────────────────────────────────────────────────────────────────────

struct IniModel
    title::String
    n_air_cells::Int
    x_cell_sizes::Vector{Float64}
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    resistivity::Matrix{Float64}   # (n_z, n_y), linear Ω·m, iz-major as on disk
    origin::Vector{Float64}
    rotation::Float64
end

function read_ini_model(path::String)::IniModel
    lines = readlines(path)
    title = lstrip(lines[1], ['#', ' '])
    nza = parse(Int, split(lines[2], "=")[2])
    parts = split(lines[3])
    n_y, n_z = parse(Int, parts[2]), parse(Int, parts[3])

    nums = Float64[]
    for l in lines[4:end]
        isempty(strip(l)) && continue
        append!(nums, parse.(Float64, split(l)))
    end

    i = 1
    x_cell = nums[i:i]; i += 1
    y_cells = nums[i:i+n_y-1]; i += n_y
    z_cells = nums[i:i+n_z-1]; i += n_z
    vals = nums[i:i+n_y*n_z-1]; i += n_y * n_z
    origin = nums[i:i+2]; i += 3
    rotation = nums[i]

    # values on disk are log_e(rho), iz-major: for iz in 1:n_z, for iy in 1:n_y
    resistivity = Matrix{Float64}(undef, n_z, n_y)
    k = 1
    for iz in 1:n_z, iy in 1:n_y
        resistivity[iz, iy] = exp(vals[k])
        k += 1
    end

    return IniModel(title, nza, x_cell, y_cells, z_cells, resistivity, origin, rotation)
end

function write_ini_model(path::String, m::IniModel)
    mkpath(dirname(abspath(path)))
    n_z, n_y = size(m.resistivity)
    open(path, "w") do io
        println(io, "# $(m.title)")
        println(io, "# NZA=$(m.n_air_cells)")
        println(io, "1 $n_y $n_z 0 LOGE")
        for v in (m.x_cell_sizes, m.y_cell_sizes, m.z_cell_sizes)
            for first_idx in 1:12:length(v)
                last_idx = min(first_idx + 11, length(v))
                println(io, join([@sprintf("%.8e", v[k]) for k in first_idx:last_idx], " "))
            end
        end
        flat = Float64[]
        sizehint!(flat, n_y * n_z)
        for iz in 1:n_z, iy in 1:n_y
            push!(flat, log(m.resistivity[iz, iy]))
        end
        for first_idx in 1:12:length(flat)
            last_idx = min(first_idx + 11, length(flat))
            println(io, join([@sprintf("%.8e", flat[k]) for k in first_idx:last_idx], " "))
        end
        @printf(io, "%.8e %.8e %.8e\n", m.origin[1], m.origin[2], m.origin[3])
        println(io, @sprintf("%.8e", m.rotation))
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────
# Truncation: continue the row at target_depth downward (PriorModel's own
# "deep zone continues bottom target row" convention, applied to COMMEMI's
# own true model / mesh, above n_air_cells)
# ─────────────────────────────────────────────────────────────────────────

function truncate_below_target!(m::IniModel, target_depth_m::Float64)
    n_z, n_y = size(m.resistivity)
    ground_rows = (m.n_air_cells+1):n_z
    depth = 0.0
    boundary_row = ground_rows[end]
    for iz in ground_rows
        depth += m.z_cell_sizes[iz]
        if depth >= target_depth_m
            boundary_row = iz
            break
        end
    end
    @info "Truncation boundary" boundary_row depth_reached=depth target_depth_m n_ground_rows=length(ground_rows)
    for iz in (boundary_row+1):n_z, iy in 1:n_y
        m.resistivity[iz, iy] = m.resistivity[boundary_row, iy]
    end
    return m, boundary_row, depth
end

# ─────────────────────────────────────────────────────────────────────────
# VFSA run + RMS extraction (same convention as evaluate_mid_scale_v8.jl)
# ─────────────────────────────────────────────────────────────────────────

function run_vfsa_and_extract(ini_path::String, obs_path::String, out_dir::String;
                              n_ctrl::Int, max_iter::Int, seed::Int)
    @eval using MTGeophysics
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    run_dir = joinpath(abspath(out_dir), "VFSA2DMT_$stamp")
    mkpath(run_dir)
    # World-age: MTGeophysics is loaded inside this function, so name lookup
    # of VFSA2DMTConfig / run_mt2d_vfsa must happen in invokelatest (same
    # pattern as scripts/data_coverage_check.jl).
    init_rms, best_rms = Base.invokelatest() do
        cfg = MTGeophysics.VFSA2DMTConfig(;
            n_chains=1, n_ctrl=n_ctrl, max_iter=max_iter, n_trials=1,
            log_bounds=(0.0, 4.0), seed=seed, keep_models=false, snapshot_interval=0,
            output_root=abspath(out_dir))
        result = MTGeophysics.run_mt2d_vfsa(
            abspath(ini_path), abspath(obs_path); run_dir=run_dir, config=cfg)
        best = Float64(result.best_chain.best_rms)
        init = nothing
        log_path = joinpath(run_dir, "chain_01", "0vfsa2DMT.log")
        if isfile(log_path)
            for line in readlines(log_path)
                m = match(r"^\s+1\s+1\s+", line)
                if m !== nothing
                    cols = split(strip(line))
                    length(cols) >= 7 && (init = parse(Float64, cols[7]))
                    break
                end
            end
        end
        return init, best
    end
    return init_rms, best_rms, run_dir
end

# ─────────────────────────────────────────────────────────────────────────
# Reference numbers
# ─────────────────────────────────────────────────────────────────────────

const TRUE_BEST = 1.3002
const HOMO_BEST = 5.7354
const V5_BEST   = 11.8539
const V8_BEST   = 10.7045

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

function main(cfg::Config)
    @info "Reading COMMEMI true model" cfg.true_path
    model = read_ini_model(cfg.true_path)
    @info "Model geometry" n_air=model.n_air_cells n_z=size(model.resistivity,1) n_y=size(model.resistivity,2)

    truncated, boundary_row, depth_reached = truncate_below_target!(model, cfg.target_depth_m)

    mkpath(cfg.out_dir)
    trunc_path = joinpath(cfg.out_dir, "commemi_truncated_$(Int(cfg.target_depth_m))m.ini")
    write_ini_model(trunc_path, truncated)
    @info "Truncated model written" trunc_path

    @info "Running VFSA on truncated model" cfg.max_iter cfg.n_ctrl
    init_rms, best_rms, run_dir = run_vfsa_and_extract(
        trunc_path, cfg.data_path, cfg.out_dir;
        n_ctrl=cfg.n_ctrl, max_iter=cfg.max_iter, seed=cfg.seed)

    report_path = joinpath(cfg.out_dir, "report.txt")
    open(report_path, "w") do io
        println(io, "Depth truncation test")
        println(io, "  true model     : $(cfg.true_path)")
        println(io, "  target depth   : $(cfg.target_depth_m) m  (boundary row=$boundary_row, depth_reached=$(round(depth_reached, digits=1)) m)")
        println(io, "  truncated ini  : $trunc_path")
        println(io, "  VFSA run_dir   : $run_dir")
        println(io, "")
        @printf(io, "%-30s %10s %10s\n", "model", "init RMS", "best RMS")
        @printf(io, "%-30s %10s %10.4f\n", "true model (full, reference)", "1.30", TRUE_BEST)
        @printf(io, "%-30s %10s %10.4f\n", "homogeneous (reference)", "12.31", HOMO_BEST)
        @printf(io, "%-30s %10s %10.4f\n", "U-Net v5 (reference)", "39.86", V5_BEST)
        @printf(io, "%-30s %10s %10.4f\n", "U-Net v8 (reference)", "~10.70", V8_BEST)
        @printf(io, "%-30s %10s %10.4f\n", "TRUNCATED true model (this run)",
                init_rms === nothing ? "n/a" : @sprintf("%.2f", init_rms), best_rms)
        println(io, "")
        unet_avg = (V5_BEST + V8_BEST) / 2
        # Midpoint between homogeneous (5.74) and U-Net (~11.3). Above this
        # the truncated model sits with the U-Net priors (depth is a ceiling);
        # at/below HOMO_BEST it sits with the homogeneous baseline (data is
        # the lever). The original `<` on the midpoint made the U-Net-ceiling
        # branch fire for every score below ~8.5, including scores that beat
        # homogeneous, and made the elseif unreachable.
        midpoint = HOMO_BEST + 0.5 * (unet_avg - HOMO_BEST)
        if best_rms >= midpoint
            println(io, "=> Truncated-but-otherwise-perfect model scores close to the")
            println(io, "   U-Net priors, NOT close to the true-model reference (1.30).")
            println(io, "   This confirms depth truncation (1200 m target zone vs COMMEMI's")
            println(io, "   200-3600 m body) is a first-order, DATA-INDEPENDENT ceiling on")
            println(io, "   achievable prior accuracy. More/better training data will NOT")
            println(io, "   close this gap by itself — the fix is extending the target zone")
            println(io, "   (n_target/target_dz) or explicitly documenting/accepting the")
            println(io, "   ceiling for benchmarks with structure below 1200 m.")
        elseif best_rms < HOMO_BEST
            println(io, "=> Truncated model already beats the homogeneous baseline, closer")
            println(io, "   to the true-model reference than to the U-Net priors. Depth")
            println(io, "   truncation costs some accuracy but is not the dominant")
            println(io, "   bottleneck — Hypothesis A (training data quality/diversity)")
            println(io, "   remains the primary lever.")
        else
            println(io, "=> Ambiguous / between references — inspect numbers manually")
            println(io, "   alongside the shallow-zone match quality before concluding.")
        end
    end

    println(read(report_path, String))
    @info "Report written" report_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_args(ARGS))
end
