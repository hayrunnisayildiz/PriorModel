"""
    CommemiProbe

Periodic COMMEMI 2-D-I VFSA probe used during U-Net training.

Each call: write the current prior as an `.ini` (same mesh recipe as
`write_ini_prior` in the export module) → short VFSA (`max_iter` ≪ 100) →
return `best_chain.best_rms`. Plots are skipped (`run_mt2d_vfsa`) so the
hook stays cheap enough to run every N epochs.
"""
module CommemiProbe

using Printf
using Dates
using Statistics

export load_commemi_mt, write_probe_ini, run_short_vfsa, probe_commemi_rms
export should_save_checkpoint, should_probe_epoch

# MTGeophysics.jl declares GLMakie as a hard dependency and does `using GLMakie`
# at package load (try/catch LoadError only). That import can hang or crash a
# headless Colab runtime. Load it only when a probe actually runs; training
# with `--commemi-every 0` never touches GLMakie.
function _ensure_mtgeophysics()
    isdefined(@__MODULE__, :MTGeophysics) && return
    @eval using MTGeophysics
    return
end

const DEFAULT_OBS = joinpath(@__DIR__, "..", "..", "examples", "0COMEMI2D-I", "Comemi2D1.obs")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint policy
# ─────────────────────────────────────────────────────────────────────────────

"""
    should_probe_epoch(epoch, n_epochs, every) -> Bool

Probe on every `every`-th epoch and always on the final epoch. `every <= 0`
disables the hook.
"""
function should_probe_epoch(epoch::Int, n_epochs::Int, every::Int)::Bool
    every <= 0 && return false
    return epoch == n_epochs || (epoch % every == 0)
end

"""
    should_save_checkpoint(val_loss, commemi_rms, best_val, best_rms) -> Bool

Primary metric is `commemi_rms` whenever it has been measured. `val_loss` is
the fallback before the first probe and the tie-breaker when RMS values are
equal. A better synthetic val loss never replaces a checkpoint whose measured
COMMEMI RMS is worse — that was the v5→v6 failure mode.
"""
function should_save_checkpoint(val_loss::Real, commemi_rms::Real,
                                best_val::Real, best_rms::Real)::Bool
    measured = isfinite(Float64(commemi_rms))
    have_rms = isfinite(Float64(best_rms))
    v = Float64(val_loss)
    r = Float64(commemi_rms)
    bv = Float64(best_val)
    br = Float64(best_rms)

    if measured
        if !have_rms
            return true
        elseif r < br - 1.0e-8
            return true
        elseif r > br + 1.0e-8
            return false
        else
            return v < bv
        end
    else
        have_rms && return false
        return v < bv
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# COMMEMI MT tensor
# ─────────────────────────────────────────────────────────────────────────────

"""Duck-typed load: `standardize_fn(data; mp)` returns `(nS, nP, C)` Float32."""
function load_commemi_mt(obs_path::AbstractString, mp, standardize_fn)
    isfile(obs_path) || error("COMMEMI obs not found: $obs_path")
    _ensure_mtgeophysics()
    data = MTGeophysics.load_data2d(obs_path)
    return standardize_fn(data; mp=mp, method=:bilinear)
end

# ─────────────────────────────────────────────────────────────────────────────
# .ini writer — identical defaults to ExportPriorToMTGeophysics.write_ini_prior
# ─────────────────────────────────────────────────────────────────────────────

function _write_vector_lines(io::IO, values::AbstractVector{<:Real}; per_line::Int=12)
    for first_idx in 1:per_line:length(values)
        last_idx = min(first_idx + per_line - 1, length(values))
        println(io, join([@sprintf("%.8e", Float64(values[idx]))
                          for idx in first_idx:last_idx], " "))
    end
end

"""Write a log10-ρ grid as an MTGeophysics `.ini` start model (duck-typed `mp`)."""
function write_probe_ini(resistivity_grid::AbstractMatrix{<:Real},
                         output_path::AbstractString,
                         mp;
                         title::String="PriorModel training probe",
                         n_air_cells::Int=8,
                         air_resistivity::Float64=1e9,
                         background_resistivity::Float64=100.0,
                         y_padding::Float64=9000.0,
                         pad_factor::Float64=1.25,
                         air_top::Float64=-12000.0)
    nz_target, nx_core = size(resistivity_grid)
    nz_target == Int(mp.nz) || error("grid rows ($nz_target) ≠ mp.nz ($(mp.nz))")
    nx_core == Int(mp.nx)   || error("grid cols ($nx_core) ≠ mp.nx ($(mp.nx))")

    y_core_half = (nx_core * Float64(mp.dx)) / 2.0
    y_core_sizes = fill(Float64(mp.dx), nx_core)

    left_pad = Float64[]
    Δy = Float64(mp.dx)
    accum = 0.0
    while accum < y_padding - 1e-9
        Δy *= pad_factor
        push!(left_pad, Δy)
        accum += Δy
    end
    right_pad = copy(left_pad)
    reverse!(left_pad)
    y_cell_sizes = vcat(left_pad, y_core_sizes, right_pad)
    n_y = length(y_cell_sizes)

    z_air = collect(range(air_top, 0.0, length=n_air_cells + 1))
    z_air_sizes = diff(z_air)
    z_ground_sizes = fill(Float64(mp.dz), nz_target)
    z_cell_sizes = vcat(z_air_sizes, z_ground_sizes)
    n_z = length(z_cell_sizes)

    full_rho = fill(background_resistivity, n_z, n_y)
    for iz in 1:n_air_cells
        full_rho[iz, :] .= air_resistivity
    end
    n_pad_left = length(left_pad)
    for iz in 1:nz_target, ix in 1:nx_core
        full_rho[n_air_cells + iz, n_pad_left + ix] = 10.0^Float64(resistivity_grid[iz, ix])
    end

    mkpath(dirname(abspath(output_path)))
    y_nodes_start = -sum(left_pad) - y_core_half
    open(output_path, "w") do io
        println(io, "# $title")
        println(io, "# NZA=$n_air_cells")
        println(io, "1 $n_y $n_z 0 LOGE")
        _write_vector_lines(io, [1.0])
        _write_vector_lines(io, y_cell_sizes)
        _write_vector_lines(io, z_cell_sizes)
        values = Float64[]
        sizehint!(values, n_y * n_z)
        for iz in 1:n_z, iy in 1:n_y
            push!(values, log(full_rho[iz, iy]))
        end
        _write_vector_lines(io, values)
        @printf(io, "%.8e %.8e %.8e\n", 0.0, y_nodes_start, air_top)
        println(io, "0.0")
    end
    return abspath(output_path)
end

# ─────────────────────────────────────────────────────────────────────────────
# Short VFSA
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_short_vfsa(start_ini, obs_path; max_iter=25, work_dir, seed=20260308)
        -> (rms, run_dir)

1 chain, 200 ctrl (same as the full COMMEMI eval), `max_iter` ≪ 100.
Uses `run_mt2d_vfsa` (the plot-free API; `VFSA2DMT()` is the wrapper that
writes CairoMakie figures). `keep_models=false`, `snapshot_interval=0`.
Loading MTGeophysics still pulls GLMakie at package init — see `_ensure_mtgeophysics`.
"""
function run_short_vfsa(start_ini::AbstractString, obs_path::AbstractString;
                        max_iter::Int=25,
                        n_ctrl::Int=200,
                        seed::Int=20260308,
                        work_dir::AbstractString=".")
    isfile(start_ini) || error("start model not found: $start_ini")
    isfile(obs_path)  || error("obs not found: $obs_path")
    _ensure_mtgeophysics()
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    run_dir = joinpath(abspath(work_dir), "probe_VFSA2DMT_$stamp")
    mkpath(run_dir)

    cfg = MTGeophysics.VFSA2DMTConfig(
        n_chains    = 1,
        n_ctrl      = n_ctrl,
        max_iter    = max_iter,
        n_trials    = 1,
        log_bounds  = (0.0, 4.0),
        seed        = seed,
        keep_models = false,
        snapshot_interval = 0,
        output_root = abspath(work_dir),
    )
    result = MTGeophysics.run_mt2d_vfsa(
        abspath(start_ini), abspath(obs_path);
        run_dir = run_dir, config = cfg,
    )
    rms = Float64(result.best_chain.best_rms)
    return rms, run_dir
end

"""
    probe_commemi_rms(logres, mp, obs_path; max_iter, work_dir, epoch) -> Float64

Write the current prior and run a short COMMEMI VFSA. Returns best RMS.
"""
function probe_commemi_rms(logres::AbstractMatrix{<:Real}, mp,
                           obs_path::AbstractString;
                           max_iter::Int=25,
                           work_dir::AbstractString=".",
                           epoch::Int=0)
    mkpath(abspath(work_dir))
    ini = joinpath(abspath(work_dir), @sprintf("probe_epoch_%04d.ini", epoch))
    write_probe_ini(Float32.(logres), ini, mp;
                    title=@sprintf("training probe epoch %d", epoch))
    rms, run_dir = run_short_vfsa(ini, obs_path; max_iter=max_iter, work_dir=work_dir)
    @info "COMMEMI probe" epoch=epoch max_iter=max_iter rms=rms run_dir=run_dir
    return rms
end

end # module
