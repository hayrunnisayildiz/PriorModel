#!/usr/bin/env julia
#=
Sanity check: bypass the U-Net entirely and feed the COMMEMI ground-truth
resistivity model through the mesh-transfer / .ini-export pipeline, then run
VFSA2DMT.  This isolates the prior-transfer mechanism from training quality.

Usage (from project root):
    julia --project=. scripts/sanity_check_mesh_transfer.jl
=#

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using Printf, Dates, MTGeophysics, Plots

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics: write_ini_prior, _write_vector_lines
using .ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams:
    MeshParams, DEFAULT_MESH

# ─────────────────────────────────────────────────────────────────────────────
# .ini parser (reads MTGeophysics write_model2d format)
# ─────────────────────────────────────────────────────────────────────────────

"""
    read_ini_model(path) -> (n_y, n_z, nza, y_sizes, z_sizes, loge_values, origin, rotation)

Parse an MTGeophysics `.ini` / `.true` model file.  Values are in log_e(ρ).
"""
function read_ini_model(path::String)
    isfile(path) || error("Model file not found: $path")
    lines = readlines(path)

    nza = 0
    data_lines = String[]
    for line in lines
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, "#")
            m = match(r"NZA\s*=\s*(\d+)", stripped)
            m !== nothing && (nza = parse(Int, m.captures[1]))
            continue
        end
        push!(data_lines, stripped)
    end

    header = split(data_lines[1])
    n_y = parse(Int, header[2])
    n_z = parse(Int, header[3])

    all_numbers = Float64[]
    for i in 2:length(data_lines)
        append!(all_numbers, parse.(Float64, split(data_lines[i])))
    end

    idx = 1
    _nx = 1  # x-dimension sizes (always [1.0] for 2D)
    x_sizes = all_numbers[idx:idx+_nx-1]; idx += _nx
    y_sizes = all_numbers[idx:idx+n_y-1]; idx += n_y
    z_sizes = all_numbers[idx:idx+n_z-1]; idx += n_z

    n_vals = n_y * n_z
    loge_values = all_numbers[idx:idx+n_vals-1]; idx += n_vals

    origin = (all_numbers[idx], all_numbers[idx+1], all_numbers[idx+2]); idx += 3
    rotation = all_numbers[idx]

    return (; n_y, n_z, nza, y_sizes, z_sizes, loge_values, origin, rotation)
end

"""
    extract_ground_resistivity(ini; as_log10=true) -> (Matrix{Float64}, y_core_sizes, z_ground_sizes)

Extract the ground-zone resistivity from a parsed .ini, stripping air rows.
Returns `(nz_ground, n_y)` matrix.  Values are log10(ρ) if `as_log10=true`, else log_e(ρ).
"""
function extract_ground_resistivity(ini; as_log10::Bool=true)
    (; n_y, n_z, nza, loge_values) = ini
    nz_ground = n_z - nza
    rho = reshape(loge_values, n_y, n_z)'  # (n_z, n_y) — iz-major in file means values are [iz=1,iy=1..n_y, iz=2,...]
    ground = rho[nza+1:end, :]  # (nz_ground, n_y)
    if as_log10
        ground = ground ./ log(10.0)
    end
    return ground
end

# ─────────────────────────────────────────────────────────────────────────────
# Mesh comparison & resample
# ─────────────────────────────────────────────────────────────────────────────

"""
    resample_to_uniform(ground_log10, src_y_sizes, src_z_sizes, mp) -> Matrix{Float32}

Resample a non-uniform source grid onto the uniform DEFAULT_MESH via
nearest-neighbour lookup on cell centres (log10-resistivity space).
"""
function resample_to_uniform(ground_log10::Matrix{Float64},
                             src_y_sizes::Vector{Float64},
                             src_z_sizes_ground::Vector{Float64},
                             mp::MeshParams)
    src_nz, src_ny = size(ground_log10)

    src_y_centres = cumsum(src_y_sizes) .- src_y_sizes ./ 2
    src_z_centres = cumsum(src_z_sizes_ground) .- src_z_sizes_ground ./ 2

    tgt_y_centres = [(j - 0.5) * mp.dx for j in 1:mp.nx]
    tgt_z_centres = [(i - 0.5) * mp.dz for i in 1:mp.nz]

    # Align origins: centre both profiles
    src_y_total = sum(src_y_sizes)
    tgt_y_total = mp.nx * mp.dx
    src_y_centres .+= (tgt_y_total - src_y_total) / 2

    out = Matrix{Float32}(undef, mp.nz, mp.nx)
    for jt in 1:mp.nx
        jn = argmin(abs.(src_y_centres .- tgt_y_centres[jt]))
        for it in 1:mp.nz
            in_ = argmin(abs.(src_z_centres .- tgt_z_centres[it]))
            if in_ <= src_nz && jn <= src_ny
                out[it, jt] = Float32(ground_log10[in_, jn])
            else
                out[it, jt] = Float32(log10(100.0))  # background
            end
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Initial-RMS extraction from VFSA log
# ─────────────────────────────────────────────────────────────────────────────

function extract_initial_rms(run_dir::String)
    log_path = joinpath(run_dir, "chain_01", "0vfsa2DMT.log")
    isfile(log_path) || return nothing
    for line in readlines(log_path)
        m = match(r"^\s+1\s+1\s+", line)
        if m !== nothing
            cols = split(strip(line))
            length(cols) >= 7 || continue
            return parse(Float64, cols[7])  # RMSCurr at iter 1
        end
    end
    return nothing
end

function extract_best_rms(run_dir::String)
    summary_path = joinpath(run_dir, "Summary.md")
    isfile(summary_path) || return nothing
    for line in readlines(summary_path)
        m = match(r"best_chain_rms:\s*([\d.]+)", line)
        m !== nothing && return parse(Float64, m.captures[1])
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# VFSA runner
# ─────────────────────────────────────────────────────────────────────────────

function _run_vfsa_sanity(start_model_path::String, data_path::String)
    MTGeophysics.VFSA2DMT(
        MTGeophysics.VFSA2DMTParams(
            script_path      = @__FILE__,
            start_model_path = start_model_path,
            data_path        = data_path,
            config = MTGeophysics.VFSA2DMTConfig(
                n_chains    = 1,
                n_ctrl      = 200,
                max_iter    = 100,
                n_trials    = 1,
                log_bounds  = (0.0, 4.0),
                seed        = 20260308,
                keep_models = true,
            ),
        ),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(ROOT, "results", "sanity_check")
    mkpath(output_dir)

    true_model_path = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.true")
    obs_data_path   = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    mp = DEFAULT_MESH

    println("═" ^ 65)
    println(" Sanity Check: mesh transfer (true model → .ini → VFSA2DMT)")
    println("═" ^ 65)

    # ── STEP 1: Read the COMMEMI true model ──────────────────────────────────
    println("\n[1/5] Reading COMMEMI true model...")
    ini = read_ini_model(true_model_path)
    @printf("  Source mesh: n_y=%d, n_z=%d (NZA=%d, ground=%d)\n",
            ini.n_y, ini.n_z, ini.nza, ini.n_z - ini.nza)
    @printf("  Target mesh: nx=%d, nz=%d, dx=%.0f, dz=%.0f\n",
            mp.nx, mp.nz, mp.dx, mp.dz)

    ground_log10 = extract_ground_resistivity(ini; as_log10=true)
    src_nz, src_ny = size(ground_log10)
    println("  Ground resistivity grid: ($src_nz, $src_ny)")

    # ── STEP 2: Check mesh compatibility & resample if needed ────────────────
    println("\n[2/5] Checking mesh compatibility...")

    src_z_ground = ini.z_sizes[ini.nza+1:end]
    needs_resample = (src_ny != mp.nx || src_nz != mp.nz ||
                      !all(isapprox.(ini.y_sizes, mp.dx; atol=1.0)) ||
                      !all(isapprox.(src_z_ground, mp.dz; atol=1.0)))

    if needs_resample
        println("  Meshes DIFFER — resampling required.")
        println("    Source: $(src_nz)×$(src_ny), dy range [$(minimum(ini.y_sizes)), $(maximum(ini.y_sizes))]m, dz range [$(minimum(src_z_ground)), $(maximum(src_z_ground))]m")
        println("    Target: $(mp.nz)×$(mp.nx), dx=$(mp.dx)m, dz=$(mp.dz)m")

        resampled = resample_to_uniform(ground_log10, ini.y_sizes, src_z_ground, mp)
        println("  Resampled grid: $(size(resampled))")
    else
        println("  Meshes match — no resample needed.")
        resampled = Float32.(ground_log10)
    end

    # ── STEP 3: Export via write_ini_prior and run VFSA2DMT ──────────────────
    println("\n[3/5] Exporting resampled true model to .ini...")
    sanity_ini = joinpath(output_dir, "sanity_check_true_model.ini")
    write_ini_prior(resampled, sanity_ini, mp;
                    title="Sanity check — COMMEMI true model (resampled)")
    println("  → $sanity_ini")

    println("\n  Running VFSA2DMT with true model as start model (100 iter)...")

    sanity_result = _run_vfsa_sanity(sanity_ini, obs_data_path)
    sanity_dir = sanity_result.run_info.run_dir
    @info "Sanity inversion complete" dir=sanity_dir

    # ── STEP 4: Three-way comparison table ───────────────────────────────────
    println("\n[4/5] Three-way RMS comparison")
    println("─" ^ 65)

    # Find existing halfspace & U-Net runs from earlier today
    halfspace_rms = 5.7354      # from run_VFSA2DMT_20260819_125857
    halfspace_init = 61.50178   # iter 1 RMS from the halfspace run log
    unet_rms = 8.3573           # from run_VFSA2DMT_20260819_125708

    # Try to get actual initial RMS from the known run dirs
    hs_run = joinpath(ROOT, "run_VFSA2DMT_20260819_125857")
    if isdir(hs_run)
        val = extract_initial_rms(hs_run)
        val !== nothing && (halfspace_init = val)
        val2 = extract_best_rms(hs_run)
        val2 !== nothing && (halfspace_rms = val2)
    end
    unet_run = joinpath(ROOT, "run_VFSA2DMT_20260819_125708")
    unet_init = nothing
    if isdir(unet_run)
        unet_init = extract_initial_rms(unet_run)
        val2 = extract_best_rms(unet_run)
        val2 !== nothing && (unet_rms = val2)
    end

    sanity_best = extract_best_rms(sanity_dir)
    sanity_init = extract_initial_rms(sanity_dir)

    fmt(x) = x === nothing ? "N/A" : @sprintf("%.4f", x)

    @printf("  %-30s  initial_rms = %-12s  best_rms = %-12s @ 100 iter\n",
            "Homojen başlangıç:", fmt(halfspace_init), fmt(halfspace_rms))
    @printf("  %-30s  initial_rms = %-12s  best_rms = %-12s @ 100 iter\n",
            "U-Net prior (eğitimsiz):", fmt(unet_init), fmt(unet_rms))
    @printf("  %-30s  initial_rms = %-12s  best_rms = %-12s @ 100 iter\n",
            "Gerçek model (sanity):", fmt(sanity_init), fmt(sanity_best))
    println("─" ^ 65)

    if sanity_init !== nothing
        if sanity_init < 2.0
            println("  ✓ Initial RMS çok düşük — prior aktarımı BAŞARILI.")
        else
            println("  ✗ Initial RMS beklenenden yüksek — aktarım/mesh hatası olabilir!")
        end
    end

    # ── STEP 5: Visual verification heatmaps ─────────────────────────────────
    println("\n[5/5] Generating verification heatmaps...")
    try
        # Heatmap 1: resampled true model
        p1 = Plots.heatmap(resampled;
            title="Resampled True Model",
            xlabel="Profile cell", ylabel="Depth cell",
            yflip=true, color=:turbo, clims=(0, 4),
            colorbar_title="log₁₀(ρ) [Ω·m]",
            size=(700, 400))
        resample_png = joinpath(output_dir, "resample_check.png")
        Plots.savefig(p1, resample_png)
        println("  → $resample_png")

        # Heatmap 2: round-trip — read back the exported .ini
        ini_rt = read_ini_model(sanity_ini)
        rt_ground = extract_ground_resistivity(ini_rt; as_log10=true)

        # The exported .ini has padding; extract only the core region
        n_pad_left = 0
        core_total = mp.nx * mp.dx
        accum = 0.0
        for s in ini_rt.y_sizes
            if accum + s < core_total * 0.01  # still in left pad
                if !isapprox(s, mp.dx; atol=1.0)
                    n_pad_left += 1
                else
                    break
                end
            else
                break
            end
        end
        # More robust: find contiguous block of dx-sized cells
        pad_left = 0
        for s in ini_rt.y_sizes
            isapprox(s, mp.dx; atol=1.0) ? break : (pad_left += 1)
        end
        core_region = rt_ground[:, pad_left+1:pad_left+mp.nx]

        # Only take the target nz rows (skip deeper padding if any)
        nz_show = min(size(core_region, 1), mp.nz)
        core_region = core_region[1:nz_show, :]

        p2 = Plots.heatmap(core_region;
            title="Round-trip: .ini → read-back (core)",
            xlabel="Profile cell", ylabel="Depth cell",
            yflip=true, color=:turbo, clims=(0, 4),
            colorbar_title="log₁₀(ρ) [Ω·m]",
            size=(700, 400))
        roundtrip_png = joinpath(output_dir, "ini_roundtrip_check.png")
        Plots.savefig(p2, roundtrip_png)
        println("  → $roundtrip_png")

        max_diff = maximum(abs.(Float64.(resampled[1:nz_show, :]) .- core_region))
        @printf("  Max |resampled − roundtrip| = %.6e log10(Ω·m)\n", max_diff)
        if max_diff < 1e-4
            println("  ✓ Round-trip fidelity excellent.")
        else
            println("  ✗ Round-trip mismatch detected — check export/import logic!")
        end

    catch e
        @warn "Plotting failed" exception=(e, catch_backtrace())
        println("  Heatmap generation skipped (Plots.jl not available).")
    end

    println("\n", "═" ^ 65)
    println(" Done. Results in: $output_dir")
    println("═" ^ 65)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
