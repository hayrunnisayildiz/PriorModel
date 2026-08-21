#!/usr/bin/env julia
#=
Cost probe for finer horizontal mesh (same ~19.2 km span, nz/dz unchanged).

Times 10 generate+forward samples at dx=160 (baseline), dx=80 (nx=240),
and dx=50 (nx=384), counts U-Net parameters, and estimates Colab T4 (16 GB)
training memory. Does **not** mutate UNET_MESH.

Usage (from project root):
    julia --project=. scripts/estimate_mesh_resolution_cost.jl
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Random
using Statistics
using Lux

include(joinpath(ROOT, "src", "synthetic", "synthetic_generator.jl"))
using .SyntheticGenerator
using .SyntheticGenerator.MTMeshParams: UNET_MESH, unet_log_periods
using .SyntheticGenerator: n_y, n_z, target_size, generate_model, forward_response,
                           build_generator_mesh, load_generator_priors

include(joinpath(ROOT, "src", "networks", "mt_resistivity_unet2d.jl"))
using .MTResistivityUNet2DLayers: MTResistivityUNet2D, count_parameters
using .MTResistivityUNet2DLayers.MTMeshParams: MeshParams

const N_SAMPLES = 10
const N_TARGET = 250
const N_TARGET_ALT = 300
const BATCH = 16
const BASE_CH = 32
const N_DOWN = 3
const T4_GB = 16.0
const NZ = UNET_MESH.nz
const DZ = UNET_MESH.dz
const N_STATIONS = UNET_MESH.n_stations
const SPAN_M = UNET_MESH.nx * UNET_MESH.dx   # 19.2 km

# Keep 19.2 km and divisibility by 2^n_down = 8.
const CANDIDATES = (
    (name = "baseline dx=160 m", dx = 160.0, nx = 120),
    (name = "candidate dx=80 m",  dx = 80.0,  nx = 240),
    (name = "candidate dx=50 m",  dx = 50.0,  nx = 384),
)

function make_mesh_params(nx::Int, dx::Float64)::MeshParams
    return MeshParams(nx, NZ, dx, DZ, N_STATIONS, unet_log_periods())
end

function make_cfg(nx::Int, dx::Float64)
    half = nx * dx / 2
    return GeneratorConfig(
        y_core_range = (-half, half),
        y_core_cell = dx,
        n_target = NZ,
        target_dz = DZ,
        receiver_stride = max(1, round(Int, nx / N_STATIONS)),
    )
end

"""Conservative T4 training-footprint estimate (Adam + Zygote activations), GB."""
function estimate_train_mem_gb(nz::Int, nx::Int, n_params::Int;
                               batch::Int=BATCH, base::Int=BASE_CH)::Float64
    c1, c2, c4, c8 = base, 2base, 4base, 8base
    nz2, nx2 = nz ÷ 2, nx ÷ 2
    nz4, nx4 = nz ÷ 4, nx ÷ 4
    nz8, nx8 = nz ÷ 8, nx ÷ 8
    maps = (
        (nz4, nx4, c2),
        (nz4, nx4, c4),
        (nz8, nx8, c8),
        (nz8, nx8, c8),
        (nz4, nx4, c4),
        (nz2, nx2, c2),
        (nz,  nx,  c1),
        (nz,  nx,  1),
    )
    act_elems = sum(Int(h) * Int(w) * Int(c) for (h, w, c) in maps)
    # Zygote stores several intermediates per conv; 8× is a conservative ceiling.
    act_bytes = Float64(act_elems) * batch * 4.0 * 8.0
    # params + grads + Adam m/v
    opt_bytes = Float64(n_params) * 4.0 * 4.0
    # cuDNN workspace / fragmentation headroom
    overhead = 0.75 * 1024^3
    return (act_bytes + opt_bytes + overhead) / 1024^3
end

function count_unet_params(mp::MeshParams)::Int
    model = MTResistivityUNet2D(;
        in_channels=2, base_channels=BASE_CH, n_down=N_DOWN, mesh=mp)
    ps, _ = Lux.setup(Random.default_rng(), model)
    return count_parameters(ps)
end

function projector_out_dim(nx::Int)::Int
    nz_p, nx_p = NZ ÷ 4, nx ÷ 4
    proj_ch = 2 * BASE_CH   # n_down=3 → c2
    return nz_p * nx_p * proj_ch
end

function time_forwards(nx::Int, dx::Float64; n::Int=N_SAMPLES)
    cfg = make_cfg(nx, dx)
    gmesh = build_generator_mesh(cfg)
    priors = let
        p = joinpath(ROOT, "config", "keivitsa_priors.json")
        isfile(p) ? load_generator_priors(p) : load_generator_priors()
    end
    rng = MersenneTwister(20260821)
    nzt, nyc = target_size(gmesh)
    nzt == NZ || error("nz drifted: got $nzt expected $NZ")
    nyc == nx || error("nx drifted: got $nyc expected $nx")

    println("  solver grid (n_z, n_y) = ($(n_z(gmesh)), $(n_y(gmesh)))  target=($nzt, $nyc)")
    println("  stations=$(length(gmesh.receiver_positions))  freqs=$(length(gmesh.frequencies))")
    flush(stdout)

    # Warm-up (compile MTGeophysics + first factorisation)
    println("  warmup generate+forward …")
    flush(stdout)
    t_w = @elapsed begin
        m0 = generate_model(gmesh, priors, cfg; rng=rng, index=0)
        forward_response(m0, gmesh; mode=:TETM)
    end
    @printf("  warmup wall = %.2f s\n", t_w)
    flush(stdout)

    t_gen = Float64[]
    t_fwd = Float64[]
    t_tot = Float64[]
    for i in 1:n
        t0 = time()
        model = generate_model(gmesh, priors, cfg; rng=rng, index=i)
        tg = time() - t0
        t1 = time()
        forward_response(model, gmesh; mode=:TETM)
        tf = time() - t1
        push!(t_gen, tg)
        push!(t_fwd, tf)
        push!(t_tot, tg + tf)
        @printf("    sample %2d/%d  gen=%.2fs  fwd=%.2fs  tot=%.2fs\n",
                i, n, tg, tf, tg + tf)
        flush(stdout)
    end
    return (;
        n_z=n_z(gmesh), n_y=n_y(gmesh), nzt, nyc,
        n_stations=length(gmesh.receiver_positions),
        mean_gen=mean(t_gen), mean_fwd=mean(t_fwd), mean_tot=mean(t_tot),
        std_fwd=std(t_fwd), warmup=t_w,
        times_fwd=t_fwd,
    )
end

function main()
    out_path = joinpath(ROOT, "results", "resolution_cost_v8.txt")
    mkpath(dirname(out_path))

    println("═" ^ 78)
    println(" Mesh resolution cost probe  (span=$(SPAN_M/1000) km, nz=$(NZ), dz=$(DZ) m)")
    println("═" ^ 78)
    @printf("  current UNET_MESH: nx=%d  dx=%.1f m  span=%.1f km\n",
            UNET_MESH.nx, UNET_MESH.dx, SPAN_M / 1000)
    println()

    rows = []
    io_buf = IOBuffer()
    println(io_buf, "Mesh resolution cost probe")
    println(io_buf, "span=$(SPAN_M) m  nz=$(NZ)  dz=$(DZ) m  n_stations=$(N_STATIONS)")
    println(io_buf, "U-Net: base=$(BASE_CH) n_down=$(N_DOWN) batch=$(BATCH)")
    println(io_buf, "forward samples per mesh: $N_SAMPLES  (warmup excluded from mean)")
    println(io_buf)

    println("── Parameter count + T4 memory (no forward) ──")
    for c in CANDIDATES
        mp = make_mesh_params(c.nx, c.dx)
        npar = count_unet_params(mp)
        mem = estimate_train_mem_gb(mp.nz, mp.nx, npar)
        out_dim = projector_out_dim(c.nx)
        dense_params = 513 * out_dim   # Dense(512, out) + bias
        xs = (1 - 0.5) * c.dx - c.nx * c.dx / 2
        stride = max(1, round(Int, c.nx / N_STATIONS))
        last_idx = 1 + (N_STATIONS - 1) * stride
        xe = (last_idx - 0.5) * c.dx - c.nx * c.dx / 2
        @printf("  %-22s  cells=%d×%d  span=%.1f km  params=%d  projector_Dense≈%d\n",
                c.name, NZ, c.nx, c.nx * c.dx / 1000, npar, dense_params)
        @printf("                     stations x∈[%.0f, %.0f] m  T4 est=%.2f GB / %.0f GB  %s\n",
                xs, xe, mem, T4_GB, mem < 0.6 * T4_GB ? "OK" : (mem < T4_GB ? "tight" : "OVERFLOW"))
        push!(rows, (; c..., npar, mem, dense_params, xs, xe, stride))
        @printf(io_buf,
                "%s  nx=%d dx=%.1f  params=%d  projector_Dense≈%d  T4_est=%.2fGB  x∈[%.0f,%.0f]\n",
                c.name, c.nx, c.dx, npar, dense_params, mem, xs, xe)
    end
    println()
    flush(stdout)

    println("── Forward timing ($N_SAMPLES samples, TE+TM) ──")
    timings = []
    for c in CANDIDATES
        println()
        println("• ", c.name, "  nx=", c.nx, "  dx=", c.dx, " m")
        flush(stdout)
        t = time_forwards(c.nx, c.dx; n=N_SAMPLES)
        eta250 = t.mean_tot * N_TARGET
        eta300 = t.mean_tot * N_TARGET_ALT
        @printf("  mean gen=%.2fs  mean fwd=%.2fs ± %.2fs  mean tot=%.2fs\n",
                t.mean_gen, t.mean_fwd, t.std_fwd, t.mean_tot)
        @printf("  ETA n=%d:  %.1f min   n=%d:  %.1f min\n",
                N_TARGET, eta250 / 60, N_TARGET_ALT, eta300 / 60)
        push!(timings, (; c..., t..., eta250, eta300))
        @printf(io_buf,
                "%s  solver=(%d,%d)  mean_fwd=%.2fs  mean_tot=%.2fs  ETA_n%d=%.1fmin\n",
                c.name, t.n_z, t.n_y, t.mean_fwd, t.mean_tot, N_TARGET, eta250 / 60)
        flush(stdout)
    end

    # Decision: prefer dx=80 unless dx=50 is clearly cheap and T4-safe.
    t80 = timings[2]
    t50 = timings[3]
    t160 = timings[1]
    mem80 = rows[2].mem
    mem50 = rows[3].mem
    ratio80 = t80.mean_fwd / max(t160.mean_fwd, 1e-6)
    ratio50 = t50.mean_fwd / max(t160.mean_fwd, 1e-6)

    pick = "dx=80 m (nx=240)"
    reason = "2× cell count, T4-safe, cheaper first diagnostic than dx=50."
    if mem50 > 0.85 * T4_GB
        reason = "dx=50 T4 estimate is tight/overflow; start at 80 m."
    elseif t50.eta250 > 3 * t80.eta250 && t80.eta250 < 90 * 60
        reason = "dx=50 forward ETA is >3× dx=80; start at 80 m."
    elseif t50.mean_fwd < 1.3 * t80.mean_fwd && mem50 < 0.5 * T4_GB
        pick = "dx=50 m (nx=384)"
        reason = "dx=50 is only marginally slower than 80 m and T4-safe; take the finer grid."
    end

    println()
    println("═" ^ 78)
    println(" DECISION: ", pick)
    println(" Reason:    ", reason)
    @printf(" fwd scaling vs dx=160:  80 m ×%.2f   50 m ×%.2f\n", ratio80, ratio50)
    println("═" ^ 78)

    println(io_buf)
    println(io_buf, "DECISION: ", pick)
    println(io_buf, "Reason: ", reason)
    @printf(io_buf, "fwd vs dx=160: 80m ×%.2f  50m ×%.2f\n", ratio80, ratio50)

    open(out_path, "w") do f
        write(f, String(take!(io_buf)))
    end
    println("  wrote ", out_path)
    return pick
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
