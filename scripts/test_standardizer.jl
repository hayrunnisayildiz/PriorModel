#!/usr/bin/env julia
#=
Validate MT input standardization (COMMEMI 11×7 → canonical U-Net survey).

Usage (from project root):
    julia --project=. scripts/test_standardizer.jl
=#

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using Printf
using Statistics
using HDF5
using MTGeophysics

include(joinpath(ROOT, "src", "synthetic", "MTInputStandardizer.jl"))
using .MTInputStandardizer:
    standardize_mt_input, pack_te_response,
    CANONICAL_N_STATIONS, CANONICAL_PERIODS, CANONICAL_MESH, canonical_station_positions
using .MTInputStandardizer.MTMeshParams:
    DEFAULT_MESH, UNET_MESH, n_periods, station_positions

include(joinpath(ROOT, "src", "inference", "export_prior_to_mtgeophysics.jl"))
using .ExportPriorToMTGeophysics: load_trained_model, predict_prior

const COMMEMI_OBS = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
const TRAIN_H5    = joinpath(ROOT, "data", "synthetic", "train_pairs.h5")
const CKPT_V3     = joinpath(ROOT, "models", "mid_scale_prior_v3.jld2")
const CKPT_V2     = joinpath(ROOT, "models", "mid_scale_prior_v2.jld2")
const CKPT_V1     = joinpath(ROOT, "models", "mid_scale_prior.jld2")
const PRIOR_RMS_V2_NARROW = 0.075  # bilinear vs nearest prior RMS on the narrow-T v2 mesh
const PRIOR_RMS_V3_PERIOD = 0.072  # bilinear vs nearest prior RMS after period-band expansion (v3)

function _h5_survey_shape(path::String)
    isfile(path) || return nothing
    h5open(path, "r") do f
        haskey(f, "X") || return nothing
        sz = size(f["X"])
        a = HDF5.attributes(f)
        n_st = haskey(a, "n_stations") ? Int(read(a["n_stations"])) : Int(sz[1])
        n_pe = haskey(a, "periods") ? length(read(a["periods"])) : Int(sz[2])
        return (n_st, n_pe, sz)
    end
end

function print_step1_report()
    train_from_unet = (UNET_MESH.n_stations, n_periods(UNET_MESH))
    default_mesh    = (DEFAULT_MESH.n_stations, n_periods(DEFAULT_MESH))
    h5info = _h5_survey_shape(TRAIN_H5)
    train_actual = h5info === nothing ? train_from_unet : (h5info[1], h5info[2])
    test_target  = train_actual  # evaluate_mid_scale_v2 / generate_prior use checkpoint = training mesh

    consistent = train_actual == test_target == train_from_unet
    verdict = consistent ? "TUTARLI" : "TUTARSIZ"

    println("═" ^ 72)
    println(" ADIM 1 — eğitim grid vs test-time interpolasyon hedefi")
    println("═" ^ 72)
    @printf("  MeshParams.DEFAULT_MESH (COMMEMI solver):  (%d, %d)\n",
            default_mesh[1], default_mesh[2])
    @printf("  MeshParams.UNET_MESH / CANONICAL:          (%d, %d)\n",
            train_from_unet[1], train_from_unet[2])
    if h5info === nothing
        println("  build_train_pairs.jl / train_pairs.h5:      (dosya yok; UNET_MESH varsayılanı)")
    else
        @printf("  build_train_pairs.jl / train_pairs.h5:      (%d, %d)  X=%s\n",
                h5info[1], h5info[2], h5info[3])
    end
    @printf("  test-time interpolasyon hedefi:             (%d, %d)",
            test_target[1], test_target[2])
    println("  (checkpoint MeshParams = eğitim ızgarası)")
    println()
    @printf("  eğitim: (%d,%d), test-time interpolasyon hedefi: (%d,%d) — %s\n",
            train_actual[1], train_actual[2], test_target[1], test_target[2], verdict)
    if default_mesh != train_from_unet
        println("  not: DEFAULT_MESH (11,7) ağ girdisi değildir; UNET_MESH (30,20) kanoniktir.")
    end
    xs = station_positions(UNET_MESH)
    @printf("  canonical x: %d stations, [%.1f, %.1f] m  (cell-centre stride, MeshParams.station_positions)\n",
            length(xs), first(xs), last(xs))
    Tmin, Tmax = extrema(CANONICAL_PERIODS)
    @printf("  canonical T: %d periods, [%.4g, %.4g] s  (10.^range(-3,3,length=%d) via UNET_MESH)\n",
            length(CANONICAL_PERIODS), Tmin, Tmax, length(CANONICAL_PERIODS))
    Tmin <= 1e-3 + 1e-12 && Tmax >= 1e3 - 1e-12 ||
        error("CANONICAL_PERIODS did not pick up the wide field-MT range T∈[1e-3,1e3] s")
    @assert CANONICAL_N_STATIONS == UNET_MESH.n_stations
    @assert length(CANONICAL_PERIODS) == n_periods(UNET_MESH)
    @assert canonical_station_positions() == station_positions(UNET_MESH)
    println("  CANONICAL_* ↔ UNET_MESH assertion: OK")
    return verdict, train_actual, test_target
end

function load_commemi()
    isfile(COMMEMI_OBS) || error("COMMEMI obs not found: $COMMEMI_OBS")
    return MTGeophysics.load_data2d(COMMEMI_OBS)
end

"""
`T_ood` is true iff COMMEMI periods stick out of the *training* range
(that is the original OOD bug). Interpolation edge-clamp (`T_clamp`) is the
opposite direction: U-Net periods that COMMEMI does not observe.
"""
function period_ood_flags(src_T::AbstractVector, tgt_T::AbstractVector)
    smin, smax = extrema(Float64.(src_T))
    tmin, tmax = extrema(Float64.(tgt_T))
    T_ood   = smin < tmin - 1e-12 || smax > tmax + 1e-12
    T_clamp = tmin < smin - 1e-12 || tmax > smax + 1e-12
    return (; T_ood, T_clamp, src=(smin, smax), tgt=(tmin, tmax))
end

"""
Per-station x-axis OOD vs the training survey, analogous to `period_ood_flags`.

`x_src_ood[i]` is true iff COMMEMI station `i` falls outside the canonical
interval `[tgt_min, tgt_max]` (the training coverage). Those stations would
map to the nearest boundary under edge-clamp.

`x_query_clamp` is the interpolator direction: true iff the U-Net query
locations extend outside the *observed* COMMEMI span (edge-repeat on queries).
"""
function station_ood_analysis(src_x::AbstractVector, tgt_x::AbstractVector)
    xs = Float64.(collect(src_x))
    n = length(xs)
    tmin, tmax = extrema(Float64.(tgt_x))
    smin, smax = n == 0 ? (NaN, NaN) : extrema(xs)
    inside = [(x >= tmin - 1e-12) && (x <= tmax + 1e-12) for x in xs]
    n_inside = count(inside)
    n_ood = n - n_inside
    clamps = Vector{Float64}(undef, n)
    sides = Vector{Symbol}(undef, n)
    for i in 1:n
        if xs[i] < tmin - 1e-12
            clamps[i] = tmin
            sides[i] = :left
        elseif xs[i] > tmax + 1e-12
            clamps[i] = tmax
            sides[i] = :right
        else
            clamps[i] = xs[i]
            sides[i] = :inside
        end
    end
    spacings = n >= 2 ? diff(sort(xs)) : Float64[]
    x_query_clamp = n == 0 ? true :
        (tmin < smin - 1e-12 || tmax > smax + 1e-12)
    return (; src_x=xs, n, n_inside, n_ood, inside, clamps, sides, spacings,
            src=(smin, smax), tgt=(tmin, tmax), x_query_clamp)
end

function _side_tr(side::Symbol)
    side === :left  && return "sol sınır"
    side === :right && return "sağ sınır"
    return "aralık içinde"
end

"""
Print ADIM 1–3 of the x-axis OOD check (canonical range, COMMEMI stations,
per-station inside/outside, optional bilinear-vs-nearest prior RMS).

`prior_rms` is `nothing` when the impact test is skipped (no OOD, or no
checkpoint). Returns the analysis NamedTuple.
"""
function print_x_axis_ood_report(data;
                                 tgt_x::AbstractVector = station_positions(UNET_MESH),
                                 prior_rms::Union{Nothing,Float64} = nothing,
                                 input_rms::Union{Nothing,Float64} = nothing)
    src_x = Float64.(collect(data.receivers))
    ana = station_ood_analysis(src_x, tgt_x)

    println()
    println("═" ^ 72)
    println(" ADIM 2 — X-ekseni OOD (istasyon konumu)")
    println("═" ^ 72)

    println()
    println("  — mevcut durum (canonical vs COMMEMI .obs) —")
    @printf("  CANONICAL_MESH / UNET_MESH  station_positions: %d nokta, [%.1f, %.1f] m\n",
            length(tgt_x), ana.tgt[1], ana.tgt[2])
    @printf("  COMMEMI .obs                receivers:         %d istasyon, [%.1f, %.1f] m\n",
            ana.n, ana.src[1], ana.src[2])
    if isempty(ana.spacings)
        println("  COMMEMI istasyon aralığı:   (tek istasyon, Δx yok)")
    else
        uΔ = unique(round.(ana.spacings; digits=6))
        @printf("  COMMEMI Δx dağılımı:        n=%d  min=%.1f  median=%.1f  max=%.1f m\n",
                length(ana.spacings), minimum(ana.spacings),
                median(ana.spacings), maximum(ana.spacings))
        print("  COMMEMI Δx benzersiz:       ")
        println(join((@sprintf("%.1f", d) for d in uΔ), ", "), " m")
        print("  COMMEMI x (m):              ")
        println(join((@sprintf("%.1f", x) for x in src_x), ", "))
    end

    println()
    println("  — per-istasyon OOD (COMMEMI ⊂ canonical [$(ana.tgt[1]), $(ana.tgt[2])] m) —")
    @printf("  içeride: %d / %d    dışında (OOD, sınırda clamp): %d / %d\n",
            ana.n_inside, ana.n, ana.n_ood, ana.n)
    if ana.n_ood == 0
        println("  x_ood:  0/$(ana.n) istasyon OOD — COMMEMI eğitim x-bandının içinde")
    else
        @printf("  x_ood:  %d/%d istasyon OOD, sınırda clamp ediliyor\n",
                ana.n_ood, ana.n)
    end
    println("  x_query_clamp (U-Net ızgarası COMMEMI x-aralığı dışına taşıyor mu?): ",
            ana.x_query_clamp)

    for i in 1:ana.n
        x = src_x[i]
        if ana.inside[i]
            @printf("    istasyon %2d: x=%8.1f m  İÇİNDE\n", i, x)
        else
            @printf("    istasyon %2d: x=%8.1f m  DIŞINDA → en yakın sınır %.1f m (%s)\n",
                    i, x, ana.clamps[i], _side_tr(ana.sides[i]))
        end
    end

    println()
    println("  — etki testi (bilinear vs nearest, periyot OOD ile aynı protokol) —")
    sonuc = ana.n_ood == 0 ? "TEMİZ" : "DİKKAT GEREKLİ"
    if prior_rms === nothing && input_rms === nothing
        if ana.n_ood == 0
            println("  x_ood = 0: COMMEMI eğitim x-bandının içinde.")
        end
        println("  checkpoint yok — bilinear vs nearest prior RMS atlandı.")
        etki = "ölçülemedi (checkpoint yok)"
    else
        if ana.n_ood == 0
            println("  x_ood = 0: interpolasyon belirsizliği (RMS) yine raporlanır.")
        end
        if prior_rms !== nothing
            @printf("  bilinear vs nearest PRIOR RMS (log10 ρ grid): %.6f\n", prior_rms)
            @printf("  karşılaştırma: periyot-ekseni v3 = %.3f  (dar-T v2 = %.3f)\n",
                    PRIOR_RMS_V3_PERIOD, PRIOR_RMS_V2_NARROW)
            etki = @sprintf("bilinear vs nearest RMS = %.6f", prior_rms)
        elseif input_rms !== nothing
            etki = @sprintf("bilinear vs nearest INPUT RMS = %.6f  (prior yok)", input_rms)
        else
            etki = "ölçülemedi"
        end
        if input_rms !== nothing
            @printf("  bilinear vs nearest INPUT RMS: %.6f  (log10ρ+phase tensor)\n", input_rms)
        end
    end

    println()
    println("  ═══════════════════════════════════════")
    println("   X-ekseni OOD Kontrolü")
    println("  ═══════════════════════════════════════")
    @printf("   Canonical aralık:     [%.1f, %.1f] m\n", ana.tgt[1], ana.tgt[2])
    @printf("   COMMEMI istasyonları: [%.1f, %.1f] m\n", ana.src[1], ana.src[2])
    @printf("   OOD istasyon sayısı:  %d / %d\n", ana.n_ood, ana.n)
    println("   Etki (varsa):          ", etki)
    println("   Sonuç:                 ", sonuc)
    println("  ═══════════════════════════════════════")
    return ana
end

function pick_checkpoint()
    for p in (CKPT_V3, CKPT_V2, CKPT_V1)
        isfile(p) && return p
    end
    return ""
end

function main()
    verdict, train_xy, test_xy = print_step1_report()

    println()
    println("═" ^ 72)
    println(" ADIM 4 — COMMEMI standardize + bilinear vs nearest")
    println("═" ^ 72)

    data = load_commemi()
    raw = pack_te_response(data.rho_xy, data.phase_xy)
    @printf("  COMMEMI raw tensor: %s  (%d stations × %d periods)\n",
            size(raw), length(data.receivers), length(data.periods))

    flags = period_ood_flags(data.periods, CANONICAL_PERIODS)
    @printf("  COMMEMI T∈[%.4g, %.4g] s\n", flags.src[1], flags.src[2])
    @printf("  eğitim   T∈[%.4g, %.4g] s  (UNET_MESH / CANONICAL_PERIODS)\n",
            flags.tgt[1], flags.tgt[2])
    println("  T_ood   (COMMEMI ⊂ eğitim aralığı?): ",
            flags.T_ood ? "true  ← HÂLÂ OOD, clamp olacak" : "false ← COMMEMI eğitim bandının içinde")
    println("  T_clamp (eğitim ızgarası COMMEMI dışına taşıyor mu?): ", flags.T_clamp)
    flags.T_ood && error("T_ood=true: COMMEMI periods still fall outside UNET_MESH; range expansion failed")
    println("  assertion T_ood=false: OK")

    x_bi = standardize_mt_input(data; method=:bilinear)
    x_nn = standardize_mt_input(data; method=:nearest)
    println("  standardize bilinear shape: ", size(x_bi))
    println("  standardize nearest  shape: ", size(x_nn))
    size(x_bi) == (CANONICAL_N_STATIONS, length(CANONICAL_PERIODS), 2) ||
        error("bilinear tensor is not canonical: $(size(x_bi))")

    input_rms = Float64(sqrt(mean((x_bi .- x_nn).^2)))
    @printf("  bilinear vs nearest INPUT RMS: %.6f  (log10ρ+phase tensor)\n", input_rms)

    ckpt = pick_checkpoint()
    prior_rms = nothing
    mp = nothing
    if isempty(ckpt)
        println()
        println("  Checkpoint yok — prior karşılaştırması atlandı.")
        println("  Model 11×7'yi kabul etmez (MTInputProjector sabit boyutlu);")
        println("  doğrudan ham COMMEMI vermek DimensionMismatch üretir.")
    else
        model, ps, st, mp = load_trained_model(ckpt)
        @printf("  checkpoint: %s\n", ckpt)
        @printf("  checkpoint survey: (%d, %d)  grid=(nz=%d, nx=%d)  dx=%.1f m\n",
                mp.n_stations, length(mp.periods), mp.nz, mp.nx, Float64(mp.dx))

        mesh_ok = Int(mp.nx) == UNET_MESH.nx &&
                  isapprox(Float64(mp.dx), UNET_MESH.dx; rtol=1e-9)
        if !mesh_ok
            @warn "checkpoint MeshParams is not the current UNET_MESH; skipping prior RMS" checkpoint_dx=Float64(mp.dx) unet_dx=UNET_MESH.dx
        else
            x_bi_mp = standardize_mt_input(data; mp=mp, method=:bilinear)
            x_nn_mp = standardize_mt_input(data; mp=mp, method=:nearest)

            pred_bi = predict_prior(model, ps, st, x_bi_mp)
            pred_nn = predict_prior(model, ps, st, x_nn_mp)
            prior_rms = Float64(sqrt(mean((pred_bi .- pred_nn).^2)))
            @printf("  bilinear vs nearest PRIOR RMS (log10 ρ grid): %.6f\n", prior_rms)
            @printf("  prior bilinear: mean=%.3f std=%.3f range=[%.3f, %.3f]\n",
                    mean(pred_bi), std(pred_bi), minimum(pred_bi), maximum(pred_bi))
            @printf("  prior nearest:  mean=%.3f std=%.3f range=[%.3f, %.3f]\n",
                    mean(pred_nn), std(pred_nn), minimum(pred_nn), maximum(pred_nn))

            rel = prior_rms / max(std(pred_bi), 1.0f-6)
            @printf("  önceki (v2, dar periyot) bilinear vs nearest PRIOR RMS: %.3f\n",
                    PRIOR_RMS_V2_NARROW)
            if prior_rms < PRIOR_RMS_V2_NARROW
                println("  yorum: PRIOR RMS düştü — geniş periyot bandında interpolasyon daha kararlı.")
            else
                println("  yorum: PRIOR RMS düşmedi; interpolasyon belirsizliği hâlâ var (T_ood ayrı kontrol).")
            end
            if rel > 0.05
                println("  not: RMS > 5% of prior std — bilinear vs nearest farkı hâlâ anlamlı.")
            else
                println("  not: küçük fark (RMS ≤ 5% of prior std).")
            end
        end

        println()
        println("  Doğrudan (11×7) modele verme denemesi:")
        try
            predict_prior(model, ps, st, raw)
            println("    BEKLENMEDİK: model 11×7 kabul etti (boyut esnek).")
        catch e
            println("    ", sprint(showerror, e))
            println("    model boyut-esnek değil; COMMEMI her zaman standardize edilmeli.")
        end
    end

    tgt_x = station_positions(UNET_MESH)
    print_x_axis_ood_report(data; tgt_x=tgt_x, prior_rms=prior_rms, input_rms=input_rms)

    println()
    println("Done. ADIM 1: eğitim $(train_xy) vs test-time $(test_xy) — $verdict")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
