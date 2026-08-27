#!/usr/bin/env julia
#=
geysers_modem_probe.jl — GÖREV 1: MTGeophysics.jl ↔ Northwest Geysers ModEM köprüsü.

Amaç: USGS Northwest Geysers (Peacock et al. 2020, DOI 10.5066/P94D21UL)
referans modelini (`geysers_preferred_model.rho`) ve ModEM veri dosyasını
(`geysers_modEM_data_file.dat`) MTGeophysics.jl'in KENDİ okuyucularıyla
okumayı dener. Hiçbir şey yazmaz, sadece raporlar.

Kullanılan MTGeophysics.jl v0.4.2 API'leri (src/Model.jl, src/Data.jl):
    load_model_modem(path)  -> Model      (read_mackie3d_model sarmalayıcısı)
    read_mackie3d_model(path, block=true) -> (dx, dy, dz, A, nzAir, type, origin, rotation)
    load_data_modem(path)   -> Data

Kullanım (proje kökünden):
    julia --project=. scripts/geysers_modem_probe.jl
    julia --project=. scripts/geysers_modem_probe.jl <model.rho> <data.dat>
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics

using MTGeophysics: load_model_modem, read_mackie3d_model, load_data_modem

const DEFAULT_MODEL_PATH = joinpath(ROOT, "data", "geysers", "model",
                                    "geysers_preferred_model.rho")
const DEFAULT_DATA_PATH = joinpath(ROOT, "data", "geysers", "model",
                                   "geysers_modEM_data_file.dat")

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

section(title::AbstractString) = println("\n", "="^78, "\n", title, "\n", "="^78)

"""Bir kenar-uzunluğu vektörünün özeti: eşsiz değerler ve çekirdek adım."""
function spacing_summary(name::AbstractString, d::AbstractVector{<:Real})
    u = sort(unique(round.(Float64.(d); digits = 3)))
    @printf("  %-3s : n=%d  min=%.1f m  max=%.1f m  toplam=%.1f m  (%d eşsiz değer)\n",
            name, length(d), minimum(d), maximum(d), sum(d), length(u))
    # Çekirdek (padding olmayan) adım = en sık görülen kenar uzunluğu.
    counts = Dict{Float64,Int}()
    for v in round.(Float64.(d); digits = 3)
        counts[v] = get(counts, v, 0) + 1
    end
    mode_val, mode_cnt = argmax(last, collect(counts))
    @printf("        çekirdek adım (mod) = %.1f m, %d hücrede\n", mode_val, mode_cnt)
    return nothing
end

"""Log10 aralığı da dahil özdirenç istatistiği (NaN'leri atar)."""
function resistivity_summary(A::AbstractArray{<:Real})
    finite = filter(isfinite, vec(A))
    nnan = length(A) - length(finite)
    if isempty(finite)
        println("  UYARI: sonlu özdirenç değeri yok.")
        return nothing
    end
    ρmin, ρmax = minimum(finite), maximum(finite)
    @printf("  hücre sayısı      : %d (NaN/air: %d)\n", length(A), nnan)
    @printf("  ρ aralığı         : %.4g – %.4g Ω·m\n", ρmin, ρmax)
    @printf("  log10(ρ) aralığı  : %.4f – %.4f\n", log10(ρmin), log10(ρmax))
    @printf("  ρ medyan / ortalama: %.4g / %.4g Ω·m\n", median(finite), mean(finite))
    @printf("  log10(ρ) medyan   : %.4f  (std %.4f)\n",
            median(log10.(finite)), std(log10.(finite)))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# 1) Model (.rho)
# ─────────────────────────────────────────────────────────────────────────────

function probe_model(path::AbstractString)
    section("1) MODEL — load_model_modem(\"$(basename(path))\")")
    isfile(path) || error("Model dosyası yok: $path")
    @printf("dosya boyutu: %.2f MB\n", filesize(path) / 1024^2)
    println("ham başlık satırı: ", strip(readlines(path)[2]))

    # Ham okuyucu: dönüştürülmemiş değerler + LOGE/LINEAR bayrağı.
    dx, dy, dz, A_raw, nzAir, type, origin, rotation =
        read_mackie3d_model(path, true)
    println("\n-- read_mackie3d_model (ham) --")
    @printf("  type = %s, nzAir = %d, rotation = %.3f\n", type, nzAir, rotation)
    @printf("  origin = (%.3f, %.3f, %.3f)\n", origin[1], origin[2], origin[3])
    @printf("  ham dizi boyutu = %s\n", string(size(A_raw)))
    raw_finite = filter(isfinite, vec(A_raw))
    if !isempty(raw_finite)
        @printf("  ham değer aralığı = %.5f – %.5f  (%s)\n",
                minimum(raw_finite), maximum(raw_finite), type)
    end

    # Yüksek seviye okuyucu: LOGE ise exp() uygulanmış, yani lineer Ω·m.
    m = load_model_modem(path)
    println("\n-- load_model_modem (Model struct) --")
    @printf("  (nx, ny, nz)  = (%d, %d, %d)\n", m.nx, m.ny, m.nz)
    @printf("  size(m.A)     = %s\n", string(size(m.A)))
    @printf("  npad (x, y)   = %s\n", string(m.npad))
    @printf("  origin        = (%.3f, %.3f, %.3f) m\n",
            m.origin[1], m.origin[2], m.origin[3])

    println("\n  hücre boyutları:")
    spacing_summary("dx", m.dx)
    spacing_summary("dy", m.dy)
    spacing_summary("dz", m.dz)

    @printf("\n  x kenarları : %.1f … %.1f m (kapsam %.1f km)\n",
            first(m.x), last(m.x), (last(m.x) - first(m.x)) / 1000)
    @printf("  y kenarları : %.1f … %.1f m (kapsam %.1f km)\n",
            first(m.y), last(m.y), (last(m.y) - first(m.y)) / 1000)
    @printf("  z kenarları : %.1f … %.1f m (derinlik %.1f km)\n",
            first(m.z), last(m.z), (last(m.z) - first(m.z)) / 1000)

    println("\n  özdirenç (load_model_modem çıktısı, lineer Ω·m):")
    resistivity_summary(m.A)

    air_report(m)
    return m
end

"""
    air_report(m)

`nzAir = 0` olmasına rağmen model tepesi 1e12 Ω·m hücrelerle doludur: ModEM
topografyayı toprak-üstü hücrelere sabit yüksek özdirenç atayarak kodlar.
Bu fonksiyon o örtünün ne kadar kalın olduğunu ve gerçek yer yüzeyinin hangi
katmanda başladığını ölçer — aşağı akış (2B profil çıkarma) için kritik.
"""
function air_report(m; air_threshold::Float64 = 1e11)
    println("\n-- topografya / hava örtüsü (ρ ≥ $(air_threshold) Ω·m) --")
    is_air = m.A .>= air_threshold
    @printf("  hava hücresi   : %d / %d (%.1f%%)\n",
            count(is_air), length(is_air), 100 * count(is_air) / length(is_air))

    air_vals = unique(round.(m.A[is_air]; sigdigits = 6))
    @printf("  hava özdirenci : %s Ω·m (%d eşsiz değer)\n",
            join(air_vals, ", "), length(air_vals))

    # Her (i, j) kolonu için ilk toprak hücresi.
    first_earth = fill(0, m.nx, m.ny)
    @inbounds for i in 1:m.nx, j in 1:m.ny
        k = findfirst(!, view(is_air, i, j, :))
        first_earth[i, j] = k === nothing ? 0 : k
    end
    ok = first_earth[first_earth .> 0]
    @printf("  ilk toprak katmanı k: min=%d, medyan=%d, maks=%d (tamamen hava kolon: %d)\n",
            minimum(ok), round(Int, median(ok)), maximum(ok), count(==(0), first_earth))

    # Kolon başına yer yüzeyi kotu (z pozitif aşağı, origin[3] deniz seviyesine göre).
    surf_z = [m.z[k] for k in ok]
    @printf("  yer yüzeyi z   : %.1f … %.1f m (z pozitif aşağı, origin z=%.1f)\n",
            minimum(surf_z), maximum(surf_z), m.origin[3])
    @printf("  topografya kotu: %.1f … %.1f m (deniz seviyesi üstü)\n",
            -maximum(surf_z), -minimum(surf_z))
    @printf("  hava katmanı sayısı: en fazla %d katman (%.1f m)\n",
            maximum(ok) - 1, sum(m.dz[1:(maximum(ok) - 1)]))

    # Toprak-only özdirenç: eğitim hedefi bu olacak.
    earth = filter(isfinite, m.A[.!is_air])
    if !isempty(earth)
        println("\n  TOPRAK-ONLY özdirenç (hava hücreleri hariç):")
        @printf("    hücre sayısı     : %d\n", length(earth))
        @printf("    ρ aralığı        : %.4g – %.4g Ω·m\n", minimum(earth), maximum(earth))
        @printf("    log10(ρ) aralığı : %.4f – %.4f\n",
                log10(minimum(earth)), log10(maximum(earth)))
        @printf("    log10(ρ) medyan  : %.4f (std %.4f)\n",
                median(log10.(earth)), std(log10.(earth)))
    end

    # İlk toprak katmanından itibaren katman katman özet.
    k0 = minimum(ok)
    println("\n  katmanlar k=$(k0)'dan itibaren (hava hariç medyan):")
    for k in k0:min(k0 + 15, m.nz)
        col = filter(isfinite, m.A[.!view(is_air, :, :, k), k])
        med = isempty(col) ? NaN : median(log10.(col))
        @printf("    k=%2d  dz=%8.1f m  z_top=%9.1f m  n_toprak=%6d  log10ρ_med=%7.4f\n",
                k, m.dz[k], m.z[k], length(col), med)
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# 2) Veri (.dat)
# ─────────────────────────────────────────────────────────────────────────────

function probe_data(path::AbstractString)
    section("2) VERİ — load_data_modem(\"$(basename(path))\")")
    isfile(path) || error("Veri dosyası yok: $path")
    @printf("dosya boyutu: %.2f MB\n", filesize(path) / 1024^2)
    println("ham başlık blokları:")
    for line in readlines(path)
        s = strip(line)
        (startswith(s, "#") || startswith(s, ">")) || break
        println("  ", s)
    end

    d = load_data_modem(path)
    println("\n-- Data struct --")
    @printf("  istasyon sayısı (ns) = %d\n", d.ns)
    @printf("  periyot sayısı  (nf) = %d\n", d.nf)
    @printf("  bileşenler (nr = %d)  = %s\n", d.nr, join(d.responses, ", "))
    @printf("  periyot bandı        = %.6g – %.6g s\n", minimum(d.T), maximum(d.T))
    @printf("  frekans bandı        = %.6g – %.6g Hz\n", minimum(d.f), maximum(d.f))
    @printf("  mesh origin (lat, lon) = (%.6f, %.6f)\n", d.origin[1], d.origin[2])
    @printf("  zrot (tek değer)     = %.3f\n", isempty(d.zrot) ? NaN : d.zrot[1])

    println("\n  istasyon adları: ", join(d.site, " "))

    @printf("\n  enlem  : %.6f … %.6f\n", minimum(d.loc[:, 1]), maximum(d.loc[:, 1]))
    @printf("  boylam : %.6f … %.6f\n", minimum(d.loc[:, 2]), maximum(d.loc[:, 2]))
    @printf("  model X: %.1f … %.1f m (kapsam %.2f km)\n",
            minimum(d.x), maximum(d.x), (maximum(d.x) - minimum(d.x)) / 1000)
    @printf("  model Y: %.1f … %.1f m (kapsam %.2f km)\n",
            minimum(d.y), maximum(d.y), (maximum(d.y) - minimum(d.y)) / 1000)
    @printf("  model Z: %.1f … %.1f m\n", minimum(d.z), maximum(d.z))

    println("\n  periyotlar (s):")
    for (i, T) in enumerate(d.T)
        @printf("    %2d  %12.6g", i, T)
        i % 4 == 0 && println()
    end
    d.nf % 4 == 0 || println()

    # Doluluk: her bileşen için kaç (periyot, istasyon) hücresi sonlu.
    comp_names = ("ZXX", "ZXY", "ZYX", "ZYY")
    total = d.nf * d.ns
    println("\n  empedans doluluğu (sonlu / toplam = $(d.nf)×$(d.ns) = $total):")
    for (ic, cname) in enumerate(comp_names)
        nz_ok = count(z -> isfinite(real(z)) && isfinite(imag(z)), view(d.Z, :, ic, :))
        ne_ok = count(z -> isfinite(real(z)), view(d.Zerr, :, ic, :))
        @printf("    %-4s Z: %5d (%5.1f%%)   Zerr: %5d (%5.1f%%)\n",
                cname, nz_ok, 100 * nz_ok / total, ne_ok, 100 * ne_ok / total)
    end

    ρ_ok = filter(isfinite, vec(d.ρ))
    if !isempty(ρ_ok)
        @printf("\n  görünür özdirenç ρa: %.4g – %.4g Ω·m (log10: %.3f – %.3f)\n",
                minimum(ρ_ok), maximum(ρ_ok), log10(minimum(ρ_ok)), log10(maximum(ρ_ok)))
    end
    φ_ok = filter(isfinite, vec(d.φ))
    isempty(φ_ok) || @printf("  faz φ: %.2f° – %.2f°\n", minimum(φ_ok), maximum(φ_ok))
    return d
end

# ─────────────────────────────────────────────────────────────────────────────
# 3) Model ↔ veri tutarlılığı
# ─────────────────────────────────────────────────────────────────────────────

function cross_check(m, d)
    section("3) MODEL ↔ VERİ TUTARLILIĞI")
    # ModEM'de veri X/Y istasyon koordinatları model mesh'iyle aynı çerçevede.
    @printf("model x kapsamı : %.1f … %.1f m\n", first(m.x), last(m.x))
    @printf("istasyon X      : %.1f … %.1f m\n", minimum(d.x), maximum(d.x))
    @printf("model y kapsamı : %.1f … %.1f m\n", first(m.y), last(m.y))
    @printf("istasyon Y      : %.1f … %.1f m\n", minimum(d.y), maximum(d.y))

    inside_x = all(first(m.x) .<= d.x .<= last(m.x))
    inside_y = all(first(m.y) .<= d.y .<= last(m.y))
    println("istasyonlar model x aralığında: ", inside_x)
    println("istasyonlar model y aralığında: ", inside_y)

    if !(inside_x && inside_y)
        # ModEM mesh origin'i genelde çekirdeğin köşesinde; kaydırma gerekebilir.
        @printf("  NOT: model origin (%.1f, %.1f) — istasyon merkezi (%.1f, %.1f)\n",
                m.origin[1], m.origin[2], mean(d.x), mean(d.y))
        println("  Kaydırılmış (origin merkeze alınmış) kontrol:")
        xs = m.x .- (first(m.x) + last(m.x)) / 2
        ys = m.y .- (first(m.y) + last(m.y)) / 2
        @printf("    merkezlenmiş model x: %.1f … %.1f m → istasyonlar içinde: %s\n",
                first(xs), last(xs), all(first(xs) .<= d.x .<= last(xs)))
        @printf("    merkezlenmiş model y: %.1f … %.1f m → istasyonlar içinde: %s\n",
                first(ys), last(ys), all(first(ys) .<= d.y .<= last(ys)))
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────

function main(args::Vector{String} = String[])
    model_path = length(args) >= 1 ? args[1] : DEFAULT_MODEL_PATH
    data_path = length(args) >= 2 ? args[2] : DEFAULT_DATA_PATH

    section("GÖREV 1 — MTGeophysics.jl ModEM köprü testi (Northwest Geysers)")
    println("model : ", model_path)
    println("veri  : ", data_path)

    m = probe_model(model_path)
    d = probe_data(data_path)
    cross_check(m, d)

    section("SONUÇ: her iki dosya da MTGeophysics.jl okuyucularıyla okundu.")
    return (model = m, data = d)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(copy(ARGS))
