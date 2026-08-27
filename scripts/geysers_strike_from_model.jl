#!/usr/bin/env julia
#=
geysers_strike_from_model.jl — structural strike from the Northwest Geysers 3-D
resistivity model, used to break the phase-tensor 90° ambiguity.

Phase-tensor analysis (scripts/geysers_dimensionality.jl) gives a consistent
strike in the 1–10 s band of N53°E or N143°E; without tipper the two cannot be
separated. The direction along which log10(ρ) structures elongate in the
preferred 3-D model *is* the geoelectric strike (the λ2 eigenvector of the
horizontal structure tensor).

Usage (from project root):
    julia --project=. scripts/geysers_strike_from_model.jl
    julia --project=. scripts/geysers_strike_from_model.jl <model.rho> <out_dir>

Outputs:
    results/geysers_model_strike.md
    results/geysers_model_strike.png
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics
using LinearAlgebra
using Dates

using MTGeophysics
using MTGeophysics: load_model_modem, load_data_modem

const DEFAULT_MODEL_PATH = joinpath(ROOT, "data", "geysers", "model",
                                    "geysers_preferred_model.rho")
const DEFAULT_DATA_PATH = joinpath(ROOT, "data", "geysers", "model",
                                   "geysers_modEM_data_file.dat")
const DEFAULT_OUT_DIR = joinpath(ROOT, "results")

# Air / topography encoding in this ModEM model is ρ ≈ 9.99979e11 Ω·m even
# though the header writes nzAir = 0 (see geysers_modem_probe.jl). Threshold
# 1e8 is well above any earth cell (earth max ~ 3.5e5 Ω·m) and well below air.
const AIR_RHO_THRESHOLD = 1.0e8

# Uniform-core cell size. The Mackie/ModEM mesh has a 200 m block with two
# 196/204 m cells; m.npad treats those as padding and is NOT used here.
const CORE_DX_M = 200.0
const CORE_DX_TOL = 0.05          # ±5 %

# Station footprint from the ModEM .dat (X north, Y east) plus 1 km margin.
const STATION_X = (-4100.0, 4100.0)
const STATION_Y = (-5200.0, 5200.0)
const CORE_MARGIN_M = 1000.0
const CORE_X = (STATION_X[1] - CORE_MARGIN_M, STATION_X[2] + CORE_MARGIN_M)
const CORE_Y = (STATION_Y[1] - CORE_MARGIN_M, STATION_Y[2] + CORE_MARGIN_M)

const DEPTH_MAX_M = 10_000.0
const DEPTH_MEAN_MAX_M = 5_000.0
const MIN_TENSOR_CELLS = 30

# Phase-tensor candidates (1–10 s band; axial, 90° apart).
const PT_A_DEG = 53.0
const PT_B_DEG = 143.0

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

"""True if a cell width belongs to the 200 m ± 5 % uniform block."""
is_core_spacing(d::Real) = abs(Float64(d) - CORE_DX_M) / CORE_DX_M <= CORE_DX_TOL

"""
    axial_mean(angles_deg; weights=nothing) -> (mean_deg, R)

Circular mean of an *axial* quantity (strike: θ ≡ θ+180°). Double the angle,
average `exp(i·2θ)`, halve the argument (Mardia & Jupp 2000, *Directional
Statistics*, §9.2). `R ∈ [0,1]` is the mean resultant length. Mean is folded
into `[0, 180)`. Unweighted unless `weights` is given.
"""
function axial_mean(angles_deg::AbstractVector{<:Real};
                    weights::Union{Nothing,AbstractVector{<:Real}} = nothing)
    θ = Float64[]
    w = Float64[]
    for (k, a) in enumerate(angles_deg)
        isfinite(a) || continue
        wk = weights === nothing ? 1.0 : Float64(weights[k])
        (isfinite(wk) && wk > 0) || continue
        push!(θ, a)
        push!(w, wk)
    end
    isempty(θ) && return (NaN, NaN)
    W = sum(w)
    C = sum(w .* cosd.(2 .* θ)) / W
    S = sum(w .* sind.(2 .* θ)) / W
    R = hypot(C, S)
    μ = mod(rad2deg(atan(S, C)) / 2, 180.0)
    return (μ, R)
end

"""Axial circular distance on [0, 180): min(|Δ|, 180−|Δ|)."""
function axial_distance(a::Real, b::Real)::Float64
    d = abs(mod(Float64(a) - Float64(b), 180.0))
    return min(d, 180.0 - d)
end

"""
    strike_from_lambda2(vx, vy) -> degrees in [0, 180)

Convert the λ2 eigenvector in model coordinates (vx along +X = north, vy
along +Y = east) to a geographic strike, clockwise from north, reported as
Nxx°E in `[0, 180)`. This is the *elongation* direction, not the gradient.
"""
function strike_from_lambda2(vx::Real, vy::Real)::Float64
    n = hypot(vx, vy)
    n > 0 || return NaN
    # atan2(east, north) = geographic azimuth clockwise from north.
    return mod(rad2deg(atan(vy / n, vx / n)), 180.0)
end

"""
    structure_tensor_slice(f, earth, cx, cy, ix, iy) -> NamedTuple

Horizontal structure tensor of the 2-D scalar field `f = log10(ρ)` on one
depth slice (Förstner 1986; Bigun & Granlund 1987; Weickert 1998).

    T = [ ⟨gx²⟩    ⟨gx gy⟩ ]
        [ ⟨gx gy⟩  ⟨gy²⟩   ]

`gx = ∂f/∂x`, `gy = ∂f/∂y` by central differences on the actual cell-centre
spacing. Only earth cells whose four orthogonal neighbours are also earth
enter the average — differencing across the air/topo interface (~1e12 Ω·m)
would let topography dominate T.

Eigen-decomposition of the 2×2 symmetric T:
* λ1 ≥ λ2 ≥ 0
* eigenvector of λ1 = strongest gradient (perpendicular to structures)
* eigenvector of λ2 = weakest gradient = elongation / strike

Consistency `(λ1 − λ2) / (λ1 + λ2) ∈ [0, 1]` (1 = one dominant direction).
"""
function structure_tensor_slice(f::AbstractMatrix{<:Real},
                                earth::AbstractMatrix{Bool},
                                cx::AbstractVector{<:Real},
                                cy::AbstractVector{<:Real},
                                ix::AbstractVector{Int},
                                iy::AbstractVector{Int})
    nx, ny = size(f)
    Txx = 0.0
    Txy = 0.0
    Tyy = 0.0
    n = 0
    @inbounds for i in ix, j in iy
        (1 < i < nx && 1 < j < ny) || continue
        earth[i, j] || continue
        earth[i - 1, j] && earth[i + 1, j] || continue
        earth[i, j - 1] && earth[i, j + 1] || continue
        gx = (f[i + 1, j] - f[i - 1, j]) / (cx[i + 1] - cx[i - 1])
        gy = (f[i, j + 1] - f[i, j - 1]) / (cy[j + 1] - cy[j - 1])
        (isfinite(gx) && isfinite(gy)) || continue
        Txx += gx * gx
        Txy += gx * gy
        Tyy += gy * gy
        n += 1
    end
    n_earth = 0
    @inbounds for i in ix, j in iy
        earth[i, j] && (n_earth += 1)
    end
    if n < MIN_TENSOR_CELLS
        return (n_tensor = n, n_earth = n_earth, λ1 = NaN, λ2 = NaN,
                consistency = NaN, ratio = NaN, strike = NaN, vx = NaN, vy = NaN)
    end
    Txx /= n
    Txy /= n
    Tyy /= n
    T = Symmetric(Float64[Txx Txy; Txy Tyy])
    F = eigen(T)
    perm = sortperm(F.values; rev = true)
    λ1 = max(F.values[perm[1]], 0.0)
    λ2 = max(F.values[perm[2]], 0.0)
    v2 = F.vectors[:, perm[2]]
    tr = λ1 + λ2
    cons = tr > 0 ? (λ1 - λ2) / tr : NaN
    ratio = λ2 > 0 ? λ1 / λ2 : Inf
    strike = strike_from_lambda2(v2[1], v2[2])
    return (n_tensor = n, n_earth = n_earth, λ1 = λ1, λ2 = λ2,
            consistency = cons, ratio = ratio, strike = strike,
            vx = v2[1], vy = v2[2])
end

fmt_ne(deg) = isfinite(deg) ? @sprintf("N%.1f°E", deg) : "—"

function pick_plot_rows(rows::Vector, n::Int)
    nvalid = [r for r in rows if isfinite(r.strike)]
    isempty(nvalid) && return nvalid
    length(nvalid) <= n && return nvalid
    targets = collect(range(first(nvalid).depth_km, last(nvalid).depth_km; length = n))
    used = Set{Int}()
    out = eltype(nvalid)[]
    for t in targets
        k = 0
        best = Inf
        for (i, r) in enumerate(nvalid)
            i in used && continue
            d = abs(r.depth_km - t)
            if d < best
                best = d
                k = i
            end
        end
        k == 0 && continue
        push!(used, k)
        push!(out, nvalid[k])
    end
    sort!(out; by = r -> r.depth_km)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

function write_report(path::AbstractString, model_path::AbstractString,
                      m, air, n_air, n_earth_full,
                      ix, iy, iz, kz,
                      rows, mean_u, mean_w, mean_hi, n_hi,
                      closer, dA, dB)
    open(path, "w") do io
        println(io, "# Northwest Geysers — structural strike from the 3-D resistivity model")
        println(io)
        println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Script: `scripts/geysers_strike_from_model.jl`.")
        println(io, "Model: `", relpath(model_path, ROOT), "` via `MTGeophysics.load_model_modem`.")
        println(io, "No forward solve is called; the preferred inversion model is treated as a")
        println(io, "3-D image of log10(ρ).")
        println(io)

        println(io, "## 1. How the model was loaded")
        println(io)
        println(io, "`load_model_modem` (`MTGeophysics.jl` v0.4.2, `src/Model.jl`) wraps")
        println(io, "`read_mackie3d_model(path, block=true)`. The Geysers file is Mackie/WS")
        println(io, "`LOGE`: values are exponentiated to linear Ω·m. Cell-edge coordinates are")
        println(io)
        println(io, "```")
        println(io, "x = cumsum([0; dx]) .+ origin[1]     # +X = geographic north")
        println(io, "y = cumsum([0; dy]) .+ origin[2]     # +Y = geographic east")
        println(io, "z = cumsum([0; dz]) .+ origin[3]     # +Z = down")
        println(io, "```")
        println(io)
        println(io, "Cell centres `cx, cy, cz` are midpoints of those edges. This is the WS/ModEM")
        println(io, "frame (Kelbert et al.; `data rotated 0.0_deg clockwise from N` in the `.dat`")
        println(io, "header). Confirmed on station GZ01: south and east of the mesh origin")
        println(io, "(38.831979°N, 122.828190°W) → X = −700 m, Y = +4800 m.")
        println(io)
        @printf(io, "- Grid: nx = %d, ny = %d, nz = %d (%d cells)\n",
                m.nx, m.ny, m.nz, length(m.A))
        @printf(io, "- Origin: (%.3f, %.3f, %.3f) m\n",
                m.origin[1], m.origin[2], m.origin[3])
        @printf(io, "- Mesh top z-edge: %.3f m. z = 0 on this axis is ~sea level; the top of\n",
                m.z[1])
        println(io, "  the mesh sits at the highest topography (~1084 m above that datum).")
        println(io, "  Depth below mesh top of layer k is `cz[k] − z[1]`.")
        @printf(io, "- Header `nzAir = 0`. `m.npad` from the loader is %s — **not used**.\n",
                string(m.npad))
        println(io, "  That heuristic counts only cells whose `dx` equals the exact mid-grid")
        println(io, "  value (200 m) and therefore treats the 196 m / 204 m cells as padding.")
        println(io)

        println(io, "## 2. Air / topography mask")
        println(io)
        println(io, "The header claims no air layers, but ModEM encodes topography by filling")
        println(io, "cells above the free surface with a sentinel resistivity of ~9.99979×10¹¹ Ω·m.")
        println(io, "`load_model_modem` only NaNs values `> 1e15`, so those cells remain finite.")
        println(io, "Using them in log10(ρ) would put 12.0 into every statistic.")
        println(io)
        @printf(io, "- Threshold: ρ ≥ %.0e Ω·m (task value; earth max is ~3.5×10⁵ Ω·m).\n",
                AIR_RHO_THRESHOLD)
        @printf(io, "- Masked (air/topo) cells: **%d / %d (%.1f%%)**\n",
                n_air, length(m.A), 100 * n_air / length(m.A))
        @printf(io, "- Remaining earth cells: **%d** (expected 278,369)\n", n_earth_full)
        println(io)
        if n_earth_full == 278_369
            println(io, "Earth-cell count matches the probe / DATA_NOTES figure exactly.")
        else
            @printf(io, "Earth-cell count differs from 278,369 by %d. Investigate before using.\n",
                    n_earth_full - 278_369)
        end
        println(io)

        println(io, "## 3. Core volume (from geometry, not `m.npad`)")
        println(io)
        println(io, "Station footprint in the ModEM file: X ∈ [−4100, 4100] m (north),")
        println(io, "Y ∈ [−5200, 5200] m (east). Plus 1 km margin:")
        println(io)
        println(io, "- X ∈ [−5100, 5100] m")
        println(io, "- Y ∈ [−6200, 6200] m")
        println(io, "- Depth: native layers whose cell centres lie 0–10 km below mesh top")
        println(io)
        println(io, "A cell is in the horizontal core if **both**")
        println(io)
        println(io, "1. its `dx` and `dy` are 200 m ± 5 % (the uniform block, including the")
        println(io, "   196/204 m pair that `m.npad` would drop), and")
        println(io, "2. its centre `(cx, cy)` falls in the box above.")
        println(io)
        @printf(io, "- Core X indices i = %d … %d  (n = %d), cx = %.1f … %.1f m, dx = %.1f … %.1f m\n",
                first(ix), last(ix), length(ix),
                m.cx[first(ix)], m.cx[last(ix)],
                minimum(m.dx[ix]), maximum(m.dx[ix]))
        @printf(io, "- Core Y indices j = %d … %d  (n = %d), cy = %.1f … %.1f m, dy = %.1f … %.1f m\n",
                first(iy), last(iy), length(iy),
                m.cy[first(iy)], m.cy[last(iy)],
                minimum(m.dy[iy]), maximum(m.dy[iy]))
        @printf(io, "- Depth layers k = %d … %d  (n = %d), centre depth %.0f … %.0f m below mesh top\n",
                first(kz), last(kz), length(kz),
                first(iz), last(iz))
        @printf(io, "- Loader `m.npad` was %s; core X pad equivalent from geometry is %d + %d\n",
                string(m.npad), first(ix) - 1, m.nx - last(ix))
        println(io, "  (the 196/204 m cells sit inside the core, so geometry pad ≠ `m.npad`).")
        println(io)

        println(io, "## 4. Method")
        println(io)
        println(io, "On each native depth slice, over earth cells in the core XY window:")
        println(io)
        println(io, "1. `f = log10(ρ)` (air cells excluded).")
        println(io, "2. Horizontal gradients by central differences that honour cell sizes:")
        println(io, "   `gx = (f[i+1,j] − f[i−1,j]) / (cx[i+1] − cx[i−1])` (north),")
        println(io, "   `gy = (f[i,j+1] − f[i,j−1]) / (cy[j+1] − cy[j−1])` (east).")
        println(io, "   A cell contributes only if it and its four orthogonal neighbours are earth.")
        println(io, "   This keeps the free-surface (air/topo) gradient out of T.")
        println(io, "3. Slice-averaged structure tensor")
        println(io)
        println(io, "       T = [ ⟨gx²⟩   ⟨gx gy⟩ ]")
        println(io, "           [ ⟨gx gy⟩  ⟨gy²⟩  ]")
        println(io)
        println(io, "   (Förstner 1986; Bigun & Granlund 1987; Weickert 1998, *Anisotropic")
        println(io, "   Diffusion in Image Processing*).")
        println(io, "4. Eigen-decomposition, λ1 ≥ λ2 ≥ 0. **Strike = eigenvector of λ2**")
        println(io, "   (elongation / weakest gradient). The λ1 direction is perpendicular to")
        println(io, "   structures and is *not* the strike.")
        println(io, "5. Consistency = (λ1 − λ2) / (λ1 + λ2) ∈ [0, 1]. Also λ1/λ2 when λ2 > 0.")
        println(io, "6. `(vx, vy)` in (north, east) → geographic azimuth `atan2(vy, vx)`, folded")
        println(io, "   to `[0, 180)` because strike is axial (θ ≡ θ+180°).")
        println(io, "7. Slices with fewer than ", MIN_TENSOR_CELLS, " valid gradient cells are skipped.")
        println(io)
        println(io, "0–5 km mean: doubled-angle circular mean of the per-slice strikes")
        println(io, "(Mardia & Jupp 2000 §9.2),")
        println(io)
        println(io, "    R = |mean exp(i·2θ)| ,   θ̄ = ½ atan2(mean sin 2θ, mean cos 2θ).")
        println(io)
        println(io, "The **unweighted** mean over slices is the conservative default (each depth")
        println(io, "counts equally). A consistency-weighted mean is reported alongside it.")
        println(io, "Comparison to N53°E / N143°E uses axial distance min(|Δ|, 180−|Δ|).")
        println(io)

        println(io, "## 5. Strike versus depth")
        println(io)
        println(io, "| k | depth (km) | z centre (m) | strike | consistency | λ1/λ2 | n_earth | n_tensor |")
        println(io, "|---:|---:|---:|---|---:|---:|---:|---:|")
        for r in rows
            ratio_s = isfinite(r.ratio) ? (r.ratio < 1e6 ? @sprintf("%.2f", r.ratio) : "∞") : "—"
            cons_s = isfinite(r.consistency) ? @sprintf("%.3f", r.consistency) : "—"
            strike_s = fmt_ne(r.strike)
            @printf(io, "| %d | %.3f | %.1f | %s | %s | %s | %d | %d |\n",
                    r.k, r.depth_km, r.cz, strike_s, cons_s, ratio_s,
                    r.n_earth, r.n_tensor)
        end
        println(io)
        println(io, "`n_earth` = earth cells in the core window at that layer; `n_tensor` = cells")
        println(io, "that entered T (interior earth with earth neighbours). Depth is below mesh")
        println(io, "top; `z centre` is the loader’s `cz` (positive down, ≈ sea-level datum).")
        println(io)

        n05 = count(r -> r.depth_m <= DEPTH_MEAN_MAX_M && isfinite(r.strike), rows)
        println(io, "## 6. 0–5 km circular mean and the 90° ambiguity")
        println(io)
        @printf(io, "- Unweighted mean of %d slices with centre depth ≤ 5 km: **%s** (R = %.3f)\n",
                n05, fmt_ne(mean_u[1]), mean_u[2])
        @printf(io, "- Consistency-weighted mean, same slices: **%s** (R = %.3f)\n",
                fmt_ne(mean_w[1]), mean_w[2])
        @printf(io, "- Restricted to 1.5–5 km and C ≥ 0.3 (n = %d): **%s** (R = %.3f)\n",
                n_hi, fmt_ne(mean_hi[1]), mean_hi[2])
        println(io)
        println(io, "The 1.5–5 km window is where the tensor is well populated (full core,")
        println(io, "no remaining air) and consistency peaks (~0.68 near 3 km). Near-surface")
        println(io, "slices scatter because few earth cells remain under the topography mask.")
        println(io)
        @printf(io, "Axial distance from the unweighted 0–5 km mean to N%.0f°E: **%.1f°**\n",
                PT_A_DEG, dA)
        @printf(io, "Axial distance from the unweighted 0–5 km mean to N%.0f°E: **%.1f°**\n",
                PT_B_DEG, dB)
        println(io)
        println(io, "**Closer phase-tensor candidate: ", fmt_ne(closer), ".**")
        println(io)
        println(io, "The phase tensor cannot tell strike from strike+90° (Caldwell, Bibby &")
        println(io, "Brown 2004). The 3-D model can: structures elongate along λ2. That")
        @printf(io, "direction is %s, which is closer to **%s** (Δ = %.1f°) than to **%s**\n",
                fmt_ne(mean_u[1]), fmt_ne(closer), min(dA, dB),
                fmt_ne(closer == PT_A_DEG ? PT_B_DEG : PT_A_DEG))
        @printf(io, "(Δ = %.1f°). The residual vs N143°E is real: the model fabric is more\n",
                max(dA, dB))
        println(io, "ESE (~N117°E) than the 1–10 s phase-tensor azimuth, but it is unambiguously")
        println(io, "in the N143°E family, not N53°E. A 2-D profile should therefore be cut")
        println(io, "**perpendicular to ~N117–N143°E**, i.e. along ~N27–N53°E.")
        println(io)

        println(io, "## 7. Caveats")
        println(io)
        println(io, "- Topography is a sentinel-ρ blanket, not `nzAir`. The air mask and the")
        println(io, "  earth-neighbour rule for gradients are mandatory; without them T is")
        println(io, "  dominated by the free surface, not by geoelectric structure.")
        println(io, "- Shallow slices have fewer earth cells (the core still contains air where")
        println(io, "  the land surface is below the mesh top). Those rows show small `n_earth`.")
        println(io, "- Deep layers thicken (`dz` grows after ~1.3 km); a 10 km centre is a thick")
        println(io, "  cell, not a thin sheet.")
        println(io, "- The structure tensor on a slice is a *global* average. Local strike")
        println(io, "  variations (e.g. the steam field vs. the regional fabric) are collapsed")
        println(io, "  into one direction and one consistency.")
        println(io, "- The model is a regularised inversion, so recovered fabrics are smoothed.")
        println(io, "  Consistency is therefore a lower bound on true structural anisotropy.")
        println(io, "- Strike is axial: Nθ°E and N(θ+180)°E are the same line. The remaining")
        println(io, "  180° sense (which way is “along strike”) is not an MT observable.")
        println(io)

        println(io, "## Artefacts")
        println(io)
        println(io, "- `results/geysers_model_strike.md` (this file)")
        println(io, "- `results/geysers_model_strike.png` — log10(ρ) maps at representative")
        println(io, "  depths with elongation bars, plus a rose of per-slice strike")
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

compass_theta(az_deg::Real) = deg2rad(90 - az_deg)

function plot_strike_figure(png_path::AbstractString, m, air, ix, iy,
                            rows, plot_rows, mean_u, closer, dA, dB,
                            sta_x, sta_y)
    CM = MTGeophysics.CairoMakie
    CM.activate!()

    east = m.cy[iy] ./ 1000
    north = m.cx[ix] ./ 1000
    nplot = length(plot_rows)
    ncol = 3
    nrow_maps = max(1, cld(nplot, ncol))

    fig = CM.Figure(size = (1480, 420 * nrow_maps + 520), fontsize = 12)

    # Shared colour range from plotted earth cells.
    vals = Float64[]
    for r in plot_rows
        sl = view(m.A, ix, iy, r.k)
        mask = view(air, ix, iy, r.k)
        for (v, a) in zip(sl, mask)
            a && continue
            v > 0 && isfinite(v) && push!(vals, log10(v))
        end
    end
    if isempty(vals)
        crange = (0.0, 3.0)
    else
        lo = quantile(vals, 0.05)
        hi = quantile(vals, 0.95)
        if !(hi > lo)
            lo, hi = extrema(vals)
        end
        crange = (lo, hi)
    end
    cmap = :turbo

    for (idx, r) in enumerate(plot_rows)
        row, col = fldmod1(idx, ncol)
        ax = CM.Axis(fig[row, col];
                     xlabel = (row == nrow_maps ? "Easting Y (km)" : ""),
                     ylabel = (col == 1 ? "Northing X (km)" : ""),
                     title = @sprintf("%.2f km   %s   C=%.2f",
                                      r.depth_km, fmt_ne(r.strike), r.consistency),
                     aspect = CM.DataAspect(),
                     xticklabelrotation = π / 6)
        logslice = fill(NaN, length(ix), length(iy))
        @inbounds for (ii, i) in enumerate(ix), (jj, j) in enumerate(iy)
            air[i, j, r.k] && continue
            ρ = m.A[i, j, r.k]
            (ρ > 0 && isfinite(ρ)) || continue
            logslice[ii, jj] = log10(ρ)
        end
        # heatmap(x=east, y=north): values[ieast, inorth] at (east, north)
        CM.heatmap!(ax, east, north, permutedims(logslice);
                    colormap = cmap, colorrange = crange,
                    nan_color = (:grey85, 1.0))
        if !isempty(sta_x)
            CM.scatter!(ax, sta_y ./ 1000, sta_x ./ 1000;
                        markersize = 4, color = :white,
                        strokecolor = :black, strokewidth = 0.4)
        end
        if isfinite(r.strike)
            # Elongation bar: geographic az, unit (east, north) = (sin, cos).
            L = 0.35 * min(east[end] - east[1], north[end] - north[1])
            s, c = sincosd(r.strike)
            mx = (east[1] + east[end]) / 2
            my = (north[1] + north[end]) / 2
            CM.lines!(ax, [mx - L * s, mx + L * s], [my - L * c, my + L * c];
                      color = :white, linewidth = 4)
            CM.lines!(ax, [mx - L * s, mx + L * s], [my - L * c, my + L * c];
                      color = :black, linewidth = 2)
        end
        CM.xlims!(ax, east[1], east[end])
        CM.ylims!(ax, north[1], north[end])
    end
    CM.Colorbar(fig[1:nrow_maps, ncol + 1]; colormap = cmap, limits = crange,
                label = "log10(ρ / Ω·m)", height = CM.Relative(0.7))

    # Rose of per-slice strike (axial, mirrored), plus PT candidates.
    valid = [r for r in rows if isfinite(r.strike)]
    tick_az = 0.0:30.0:330.0
    tick_labels = ["N", "30°", "60°", "E", "120°", "150°",
                   "S", "210°", "240°", "W", "300°", "330°"]
    axr = CM.PolarAxis(fig[nrow_maps + 1, 1:2];
                       thetaticks = (compass_theta.(collect(tick_az)), tick_labels),
                       rticklabelsize = 10,
                       title = "Per-slice elongation (λ2), axial")
    rmax = 1.05
    for r in valid
        len = 0.25 + 0.75 * (isfinite(r.consistency) ? r.consistency : 0.0)
        depth_frac = clamp(r.depth_km / 10.0, 0.0, 1.0)
        col = CM.cgrad(:thermal)[depth_frac]
        for half in (0.0, 180.0)
            CM.lines!(axr,
                      fill(compass_theta(r.strike + half), 2),
                      [0.0, len];
                      color = col, linewidth = 2.0)
        end
    end
    if isfinite(mean_u[1])
        for half in (0.0, 180.0)
            CM.lines!(axr, fill(compass_theta(mean_u[1] + half), 2), [0.0, rmax];
                      color = :black, linewidth = 3.5)
        end
    end
    for (cand, col) in ((PT_A_DEG, :orangered), (PT_B_DEG, :royalblue))
        for half in (0.0, 180.0)
            CM.lines!(axr, fill(compass_theta(cand + half), 2), [0.0, rmax];
                      color = col, linewidth = 2.0, linestyle = :dash)
        end
    end

    axd = CM.Axis(fig[nrow_maps + 1, 3:ncol];
                  xlabel = "Strike (° clockwise from N)",
                  ylabel = "Depth (km, below mesh top)",
                  title = "Strike vs depth",
                  yreversed = true,
                  xticks = 0:30:180)
    if !isempty(valid)
        CM.scatter!(axd, [r.strike for r in valid], [r.depth_km for r in valid];
                    markersize = 10,
                    color = [r.consistency for r in valid],
                    colormap = :viridis, colorrange = (0, 1),
                    strokecolor = :black, strokewidth = 0.4)
        if isfinite(mean_u[1])
            CM.vlines!(axd, mean_u[1]; color = :black, linewidth = 2,
                       label = @sprintf("0–5 km mean %s", fmt_ne(mean_u[1])))
        end
        CM.vlines!(axd, PT_A_DEG; color = :orangered, linestyle = :dash, linewidth = 2,
                   label = "PT N53°E")
        CM.vlines!(axd, PT_B_DEG; color = :royalblue, linestyle = :dash, linewidth = 2,
                   label = "PT N143°E")
        CM.axislegend(axd; position = :rt, framevisible = true, labelsize = 11)
    end
    CM.xlims!(axd, 0, 180)

    subtitle = @sprintf(
        "Northwest Geysers preferred model — log10(ρ) core slices and structural strike (λ2 elongation)\n0–5 km unweighted circular mean %s (R = %.2f). Closer PT candidate: %s (Δ = %.1f° vs %.1f°). White/black bars = slice strike. Grey = air/topo (ρ ≥ 1e8 Ω·m). Dots = MT stations.",
        fmt_ne(mean_u[1]), mean_u[2], fmt_ne(closer), min(dA, dB), max(dA, dB))
    CM.Label(fig[0, 1:ncol], subtitle; fontsize = 14, padding = (0, 0, 8, 0),
             tellheight = true)
    CM.rowgap!(fig.layout, 8)
    CM.save(png_path, fig; px_per_unit = 2)
    return png_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main(args::Vector{String} = ARGS)
    model_path = length(args) >= 1 && !isempty(args[1]) ? abspath(args[1]) : DEFAULT_MODEL_PATH
    out_dir = length(args) >= 2 && !isempty(args[2]) ? abspath(args[2]) : DEFAULT_OUT_DIR
    mkpath(out_dir)
    isfile(model_path) || error("model not found: $model_path")

    println("[1/5] load_model_modem ", model_path)
    m = load_model_modem(model_path)
    @printf("      (nx, ny, nz) = (%d, %d, %d)  origin = (%.3f, %.3f, %.3f)\n",
            m.nx, m.ny, m.nz, m.origin[1], m.origin[2], m.origin[3])
    @printf("      m.npad = %s  (not used for core indexing)\n", string(m.npad))

    air = m.A .>= AIR_RHO_THRESHOLD
    n_air = count(air)
    n_earth_full = length(m.A) - n_air
    @printf("      air mask ρ ≥ %.0e: %d cells (%.1f%%), earth = %d (expect 278369)\n",
            AIR_RHO_THRESHOLD, n_air, 100 * n_air / length(m.A), n_earth_full)

    # Horizontal core from geometry.
    ix = Int[i for i in 1:m.nx
             if is_core_spacing(m.dx[i]) && CORE_X[1] - 1e-6 <= m.cx[i] <= CORE_X[2] + 1e-6]
    iy = Int[j for j in 1:m.ny
             if is_core_spacing(m.dy[j]) && CORE_Y[1] - 1e-6 <= m.cy[j] <= CORE_Y[2] + 1e-6]
    isempty(ix) && error("no core X cells — check geometry")
    isempty(iy) && error("no core Y cells — check geometry")

    z_top = m.z[1]
    depth_c = m.cz .- z_top
    kz = Int[k for k in 1:m.nz if 0.0 <= depth_c[k] <= DEPTH_MAX_M]
    iz = depth_c[kz]
    @printf("      core i=%d:%d (n=%d, cx %.1f…%.1f m, dx %.1f…%.1f)\n",
            first(ix), last(ix), length(ix), m.cx[first(ix)], m.cx[last(ix)],
            minimum(m.dx[ix]), maximum(m.dx[ix]))
    @printf("      core j=%d:%d (n=%d, cy %.1f…%.1f m, dy %.1f…%.1f)\n",
            first(iy), last(iy), length(iy), m.cy[first(iy)], m.cy[last(iy)],
            minimum(m.dy[iy]), maximum(m.dy[iy]))
    @printf("      depth k=%d:%d (n=%d, centres %.0f…%.0f m below mesh top)\n",
            first(kz), last(kz), length(kz), first(iz), last(iz))

    println("[2/5] structure tensor per depth slice")
    logA = fill(NaN, m.nx, m.ny, m.nz)
    @inbounds for i in 1:m.nx, j in 1:m.ny, k in 1:m.nz
        air[i, j, k] && continue
        ρ = m.A[i, j, k]
        (ρ > 0 && isfinite(ρ)) || continue
        logA[i, j, k] = log10(ρ)
    end

    rows = NamedTuple[]
    for k in kz
        earth = .!view(air, :, :, k)
        st = structure_tensor_slice(view(logA, :, :, k), earth, m.cx, m.cy, ix, iy)
        push!(rows, (k = k, depth_m = depth_c[k], depth_km = depth_c[k] / 1000,
                     cz = m.cz[k], st...))
        if isfinite(st.strike)
            @printf("      k=%2d  z=%.0f m  strike=%s  C=%.3f  λ1/λ2=%s  nT=%d nE=%d\n",
                    k, depth_c[k], fmt_ne(st.strike), st.consistency,
                    isfinite(st.ratio) && st.ratio < 1e6 ? @sprintf("%.2f", st.ratio) : "∞",
                    st.n_tensor, st.n_earth)
        else
            @printf("      k=%2d  z=%.0f m  SKIP nT=%d nE=%d\n",
                    k, depth_c[k], st.n_tensor, st.n_earth)
        end
    end

    sel05 = [r for r in rows if r.depth_m <= DEPTH_MEAN_MAX_M && isfinite(r.strike)]
    isempty(sel05) && error("no valid slices in 0–5 km")
    mean_u = axial_mean(Float64[r.strike for r in sel05])
    mean_w = axial_mean(Float64[r.strike for r in sel05];
                        weights = Float64[r.consistency for r in sel05])
    sel_hi = [r for r in sel05 if r.depth_km >= 1.5 &&
              isfinite(r.consistency) && r.consistency >= 0.3]
    mean_hi = axial_mean(Float64[r.strike for r in sel_hi])
    dA = axial_distance(mean_u[1], PT_A_DEG)
    dB = axial_distance(mean_u[1], PT_B_DEG)
    closer = dA <= dB ? PT_A_DEG : PT_B_DEG
    @printf("\n      0–5 km unweighted  %s  R=%.3f  (n=%d)\n",
            fmt_ne(mean_u[1]), mean_u[2], length(sel05))
    @printf("      0–5 km weighted    %s  R=%.3f\n", fmt_ne(mean_w[1]), mean_w[2])
    @printf("      1.5–5 km C≥0.3     %s  R=%.3f  (n=%d)\n",
            fmt_ne(mean_hi[1]), mean_hi[2], length(sel_hi))
    @printf("      Δ(N%.0f°E)=%.1f°  Δ(N%.0f°E)=%.1f°  → closer %s\n",
            PT_A_DEG, dA, PT_B_DEG, dB, fmt_ne(closer))

    sta_x = Float64[]
    sta_y = Float64[]
    if isfile(DEFAULT_DATA_PATH)
        d = load_data_modem(DEFAULT_DATA_PATH)
        sta_x = collect(Float64.(d.x))
        sta_y = collect(Float64.(d.y))
    end

    md_path = joinpath(out_dir, "geysers_model_strike.md")
    png_path = joinpath(out_dir, "geysers_model_strike.png")

    println("[3/5] writing ", relpath(md_path, ROOT))
    write_report(md_path, model_path, m, air, n_air, n_earth_full,
                 ix, iy, iz, kz, rows, mean_u, mean_w, mean_hi, length(sel_hi),
                 closer, dA, dB)

    println("[4/5] writing ", relpath(png_path, ROOT))
    plot_rows = pick_plot_rows(rows, 6)
    plot_strike_figure(png_path, m, air, ix, iy, rows, plot_rows,
                       mean_u, closer, dA, dB, sta_x, sta_y)

    println("[5/5] done")
    println("      ", md_path)
    println("      ", png_path)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
