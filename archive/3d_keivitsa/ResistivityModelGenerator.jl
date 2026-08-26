"""
    ResistivityModelGenerator

MT-only, field-independent synthetic 2-D resistivity model generator for the prior
network training pipeline. Models live on a uniform [`MeshParams`](@ref) grid as
`(nz, nx)` `Matrix{Float32}` of `log10(ρ)` in Ω·m.

# Workflow
1. [`calibration_stats`](@ref) — Keivitsa petrophysics + VLF resistivity statistics
2. [`generate_synthetic_model`](@ref) — one random section on `MeshParams`
3. [`generate_dataset`](@ref) — batch export to `model_<id>.jld2`
4. [`plot_resistivity_model`](@ref) — heatmap figure (PGM or PNG via Plots)

# Example
```julia
include("src/synthetic/ResistivityModelGenerator.jl")
using .ResistivityModelGenerator

mp = DEFAULT_MESH
calib = calibration_stats("config/dataset_config.yaml")
model = generate_synthetic_model(mp; calib=calib)
generate_dataset(5, "data/synthetic/resistivity_models"; mp=mp, calib=calib)
plot_resistivity_model(model, mp; path="figures/sample.pgm")
```
"""
module ResistivityModelGenerator

include("MeshParams.jl")
using .MTMeshParams: MeshParams, DEFAULT_MESH, validate_mesh_params

export DEFAULT_MESH

using JLD2
using Printf
using Random
using Statistics

export CalibrationStats, Log10ResistivityStats, LayerThicknessStats, ResistivityPopulation
export DEFAULT_MESH, calibration_stats, generate_synthetic_model, generate_dataset
export plot_resistivity_model, load_synthetic_model

const _KEIVITSA_STATS = joinpath(@__DIR__, "..", "analysis", "keivitsa_stats.jl")
isfile(_KEIVITSA_STATS) && include(_KEIVITSA_STATS)

# ─────────────────────────────────────────────────────────────────────────────
# Calibration statistics
# ─────────────────────────────────────────────────────────────────────────────

"""Summary of resistivity samples in `log10(Ω·m)` space."""
struct Log10ResistivityStats
    n::Int
    min::Float64
    max::Float64
    mean::Float64
    std::Float64
    p05::Float64
    p10::Float64
    p25::Float64
    p50::Float64
    p75::Float64
    p90::Float64
    p95::Float64
end

"""Typical layer / body thickness prior in metres."""
struct LayerThicknessStats
    n::Int
    p05::Float64
    p25::Float64
    p50::Float64
    p75::Float64
    p95::Float64
    lognormal_mu::Float64
    lognormal_sigma::Float64
end

"""One log10-resistivity population (conductive, host, or resistive)."""
struct ResistivityPopulation
    name::String
    mean::Float64
    std::Float64
    weight::Float64
    lo::Float64
    hi::Float64
end

"""
Keivitsa-calibrated sampling envelope from borehole `LUO_R`, VLF resistivity,
and slingram geometry metadata.
"""
struct CalibrationStats
    borehole_log10::Log10ResistivityStats
    vlf_log10::Log10ResistivityStats
    combined_log10::Log10ResistivityStats
    layer_thickness_m::LayerThicknessStats
    body_thickness_m::LayerThicknessStats
    body_width_m::LayerThicknessStats
    log10_clip::Tuple{Float64,Float64}
    populations::Vector{ResistivityPopulation}
    conductive_fraction::Float64
    corr_vertical_m::Float64
    corr_horizontal_m::Float64
    source_config::String
end

function Base.show(io::IO, c::CalibrationStats)
    b = c.combined_log10
    print(io, "CalibrationStats(n=", b.n,
          " log10=[", round(b.p05; digits=2), ",", round(b.p95; digits=2),
          "] layer_p50=", round(c.layer_thickness_m.p50; digits=1), " m",
          " clip=[", round(c.log10_clip[1]; digits=2), ",",
          round(c.log10_clip[2]; digits=2), "])")
end

function _num(x, fallback::Float64)::Float64
    (x === nothing || x isa AbstractString) && return fallback
    x isa Real && isfinite(Float64(x)) && return Float64(x)
    return fallback
end

function _log10_stats_from_dict(d::AbstractDict)::Log10ResistivityStats
    Log10ResistivityStats(
        Int(get(d, "n", 0)),
        _num(get(d, "min", nothing), NaN),
        _num(get(d, "max", nothing), NaN),
        _num(get(d, "mean", nothing), NaN),
        _num(get(d, "std", nothing), NaN),
        _num(get(d, "p05", nothing), NaN),
        _num(get(d, "p10", nothing), NaN),
        _num(get(d, "p25", nothing), NaN),
        _num(get(d, "p50", nothing), NaN),
        _num(get(d, "p75", nothing), NaN),
        _num(get(d, "p90", nothing), NaN),
        _num(get(d, "p95", nothing), NaN),
    )
end

function _layer_stats_from_dict(d::AbstractDict)::LayerThicknessStats
    LayerThicknessStats(
        Int(get(d, "n", 0)),
        _num(get(d, "p05", nothing), NaN),
        _num(get(d, "p25", nothing), NaN),
        _num(get(d, "p50", nothing), NaN),
        _num(get(d, "p75", nothing), NaN),
        _num(get(d, "p95", nothing), NaN),
        _num(get(d, "lognormal_mu", nothing), NaN),
        _num(get(d, "lognormal_sigma", nothing), NaN),
    )
end

function _population_from_dict(d::AbstractDict, name::String)::ResistivityPopulation
    rng = get(d, "log10_range", nothing)
    lo, hi = if rng isa AbstractVector && length(rng) == 2
        (_num(rng[1], NaN), _num(rng[2], NaN))
    else
        (NaN, NaN)
    end
    μ = _num(get(d, "log10_mean", nothing), NaN)
    σ = _num(get(d, "log10_std", nothing), NaN)
    !isfinite(lo) && (lo = μ - 2σ)
    !isfinite(hi) && (hi = μ + 2σ)
    ResistivityPopulation(name, μ, max(σ, 1.0e-3), _num(get(d, "weight", nothing), 1 / 3),
                          lo, hi)
end

function _combined_log10_stats(bh::Log10ResistivityStats,
                               vlf::Log10ResistivityStats)::Log10ResistivityStats
    nb, nv = bh.n, vlf.n
    n = nb + nv
    n == 0 && return Log10ResistivityStats(0, NaN, NaN, NaN, NaN,
                                           NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    w1 = nb / n
    w2 = nv / n
    μ = w1 * bh.mean + w2 * vlf.mean
    var = w1 * (bh.std^2 + (bh.mean - μ)^2) + w2 * (vlf.std^2 + (vlf.mean - μ)^2)
    Log10ResistivityStats(
        n, min(bh.min, vlf.min), max(bh.max, vlf.max), μ, sqrt(max(var, 0.0)),
        min(bh.p05, vlf.p05), min(bh.p10, vlf.p10),
        w1 * bh.p25 + w2 * vlf.p25, w1 * bh.p50 + w2 * vlf.p50,
        w1 * bh.p75 + w2 * vlf.p75, max(bh.p90, vlf.p90), max(bh.p95, vlf.p95),
    )
end

"""
    calibration_stats(dataset_config_path) -> CalibrationStats

Extract log10-resistivity and layer-thickness statistics from Keivitsa field data
under `database/3_DRILLINGS` (`LUO_R`) and VLF resistivity paths in the dataset
YAML. Uses `KeivitsaStats.compute_priors` internally.
"""
function calibration_stats(dataset_config_path::AbstractString)::CalibrationStats
    isfile(_KEIVITSA_STATS) || error("KeivitsaStats not found: $(_KEIVITSA_STATS)")
    @isdefined(KeivitsaStats) || error("KeivitsaStats failed to load from $(_KEIVITSA_STATS)")
    cfg_path = abspath(dataset_config_path)
    isfile(cfg_path) || error("dataset config not found: $(cfg_path)")

    priors = KeivitsaStats.compute_priors(KeivitsaStats.StatsOptions(config_path=cfg_path))
    st = priors["statistics"]
    gen = priors["generator_priors"]
    res = st["resistivity"]

    bh = _log10_stats_from_dict(res["borehole_LUO_R_log10"])
    vlf = _log10_stats_from_dict(res["vlf_log10"])
    combined = _combined_log10_stats(bh, vlf)

    layer = _layer_stats_from_dict(gen["layer_thickness_m"])
    anom = gen["anomaly"]
    body_t = _layer_stats_from_dict(anom["vertical_thickness_m"])
    body_w = _layer_stats_from_dict(anom["across_strike_width_m"])

    clip_node = get(gen, "clip_ohm_m", [1.0, 1.0e5])
    clip_lo = log10(_num(clip_node[1], 1.0))
    clip_hi = log10(_num(clip_node[2], 1.0e5))

    pops = ResistivityPopulation[]
    for label in ("conductive_body", "host_rock", "resistive_body")
        node = get(gen, label, nothing)
        node isa AbstractDict || continue
        isfinite(_num(get(node, "log10_mean", nothing), NaN)) || continue
        push!(pops, _population_from_dict(node, label))
    end

    corr = get(gen, "correlation_length_m", Dict{String,Any}())
    cond_frac = _num(get(anom, "volume_fraction_conductive", nothing), 0.15)

    return CalibrationStats(
        bh, vlf, combined, layer, body_t, body_w,
        (clip_lo, clip_hi), pops, cond_frac,
        _num(get(corr, "vertical_variogram_range", nothing), 20.0),
        _num(get(corr, "horizontal_variogram_range", nothing), 150.0),
        cfg_path,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic model generation
# ─────────────────────────────────────────────────────────────────────────────

const _GENERIC_LOG10 = (0.3, 4.7)
const _GENERIC_LAYER_CELLS = (2, 8)
const _GENERIC_CLIP = (0.0, 5.0)

_uniform(rng::AbstractRNG, lo::Real, hi::Real)::Float64 =
    hi <= lo ? Float64(lo) : Float64(lo) + rand(rng) * (Float64(hi) - Float64(lo))

_randint(rng::AbstractRNG, r::Tuple{Int,Int})::Int =
    r[2] <= r[1] ? r[1] : rand(rng, r[1]:r[2])

function _grid_coords(mp::MeshParams)
    xs = [(i - 0.5) * mp.dx for i in 1:mp.nx]
    zs = [(k - 0.5) * mp.dz for k in 1:mp.nz]
    return xs, zs
end

"""Unit-variance 1-D Gaussian random field by cosine synthesis."""
function random_field(rng::AbstractRNG, xs::AbstractVector{<:Real};
                      corr::Real, n_modes::Int=64)::Vector{Float64}
    scaled = collect(Float64, xs) ./ max(Float64(corr), 1.0e-9)
    out = zeros(length(scaled))
    for _ in 1:n_modes
        k = randn(rng)
        φ = 2π * rand(rng)
        @inbounds for i in eachindex(scaled)
            out[i] += cos(k * scaled[i] + φ)
        end
    end
    σ = std(out)
    σ > 0 && (out ./= σ)
    out .-= mean(out)
    return out
end

"""Unit-variance 2-D Gaussian random field."""
function random_field(rng::AbstractRNG, xs::AbstractVector{<:Real},
                      zs::AbstractVector{<:Real};
                      corr::Tuple{<:Real,<:Real}, n_modes::Int=96)::Matrix{Float64}
    sx = collect(Float64, xs) ./ max(Float64(corr[1]), 1.0e-9)
    sz = collect(Float64, zs) ./ max(Float64(corr[2]), 1.0e-9)
    out = zeros(length(sx), length(sz))
    for _ in 1:n_modes
        kx, kz = randn(rng), randn(rng)
        φ = 2π * rand(rng)
        @inbounds for j in eachindex(sz), i in eachindex(sx)
            out[i, j] += cos(kx * sx[i] + kz * sz[j] + φ)
        end
    end
    σ = std(out)
    σ > 0 && (out ./= σ)
    out .-= mean(out)
    return out
end

struct _RoughPolygon
    x::Vector{Float64}
    z::Vector{Float64}
end

function _inside(poly::_RoughPolygon, x::Real, z::Real)::Bool
    px, pz = poly.x, poly.z
    n = length(px)
    inside = false
    j = n
    @inbounds for i in 1:n
        if (pz[i] > z) != (pz[j] > z)
            xint = (px[j] - px[i]) * (z - pz[i]) / (pz[j] - pz[i]) + px[i]
            x < xint && (inside = !inside)
        end
        j = i
    end
    return inside
end

function _random_polygon(rng::AbstractRNG; x0::Real, z0::Real,
                         half_w::Real, half_h::Real, n_vertices::Int,
                         roughness::Real=0.2)
    n = max(n_vertices, 3)
    θs = sort!([2π * (i - 1) / n + _uniform(rng, -0.3, 0.3) * 2π / n for i in 1:n])
    xs = Vector{Float64}(undef, n)
    zs = Vector{Float64}(undef, n)
    a = roughness * randn(rng)
    @inbounds for i in 1:n
        θ = θs[i]
        r = max(1 + a * cos(2θ), 0.35)
        xs[i] = x0 + r * half_w * cos(θ)
        zs[i] = z0 + r * half_h * sin(θ)
    end
    return _RoughPolygon(xs, zs)
end

function _sample_population(rng::AbstractRNG, pops::Vector{ResistivityPopulation})
    total = sum(p -> max(p.weight, 0.0), pops)
    total <= 0 && return pops[min(end, max(1, length(pops)))]
    t = rand(rng) * total
    acc = 0.0
    for p in pops
        acc += max(p.weight, 0.0)
        t <= acc && return p
    end
    return pops[end]
end

function _log10_clip(calib::Union{CalibrationStats,Nothing})::Tuple{Float64,Float64}
    calib === nothing && return _GENERIC_CLIP
    return calib.log10_clip
end

function _sample_log10(rng::AbstractRNG, calib::Union{CalibrationStats,Nothing},
                       kind::Symbol, clip::Tuple{Float64,Float64})::Float64
    lo_c, hi_c = clip
    if calib === nothing
        glo, ghi = _GENERIC_LOG10
        v = if kind === :conductive
            _uniform(rng, glo, glo + 0.35 * (ghi - glo))
        elseif kind === :resistive
            _uniform(rng, ghi - 0.35 * (ghi - glo), ghi)
        else
            _uniform(rng, glo, ghi)
        end
        return clamp(v, lo_c, hi_c)
    end

    idx = if kind === :conductive
        findfirst(p -> p.name == "conductive_body", calib.populations)
    elseif kind === :resistive
        findfirst(p -> p.name == "resistive_body", calib.populations)
    elseif kind === :host
        findfirst(p -> p.name == "host_rock", calib.populations)
    else
        nothing
    end
    p = idx === nothing ? _sample_population(rng, calib.populations) :
        calib.populations[idx]
    v = p.mean + p.std * randn(rng)
    for _ in 1:12
        p.lo <= v <= p.hi && break
        v = p.mean + p.std * randn(rng)
    end
    return clamp(v, max(p.lo, lo_c), min(p.hi, hi_c))
end

function _sample_thickness_m(rng::AbstractRNG, mp::MeshParams,
                             spec::LayerThicknessStats, fallback_cells::Tuple{Int,Int};
                             max_frac::Real=0.85)::Float64
    if spec.n > 0 && isfinite(spec.lognormal_mu) && isfinite(spec.lognormal_sigma)
        v = exp(spec.lognormal_mu + spec.lognormal_sigma * randn(rng))
        isfinite(spec.p05) && isfinite(spec.p95) && (v = clamp(v, spec.p05, spec.p95))
    else
        v = _uniform(rng, fallback_cells[1], fallback_cells[2]) * mp.dz
    end
    span = mp.nz * mp.dz
    return clamp(v, mp.dz, max_frac * span)
end

function _fill_layers!(canvas::Matrix{Float64}, xs::Vector{Float64}, zs::Vector{Float64},
                       values::Vector{Float64}, interfaces::Vector{Vector{Float64}},
                       transitions::Vector{Float64})
    nzc, nxc = size(canvas)
    nl = length(values)
    @inbounds for ix in 1:nxc
        for iz in 1:nzc
            z = zs[iz]
            v = values[nl]
            for k in 1:(nl - 1)
                zk = interfaces[k][ix]
                t = transitions[k]
                if z <= zk - t / 2
                    v = values[k]
                    break
                elseif t > 0 && z < zk + t / 2
                    u = (z - (zk - t / 2)) / t
                    w = u * u * (3 - 2u)
                    v = (1 - w) * values[k] + w * values[k + 1]
                    break
                end
            end
            canvas[iz, ix] = v
        end
    end
    return canvas
end

function _build_layer_stack(rng::AbstractRNG, mp::MeshParams, xs::Vector{Float64},
                            zs::Vector{Float64}, calib::Union{CalibrationStats,Nothing},
                            clip::Tuple{Float64,Float64}, diversity::Float64)
    n_layers = _randint(rng, (2, 6))
    p_cond = calib === nothing ? 0.2 : calib.conductive_fraction
    values = Vector{Float64}(undef, n_layers)
    @inbounds for k in 1:n_layers
        r = rand(rng)
        values[k] = if r < p_cond
            _sample_log10(rng, calib, :conductive, clip)
        elseif r < p_cond + 0.15
            _sample_log10(rng, calib, :resistive, clip)
        else
            _sample_log10(rng, calib, :host, clip)
        end
    end

    layer_spec = calib === nothing ?
        LayerThicknessStats(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN) :
        calib.layer_thickness_m
    corr_x = calib === nothing ? 4 * mp.dx : max(calib.corr_horizontal_m, 2 * mp.dx)
    rough_scale = clamp(0.15 + 0.35 * diversity, 0.05, 0.65)

    interfaces = Vector{Vector{Float64}}()
    transitions = Float64[]
    depth = zs[1]
    zbot = zs[end] + mp.dz
    for _ in 1:(n_layers - 1)
        thick = _sample_thickness_m(rng, mp, layer_spec, _GENERIC_LAYER_CELLS; max_frac=0.7)
        depth += thick
        depth >= zbot && (depth = zbot - 0.5 * mp.dz)
        field = random_field(rng, xs; corr=corr_x, n_modes=48)
        push!(interfaces, depth .+ rough_scale * thick .* field)
        push!(transitions, _uniform(rng, 0.0, 2.5 * mp.dz) * diversity)
    end
    return values, interfaces, transitions
end

function _paint_polygon!(canvas::Matrix{Float64}, xs::Vector{Float64}, zs::Vector{Float64},
                         poly::_RoughPolygon, value::Real)
    nzc, nxc = size(canvas)
    @inbounds for ix in 1:nxc, iz in 1:nzc
        _inside(poly, xs[ix], zs[iz]) && (canvas[iz, ix] = value)
    end
    return canvas
end

function _add_polygon_body!(canvas::Matrix{Float64}, rng::AbstractRNG,
                            mp::MeshParams, xs::Vector{Float64}, zs::Vector{Float64},
                            calib::Union{CalibrationStats,Nothing},
                            clip::Tuple{Float64,Float64}, diversity::Float64)
    kind = rand(rng) < 0.65 ? :conductive : :resistive
    value = _sample_log10(rng, calib, kind, clip)

    prof_span = mp.nx * mp.dx
    depth_span = mp.nz * mp.dz
    w_spec = calib === nothing ?
        LayerThicknessStats(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN) :
        calib.body_width_m
    h_spec = calib === nothing ?
        LayerThicknessStats(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN) :
        calib.body_thickness_m

    half_w = 0.5 * _sample_thickness_m(rng, mp, w_spec, (3, 12); max_frac=0.75)
    half_w = clamp(half_w, 2 * mp.dx, 0.45 * prof_span)
    half_h = 0.5 * _sample_thickness_m(rng, mp, h_spec, (2, 10); max_frac=0.6)
    half_h = clamp(half_h, 2 * mp.dz, 0.45 * depth_span)

    x0 = _uniform(rng, xs[1] + half_w, xs[end] - half_w)
    z0 = _uniform(rng, zs[1] + half_h, zs[end] - half_h)
    n_vert = _randint(rng, (3, 8))
    poly = _random_polygon(rng; x0=x0, z0=z0, half_w=half_w, half_h=half_h,
                           n_vertices=n_vert,
                           roughness=_uniform(rng, 0.05, 0.35) * diversity)
    _paint_polygon!(canvas, xs, zs, poly, value)
    return canvas
end

function _apply_texture!(canvas::Matrix{Float64}, rng::AbstractRNG,
                         mp::MeshParams, xs::Vector{Float64}, zs::Vector{Float64},
                         calib::Union{CalibrationStats,Nothing}, diversity::Float64)
    amp = _uniform(rng, 0.02, 0.18) * diversity
    amp <= 0 && return canvas
    cz = calib === nothing ? 3 * mp.dz : max(calib.corr_vertical_m, 2 * mp.dz)
    cx = calib === nothing ? 4 * mp.dx : max(calib.corr_horizontal_m, 2 * mp.dx)
    field = random_field(rng, xs, zs; corr=(cx, cz))
    nzc, nxc = size(canvas)
    @inbounds for ix in 1:nxc, iz in 1:nzc
        canvas[iz, ix] += amp * field[ix, iz]
    end
    return canvas
end

"""
    generate_synthetic_model(mp; calib=nothing, diversity_level=1.0, rng, seed)
        -> Matrix{Float32}

Generate one `(nz, nx)` log10-resistivity model on the uniform `MeshParams` grid.

- Layered backgrounds with Gaussian-rough interfaces
- Random polygon anomaly bodies (3–8 vertices)
- With `calib`: resistivity and sizes drawn from Keivitsa log-normal priors
- With `calib=nothing`: broad generic geology (anti field over-specialisation)
"""
function generate_synthetic_model(mp::MeshParams;
                                  calib::Union{CalibrationStats,Nothing}=nothing,
                                  diversity_level::Float64=1.0,
                                  rng::AbstractRNG=Random.default_rng(),
                                  seed::Union{Nothing,Integer}=nothing)::Matrix{Float32}
    validate_mesh_params(mp)
    diversity = clamp(Float64(diversity_level), 0.1, 3.0)
    mrng = seed === nothing ? rng : MersenneTwister(Int(seed))

    xs, zs = _grid_coords(mp)
    clip = _log10_clip(calib)
    canvas = zeros(Float64, mp.nz, mp.nx)

    scenario = rand(mrng)
    if scenario < 0.12
        fill!(canvas, _sample_log10(mrng, calib, :any, clip))
    else
        values, ifaces, trans = _build_layer_stack(mrng, mp, xs, zs, calib, clip, diversity)
        _fill_layers!(canvas, xs, zs, values, ifaces, trans)
        n_bodies = if scenario < 0.45
            0
        elseif scenario < 0.75
            1
        else
            _randint(mrng, (2, min(4, 2 + round(Int, diversity))))
        end
        for _ in 1:n_bodies
            _add_polygon_body!(canvas, mrng, mp, xs, zs, calib, clip, diversity)
        end
    end

    _apply_texture!(canvas, mrng, mp, xs, zs, calib, diversity)
    lo, hi = clip
    clamp!(canvas, lo, hi)
    return Float32.(canvas)
end

# ─────────────────────────────────────────────────────────────────────────────
# Batch I/O
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_dataset(n_models, output_dir, mp; kwargs...) -> Vector{String}

Write `n_models` synthetic models to `output_dir/model_<id>.jld2`. Each file
stores `model` (`Matrix{Float32}`) and `mp` (`MeshParams`).
"""
function generate_dataset(n_models::Integer, output_dir::AbstractString, mp::MeshParams;
                          calib::Union{CalibrationStats,Nothing}=nothing,
                          diversity_level::Float64=1.0,
                          seed::Union{Nothing,Integer}=nothing,
                          log_every::Integer=1)::Vector{String}
    n_models > 0 || error("n_models must be positive")
    out_dir = abspath(output_dir)
    mkpath(out_dir)
    base_seed = seed === nothing ? rand(MersenneTwister(), 1:typemax(Int32)) : Int(seed)
    rng = MersenneTwister(base_seed)
    paths = String[]

    @info "Generating resistivity models" n=n_models grid=(mp.nz, mp.nx) out=out_dir
    for id in 1:n_models
        model = generate_synthetic_model(mp; calib=calib, diversity_level=diversity_level,
                                         rng=rng)
        path = joinpath(out_dir, @sprintf("model_%04d.jld2", id))
        jldsave(path; model=model, mp=mp, schema="resistivity_model/v1", index=id,
                base_seed=base_seed)
        push!(paths, path)
        log_every > 0 && (id % log_every == 0 || id == n_models) &&
            @info "  progress" done=id of=n_models
    end
    return paths
end

"""
    load_synthetic_model(path) -> (model, mp)

Load one model file written by [`generate_dataset`](@ref).
"""
function load_synthetic_model(path::AbstractString)
    data = load(abspath(path))
    haskey(data, "model") || error("missing `model` in $(path)")
    haskey(data, "mp") || error("missing `mp` in $(path)")
    model = data["model"]
    mp = data["mp"]
    model isa Matrix{Float32} || (model = Matrix{Float32}(model))
    mp isa MeshParams || error("`mp` is not MeshParams in $(path)")
    return model, mp
end

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

"""Geophysical resistivity colormap: blue (conductive) → red (resistive)."""
function _colormap_resistivity(t::Float64)
    t = clamp(t, 0.0, 1.0)
    if t < 0.25
        s = t / 0.25
        return (UInt8(round(20 + 60s)), UInt8(round(40 + 80s)), UInt8(200))
    elseif t < 0.5
        s = (t - 0.25) / 0.25
        return (UInt8(round(80 + 100s)), UInt8(round(120 + 80s)), UInt8(round(200 - 120s)))
    elseif t < 0.75
        s = (t - 0.5) / 0.25
        return (UInt8(round(180 + 40s)), UInt8(round(200 - 80s)), UInt8(round(80 - 60s)))
    else
        s = (t - 0.75) / 0.25
        return (UInt8(round(220 + 35s)), UInt8(round(120 - 80s)), UInt8(round(20 + 20s)))
    end
end

function _write_pgm(path::AbstractString, rgb::Matrix{NTuple{3,UInt8}})
    h, w = size(rgb)
    open(path, "w") do io
        println(io, "P6")
        println(io, w, " ", h)
        println(io, "255")
        buf = Vector{UInt8}(undef, 3 * w * h)
        k = 1
        @inbounds for j in 1:h, i in 1:w
            r, g, b = rgb[j, i]
            buf[k] = r; buf[k+1] = g; buf[k+2] = b
            k += 3
        end
        write(io, buf)
    end
    return path
end

"""
    plot_resistivity_model(model, mp; path, log10_range=nothing, title=nothing)
        -> String

Render a depth-first resistivity heatmap. Writes PGM by default; if `path` ends
in `.png` and Plots.jl is available, saves a PNG heatmap instead.
"""
function plot_resistivity_model(model::AbstractMatrix{<:Real}, mp::MeshParams;
                                path::AbstractString="resistivity_model.pgm",
                                log10_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                                title::Union{Nothing,AbstractString}=nothing)::String
    out = abspath(path)
    mkpath(dirname(out))
    lo = log10_range === nothing ? minimum(model) : Float64(log10_range[1])
    hi = log10_range === nothing ? maximum(model) : Float64(log10_range[2])
    hi <= lo && (hi = lo + 1.0)

    ext = lowercase(splitext(out)[2])
    if ext == ".png"
        try
            @eval import Plots
            xs = [(i - 0.5) * mp.dx / 1000 for i in 1:mp.nx]
            zs = [(k - 0.5) * mp.dz / 1000 for k in 1:mp.nz]
            ttl = title === nothing ? "log10 resistivity (Ω·m)" : String(title)
            p = Plots.heatmap(xs, zs, model; color=:turbo, clims=(lo, hi),
                              xlabel="profile (km)", ylabel="depth (km)",
                              yflip=true, title=ttl, size=(900, 500))
            Plots.savefig(p, out)
            return out
        catch err
            @warn "Plots.jl unavailable; writing PGM instead" exception=err
            out = replace(out, r"\.png$" => ".pgm")
        end
    end

    endswith(out, ".pgm") || (out *= ".pgm")
    nz, nx = size(model)
    rgb = Matrix{NTuple{3,UInt8}}(undef, nz, nx)
    @inbounds for iz in 1:nz, ix in 1:nx
        t = (Float64(model[iz, ix]) - lo) / (hi - lo)
        rgb[iz, ix] = _colormap_resistivity(t)
    end
    return _write_pgm(out, rgb)
end

end # module ResistivityModelGenerator
