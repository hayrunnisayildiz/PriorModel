"""
    SyntheticGenerator

Synthetic 2-D resistivity model generator for MT prior-network training, calibrated
by `config/keivitsa_priors.json` (see `src/analysis/keivitsa_stats.jl`) and built on
a mesh that MTGeophysics.jl's 2-D TE/TM forward solver accepts unchanged.

# Mesh convention (MTGeophysics.jl v0.4.2)
The solver expects `resistivity[iz, iy]` — **depth first, profile second** — sized
`(n_z, n_y)` in linear Ω·m, with `n_air_cells` air rows of `1e9` Ω·m on top, `z`
positive downward and `z = 0` at the surface. Cells are *non-uniform*: lateral
padding grows geometrically outward and ground layers grow with depth.

That full solver array is not a good CNN tensor, so a model is stored as its
**target zone** only: the uniform-cell rectangle `(n_target_z, n_core_y)` of
`log10(ρ)` that the network predicts. [`solver_resistivity`](@ref) expands it back
to the full `(n_z, n_y)` Ω·m array (air, lateral padding, deep zone) for forward
modelling, and [`to_mt2d_mesh`](@ref) hands over a genuine `MT2DMesh`.

# Algorithm
Following the synthetic-training-set recipe of Liu et al. (2023)-type MT deep
learning work:
- layered backgrounds whose interfaces are roughened by Gaussian / fractal random
  fields, with finite-width transition zones;
- random rough polygons for discrete bodies, sized and valued from the prior
  distributions;
- a scenario library (layered, buried block, dipping fault, thrust sheet, basin,
  dyke, multi-anomaly, half-space) so the network sees generic geology rather than
  only Keivitsa;
- anisotropic textural noise at the prior correlation lengths.

# Prior scaling
Keivitsa's resistivity statistics are transferred as-is — they are the physically
meaningful calibration. Its *lengths* come from core logs (median layer ≈ 6 m) and
cannot be represented on, or resolved by, an MT mesh. Only the **shape** of those
length distributions transfers, so all prior lengths are multiplied by
`length_scale`, chosen by default so the median layer spans a few mesh cells. The
factor is recorded in every model's metadata.

# Units
`ρ` Ω·m (stored as `log10(ρ)`), lengths m, angles deg, frequency Hz.

# Example
```julia
include("src/synthetic/synthetic_generator.jl")
using .SyntheticGenerator

cfg    = GeneratorConfig()
mesh   = build_generator_mesh(cfg)
priors = load_generator_priors("config/keivitsa_priors.json")
model  = generate_model(mesh, priors, cfg; rng = MersenneTwister(1))

ρ = solver_resistivity(model, mesh)      # (n_z, n_y) Ω·m, ready for the solver
generate_dataset("data/processed/synthetic_mt2d.h5"; n_models = 2000)
```
"""
module SyntheticGenerator

using Dates
using HDF5
using JLD2
using JSON3
using Printf
using Random
using Statistics

export GeneratorConfig, GeneratorMesh, SyntheticModel, PriorSpec
export build_generator_mesh, load_generator_priors
export generate_model, generate_dataset, save_dataset, load_dataset
export solver_resistivity, to_mt2d_mesh, verify_mesh, forward_response
export SCENARIOS, n_y, n_z, y_centers, z_centers, target_size

const SCHEMA_VERSION = "synthetic_mt2d/v1"

"""Air-cell resistivity used by MTGeophysics.jl (`VFSA2DMT.air_resistivity`)."""
const AIR_RESISTIVITY::Float64 = 1.0e9

"""MTGeophysics.jl package UUID, for the lazy import in [`mtgeophysics`](@ref)."""
const MTGEOPHYSICS_UUID = Base.UUID("71b12eb5-90c3-4985-b1e7-1ec075583b1e")

"""Geological scenarios, ordered from simplest to most structurally complex."""
const SCENARIOS = (:halfspace, :layered, :layered_block, :multi_anomaly,
                   :basin, :dyke, :dipping_fault, :thrust_sheet)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

"""
    GeneratorConfig

Mesh geometry and sampling controls. Every field is echoed into the dataset file,
so a synthetic set documents its own recipe.

# Mesh fields
- `frequencies`: forward-modelling frequencies, Hz (audio-MT band by default)
- `y_core_range`, `y_core_cell`: uniform core of the profile, m
- `y_padding`, `pad_factor`: lateral padding extent and growth ratio
- `air_top`, `air_cells`: air column top (negative, m) and row count
- `target_dz`, `n_target`: uniform target-zone cell height and row count, m
- `max_depth`, `depth_growth`: deep-zone extent and layer growth ratio, m
- `receiver_stride`: keep every n-th core cell centre as a station

# Sampling fields
- `scenario_weights`: relative probability per entry of [`SCENARIOS`](@ref)
- `generic_fraction`: share of models drawing resistivity from a broad
  log-uniform range instead of the Keivitsa mixture (anti-overfitting)
- `generic_log10_range`: that broad range, log10(Ω·m)
- `log10_clip`: hard bounds on the output, log10(Ω·m); the default matches
  `VFSA2DMTConfig.log_bounds`
- `length_scale`: multiplier on all prior lengths; `nothing` derives it from
  `median_layer_cells`
- `median_layer_cells`: target median layer thickness, in target-zone cells
- `min_feature_cells`: smallest admissible feature, in cells
- `n_layers_range`: layer count per layered background
- `texture_log10_range`: amplitude of the final textural noise, log10 decades
- `texture_kind`: `:fractal`, `:gaussian`
- `hurst`: Hurst exponent of the fractal spectrum
- `interface_roughness_range`: interface undulation amplitude, as a fraction of
  the layer thickness
- `transition_cells_range`: interface transition-zone width, in cells
- `n_anomalies_range`: body count for `:multi_anomaly`
- `dip_range`, `thrust_dip_range`: dip from horizontal, deg
- `seed`: base RNG seed
"""
Base.@kwdef struct GeneratorConfig
    # mesh
    frequencies::Vector{Float64} = collect(10 .^ range(1, 4, length = 20))
    y_core_range::Tuple{Float64,Float64} = (-1500.0, 1500.0)
    y_core_cell::Float64 = 25.0
    y_padding::Float64 = 100_000.0
    pad_factor::Float64 = 1.4
    air_top::Float64 = -20_000.0
    air_cells::Int = 10
    target_dz::Float64 = 25.0
    n_target::Int = 48
    max_depth::Float64 = 40_000.0
    depth_growth::Float64 = 1.35
    receiver_stride::Int = 4
    # sampling
    scenario_weights::Vector{Float64} = [0.05, 0.15, 0.15, 0.15, 0.12, 0.12, 0.13, 0.13]
    generic_fraction::Float64 = 0.35
    generic_log10_range::Tuple{Float64,Float64} = (0.3, 4.7)
    log10_clip::Tuple{Float64,Float64} = (0.0, 5.0)
    length_scale::Union{Nothing,Float64} = nothing
    median_layer_cells::Float64 = 3.0
    min_feature_cells::Float64 = 1.5
    n_layers_range::Tuple{Int,Int} = (2, 7)
    texture_log10_range::Tuple{Float64,Float64} = (0.02, 0.25)
    texture_kind::Symbol = :fractal
    hurst::Float64 = 0.75
    interface_roughness_range::Tuple{Float64,Float64} = (0.0, 0.45)
    transition_cells_range::Tuple{Float64,Float64} = (0.0, 3.0)
    n_anomalies_range::Tuple{Int,Int} = (2, 5)
    dip_range::Tuple{Float64,Float64} = (35.0, 90.0)
    thrust_dip_range::Tuple{Float64,Float64} = (8.0, 35.0)
    seed::Int = 20260318
end

# ─────────────────────────────────────────────────────────────────────────────
# Mesh
# ─────────────────────────────────────────────────────────────────────────────

"""
    GeneratorMesh

A 2-D MT mesh in MTGeophysics.jl's own parameterisation, plus the index ranges
that mark the machine-learning window.

# Fields
- `y_nodes`, `z_nodes`: cell edges, m (`z` positive down, `z = 0` at surface)
- `y_cell_sizes`, `z_cell_sizes`: cell widths, m
- `receiver_positions`: station offsets along the profile, m
- `frequencies`: Hz
- `n_air_cells`: air rows at the top of the `z` axis
- `core_y`: column range of the uniform profile core
- `target_z`: row range of the uniform target zone (below the air rows)
- `mesh_kwargs`: the exact `BuildMesh2D` keyword arguments that reproduce this mesh
"""
struct GeneratorMesh
    y_nodes::Vector{Float64}
    z_nodes::Vector{Float64}
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    receiver_positions::Vector{Float64}
    frequencies::Vector{Float64}
    n_air_cells::Int
    core_y::UnitRange{Int}
    target_z::UnitRange{Int}
    mesh_kwargs::Dict{String,Any}
end

n_y(m::GeneratorMesh)::Int = length(m.y_cell_sizes)
n_z(m::GeneratorMesh)::Int = length(m.z_cell_sizes)

"""
    y_centers(mesh) / z_centers(mesh) -> Vector{Float64}

Cell-centre coordinates, matching `MTGeophysics.mt2d_y_centers` / `mt2d_z_centers`.
"""
y_centers(m::GeneratorMesh) = 0.5 .* (m.y_nodes[1:end-1] .+ m.y_nodes[2:end])
z_centers(m::GeneratorMesh) = 0.5 .* (m.z_nodes[1:end-1] .+ m.z_nodes[2:end])

"""
    target_size(mesh) -> (n_target_z, n_core_y)

Shape of the stored `log10(ρ)` tensor for one model.
"""
target_size(m::GeneratorMesh) = (length(m.target_z), length(m.core_y))

function Base.show(io::IO, m::GeneratorMesh)
    nt, nc = target_size(m)
    print(io, "GeneratorMesh(solver=(", n_z(m), ",", n_y(m), ") air=", m.n_air_cells,
          " target=(", nt, ",", nc, ")",
          " y=[", m.y_nodes[1], ",", m.y_nodes[end], "]",
          " z=[", m.z_nodes[1], ",", m.z_nodes[end], "]",
          " stations=", length(m.receiver_positions),
          " freqs=", length(m.frequencies), ")")
end

"""
    ground_layer_sizes(cfg) -> Vector{Float64}

Ground layer thicknesses: `n_target` uniform cells of `target_dz` (the target
zone) followed by geometrically growing cells down to `max_depth`, the usual MT
mesh construction. The deep zone exists so the lowest frequencies see a
half-space; it is filled by continuation, not by generated structure.
"""
function ground_layer_sizes(cfg::GeneratorConfig)::Vector{Float64}
    cfg.target_dz > 0 || error("target_dz must be positive")
    cfg.n_target > 0 || error("n_target must be positive")
    cfg.depth_growth > 1 || error("depth_growth must be greater than 1")
    layers = fill(Float64(cfg.target_dz), cfg.n_target)
    depth = cfg.target_dz * cfg.n_target
    depth < cfg.max_depth ||
        error("max_depth ($(cfg.max_depth) m) must exceed the target zone ($(depth) m)")
    dz = Float64(cfg.target_dz)
    while depth < cfg.max_depth
        dz *= cfg.depth_growth
        push!(layers, dz)
        depth += dz
    end
    return layers
end

"""
    build_generator_mesh(cfg=GeneratorConfig()) -> GeneratorMesh

Construct the mesh with MTGeophysics.jl's own node algorithm (`build_mt2d_mesh`,
v0.4.2): a uniform core flanked by geometrically expanding padding, an air column
of `air_cells` equal cells from `air_top` to `0`, and the ground layers of
[`ground_layer_sizes`](@ref).

[`verify_mesh`](@ref) asserts the result is identical to calling `BuildMesh2D`
with `mesh_kwargs`, so the geometry is guaranteed to match the solver rather than
merely resemble it.
"""
function build_generator_mesh(cfg::GeneratorConfig = GeneratorConfig())::GeneratorMesh
    y1, y2 = Float64.(cfg.y_core_range)
    y2 > y1 || error("y_core_range must be increasing")
    cfg.y_core_cell > 0 || error("y_core_cell must be positive")
    cfg.y_padding > 0 || error("y_padding must be positive")
    cfg.pad_factor > 1 || error("pad_factor must be greater than 1")
    cfg.air_cells > 0 || error("air_cells must be positive")
    cfg.air_top < 0 || error("air_top must be negative (above the surface)")
    cfg.receiver_stride > 0 || error("receiver_stride must be positive")

    y_core_nodes = collect(y1:cfg.y_core_cell:y2)
    abs(y_core_nodes[end] - y2) > 1e-9 && push!(y_core_nodes, y2)

    left_nodes = Float64[]
    Δy = Float64(cfg.y_core_cell)
    y = y1
    while y - Δy > y1 - cfg.y_padding - 1e-9
        Δy *= cfg.pad_factor
        y -= Δy
        push!(left_nodes, y)
    end
    reverse!(left_nodes)

    right_nodes = Float64[]
    Δy = Float64(cfg.y_core_cell)
    y = y2
    while y + Δy < y2 + cfg.y_padding + 1e-9
        Δy *= cfg.pad_factor
        y += Δy
        push!(right_nodes, y)
    end

    y_nodes = vcat(left_nodes, y_core_nodes, right_nodes)
    y_cell_sizes = diff(y_nodes)

    ground_layers = ground_layer_sizes(cfg)
    z_air = collect(range(Float64(cfg.air_top), 0.0, length = cfg.air_cells + 1))
    z_ground = vcat(0.0, cumsum(ground_layers))
    z_nodes = vcat(z_air[1:end-1], z_ground)
    z_cell_sizes = diff(z_nodes)

    y_recv = 0.5 .* (y_core_nodes[1:end-1] .+ y_core_nodes[2:end])
    receivers = collect(y_recv[1:cfg.receiver_stride:end])

    core_y = (length(left_nodes)+1):(length(left_nodes)+length(y_core_nodes)-1)
    target_z = (cfg.air_cells+1):(cfg.air_cells+cfg.n_target)

    kwargs = Dict{String,Any}(
        "frequencies" => copy(cfg.frequencies),
        "y_core_range" => [y1, y2],
        "y_core_cell" => cfg.y_core_cell,
        "y_padding" => cfg.y_padding,
        "pad_factor" => cfg.pad_factor,
        "air_top" => cfg.air_top,
        "air_cells" => cfg.air_cells,
        "ground_layers" => ground_layers,
        "receiver_stride" => cfg.receiver_stride,
    )

    return GeneratorMesh(y_nodes, z_nodes, y_cell_sizes, z_cell_sizes,
                         Float64.(receivers), Float64.(cfg.frequencies),
                         cfg.air_cells, core_y, target_z, kwargs)
end

"""
    mtgeophysics() -> Module

Lazily load MTGeophysics.jl. It is deliberately not a hard dependency: model
generation and storage need none of it, only forward modelling does.
"""
function mtgeophysics()::Module
    return Base.require(Base.PkgId(MTGEOPHYSICS_UUID, "MTGeophysics"))
end

"""
    _mtg_call(f, args...; kwargs...)

Invoke a function from the lazily loaded MTGeophysics.jl. `Base.invokelatest` is
required because the package enters the session after this module was compiled,
so its methods live in a newer world age.
"""
_mtg_call(f, args...; kwargs...) = Base.invokelatest(f, args...; kwargs...)

"""
    to_mt2d_mesh(mesh) -> MTGeophysics.MT2DMesh

Wrap a [`GeneratorMesh`](@ref) as the solver's own mesh type, field for field.
"""
function to_mt2d_mesh(mesh::GeneratorMesh)
    MTG = mtgeophysics()
    return _mtg_call(MTG.MT2DMesh;
        y_nodes = copy(mesh.y_nodes),
        z_nodes = copy(mesh.z_nodes),
        y_cell_sizes = copy(mesh.y_cell_sizes),
        z_cell_sizes = copy(mesh.z_cell_sizes),
        receiver_positions = copy(mesh.receiver_positions),
        frequencies = copy(mesh.frequencies),
        n_air_cells = mesh.n_air_cells,
    )
end

"""
    verify_mesh(mesh; atol=0.0) -> NamedTuple

Rebuild the mesh through `MTGeophysics.BuildMesh2D(; mesh_kwargs...)` and compare
every geometric vector. Returns `(ok, max_y_error, max_z_error, ...)`; `ok` is
true only when the node vectors agree to `atol` (exact by default).
"""
function verify_mesh(mesh::GeneratorMesh; atol::Real = 0.0)
    MTG = mtgeophysics()
    kw = mesh.mesh_kwargs
    reference = _mtg_call(MTG.BuildMesh2D;
        frequencies = kw["frequencies"],
        y_core_range = (kw["y_core_range"][1], kw["y_core_range"][2]),
        y_core_cell = kw["y_core_cell"],
        y_padding = kw["y_padding"],
        pad_factor = kw["pad_factor"],
        air_top = kw["air_top"],
        air_cells = kw["air_cells"],
        ground_layers = kw["ground_layers"],
        receiver_stride = kw["receiver_stride"],
    )
    dy = length(reference.y_nodes) == length(mesh.y_nodes) ?
         maximum(abs.(reference.y_nodes .- mesh.y_nodes)) : Inf
    dz = length(reference.z_nodes) == length(mesh.z_nodes) ?
         maximum(abs.(reference.z_nodes .- mesh.z_nodes)) : Inf
    dr = length(reference.receiver_positions) == length(mesh.receiver_positions) ?
         maximum(abs.(reference.receiver_positions .- mesh.receiver_positions)) : Inf
    ok = dy <= atol && dz <= atol && dr <= atol &&
         reference.n_air_cells == mesh.n_air_cells
    return (ok = ok, max_y_error = dy, max_z_error = dz, max_receiver_error = dr,
            n_y = length(reference.y_cell_sizes), n_z = length(reference.z_cell_sizes))
end

# ─────────────────────────────────────────────────────────────────────────────
# Priors
# ─────────────────────────────────────────────────────────────────────────────

"""
    PopulationSpec

One resistivity population from the prior mixture, in `log10(Ω·m)`.
"""
struct PopulationSpec
    name::String
    mean::Float64
    std::Float64
    weight::Float64
    lo::Float64
    hi::Float64
end

"""
    SizeSpec

A lognormal length prior: `exp(mu + sigma·randn())` metres, with the empirical
p05/p95 as soft bounds.
"""
struct SizeSpec
    name::String
    mu::Float64
    sigma::Float64
    lo::Float64
    hi::Float64
end

"""
    PriorSpec

Everything the generator draws from, distilled from `keivitsa_priors.json`.

# Fields
- `populations`: mixture components, conductive → resistive
- `host`, `conductive`, `resistive`: named shortcuts into `populations`
- `layer_thickness`, `body_thickness`, `body_width`: length priors, m
- `aspect_ratio`: body vertical/horizontal extent ratio (dimensionless)
- `conductive_fraction`: share of logged metres below the conductor cutoff
- `corr_vertical`, `corr_horizontal`: variogram ranges, m
- `anisotropy`: `corr_horizontal / corr_vertical`
- `log10_clip`: prior's own resistivity envelope, log10(Ω·m)
- `source`: path of the JSON the spec came from
"""
struct PriorSpec
    populations::Vector{PopulationSpec}
    host::PopulationSpec
    conductive::PopulationSpec
    resistive::PopulationSpec
    layer_thickness::SizeSpec
    body_thickness::SizeSpec
    body_width::SizeSpec
    aspect_ratio::Float64
    conductive_fraction::Float64
    corr_vertical::Float64
    corr_horizontal::Float64
    anisotropy::Float64
    log10_clip::Tuple{Float64,Float64}
    source::String
end

_num(x, fallback::Float64)::Float64 =
    (x === nothing || x isa AbstractString) ? fallback :
    (x isa Real && isfinite(Float64(x)) ? Float64(x) : fallback)

function _dig(node, path::Vararg{String})
    cur::Any = node
    for k in path
        cur isa AbstractDict || return nothing
        haskey(cur, k) || return nothing
        cur = cur[k]
    end
    return cur
end

function _population(node, name::String, fallback::NTuple{4,Float64})::PopulationSpec
    μ = _num(_dig(node, "log10_mean"), fallback[1])
    σ = max(_num(_dig(node, "log10_std"), fallback[2]), 1.0e-3)
    w = _num(_dig(node, "weight"), fallback[3])
    rng_ = _dig(node, "log10_range")
    lo, hi = if rng_ isa AbstractVector && length(rng_) == 2
        (_num(rng_[1], μ - 2σ), _num(rng_[2], μ + 2σ))
    else
        (μ - 2σ, μ + 2σ)
    end
    hi <= lo && (hi = lo + fallback[4])
    return PopulationSpec(name, μ, σ, w, lo, hi)
end

function _size_spec(node, name::String, fallback::NTuple{4,Float64})::SizeSpec
    μ = _num(_dig(node, "lognormal_mu"), fallback[1])
    σ = max(_num(_dig(node, "lognormal_sigma"), fallback[2]), 1.0e-3)
    lo = _num(_dig(node, "p05"), fallback[3])
    hi = _num(_dig(node, "p95"), fallback[4])
    hi <= lo && (hi = lo * 10)
    return SizeSpec(name, μ, σ, lo, hi)
end

"""
    load_generator_priors(path="config/keivitsa_priors.json") -> PriorSpec

Read the `generator_priors` block written by `KeivitsaStats`. Missing or `null`
entries fall back to generic mid-crustal values, so an incomplete prior file
degrades to a sane generic generator instead of failing.
"""
function load_generator_priors(path::AbstractString = "")::PriorSpec
    p = isempty(path) ? joinpath(_project_root(), "config", "keivitsa_priors.json") :
        abspath(path)
    isfile(p) || error("Prior file not found: $(p) — run scripts/build_keivitsa_priors.jl")
    doc = JSON3.read(read(p, String), Dict{String,Any})
    g = get(doc, "generator_priors", Dict{String,Any}())

    conductive = _population(get(g, "conductive_body", nothing), "conductive_body",
                             (1.0, 0.6, 0.25, 1.5))
    host = _population(get(g, "host_rock", nothing), "host_rock",
                       (3.0, 0.8, 0.5, 1.5))
    resistive = _population(get(g, "resistive_body", nothing), "resistive_body",
                            (4.4, 0.3, 0.25, 1.0))

    anom = get(g, "anomaly", Dict{String,Any}())
    layer = _size_spec(get(g, "layer_thickness_m", nothing), "layer_thickness",
                       (1.9, 0.85, 2.0, 34.0))
    body_t = _size_spec(get(anom, "vertical_thickness_m", nothing), "body_thickness",
                        (1.8, 0.8, 2.2, 26.0))
    body_w = _size_spec(get(anom, "across_strike_width_m", nothing), "body_width",
                        (3.3, 0.93, 5.0, 143.0))
    aspect = _num(get(anom, "thickness_to_width_ratio", nothing), 0.3)
    cond_frac = clamp(_num(get(anom, "volume_fraction_conductive", nothing), 0.15),
                      0.01, 0.6)

    corr = get(g, "correlation_length_m", Dict{String,Any}())
    cz = _num(get(corr, "vertical_variogram_range", nothing), 20.0)
    cy = _num(get(corr, "horizontal_variogram_range", nothing), 150.0)

    clip = get(g, "clip_ohm_m", nothing)
    log_clip = if clip isa AbstractVector && length(clip) == 2
        (log10(max(_num(clip[1], 1.0), 1.0e-6)), log10(max(_num(clip[2], 1.0e5), 10.0)))
    else
        (0.0, 5.0)
    end

    return PriorSpec([conductive, host, resistive], host, conductive, resistive,
                     layer, body_t, body_w, aspect, cond_frac, cz, cy,
                     cz > 0 ? cy / cz : 1.0, log_clip, p)
end

function _project_root()::String
    d = @__DIR__
    while true
        isdir(joinpath(d, "config")) && isdir(joinpath(d, "src")) && return d
        parent = dirname(d)
        parent == d && return @__DIR__
        d = parent
    end
end

"""
    resolve_length_scale(cfg, priors) -> Float64

Factor mapping core-scale prior lengths onto the mesh. When
`cfg.length_scale === nothing` it is set so the prior median layer thickness
covers `cfg.median_layer_cells` target cells; scaling a lognormal shifts `mu`
only, so the distribution's shape (and hence every ratio) is preserved.
"""
function resolve_length_scale(cfg::GeneratorConfig, priors::PriorSpec)::Float64
    cfg.length_scale !== nothing && return Float64(cfg.length_scale)
    median_layer = exp(priors.layer_thickness.mu)
    median_layer > 0 || return 1.0
    return max(cfg.median_layer_cells * cfg.target_dz / median_layer, 1.0e-6)
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling primitives
# ─────────────────────────────────────────────────────────────────────────────

"""
    SampleContext

Per-model scratch state: the RNG, the geometry of the target window, the active
prior mode and the metadata being accumulated.
"""
struct SampleContext
    rng::AbstractRNG
    cfg::GeneratorConfig
    priors::PriorSpec
    ys::Vector{Float64}          # target-window cell centres, m
    zs::Vector{Float64}
    dy::Float64                  # uniform core cell size, m
    dz::Float64
    length_scale::Float64
    mode::Symbol                 # :keivitsa | :generic
    meta::Dict{String,Any}
end

_uniform(rng::AbstractRNG, lo::Real, hi::Real)::Float64 =
    hi <= lo ? Float64(lo) : Float64(lo) + rand(rng) * (Float64(hi) - Float64(lo))

_uniform(rng::AbstractRNG, r::Tuple{<:Real,<:Real})::Float64 = _uniform(rng, r[1], r[2])

_randint(rng::AbstractRNG, r::Tuple{Int,Int})::Int =
    r[2] <= r[1] ? r[1] : rand(rng, r[1]:r[2])

"""
    sample_log10_rho(ctx, kind) -> Float64

Draw a resistivity in `log10(Ω·m)`.

In `:keivitsa` mode the value comes from the requested mixture population
(`:host`, `:conductive`, `:resistive`, or `:any` to sample by mixture weight),
truncated to that population's empirical range. In `:generic` mode it is drawn
log-uniformly from `cfg.generic_log10_range`, biased low or high for
`:conductive` / `:resistive`, so the network is not tuned to Keivitsa's modes.
"""
function sample_log10_rho(ctx::SampleContext, kind::Symbol)::Float64
    lo_c, hi_c = ctx.cfg.log10_clip
    if ctx.mode === :generic
        glo, ghi = ctx.cfg.generic_log10_range
        v = if kind === :conductive
            _uniform(ctx.rng, glo, glo + 0.35 * (ghi - glo))
        elseif kind === :resistive
            _uniform(ctx.rng, ghi - 0.35 * (ghi - glo), ghi)
        else
            _uniform(ctx.rng, glo, ghi)
        end
        return clamp(v, lo_c, hi_c)
    end

    pop = if kind === :host
        ctx.priors.host
    elseif kind === :conductive
        ctx.priors.conductive
    elseif kind === :resistive
        ctx.priors.resistive
    else
        _weighted_pick(ctx.rng, ctx.priors.populations)
    end
    v = pop.mean + pop.std * randn(ctx.rng)
    for _ in 1:16
        pop.lo <= v <= pop.hi && break
        v = pop.mean + pop.std * randn(ctx.rng)
    end
    return clamp(v, max(pop.lo, lo_c), min(pop.hi, hi_c))
end

function _weighted_pick(rng::AbstractRNG, pops::Vector{PopulationSpec})::PopulationSpec
    total = sum(p -> max(p.weight, 0.0), pops)
    total <= 0 && return pops[min(2, length(pops))]
    t = rand(rng) * total
    acc = 0.0
    for p in pops
        acc += max(p.weight, 0.0)
        t <= acc && return p
    end
    return pops[end]
end

"""
    depth_span(ctx) / profile_span(ctx) -> Float64

Extent of the target window along `z` and along `y`, in metres.
"""
depth_span(ctx::SampleContext)::Float64 = ctx.zs[end] - ctx.zs[1] + ctx.dz
profile_span(ctx::SampleContext)::Float64 = ctx.ys[end] - ctx.ys[1] + ctx.dy

"""
    sample_length(ctx, spec; min_cells, max_frac, cell, span) -> Float64

Draw a length in metres from a lognormal [`SizeSpec`](@ref), scaled by
`ctx.length_scale`, floored at `min_cells` mesh cells and capped at `max_frac` of
`span` so a single feature cannot swallow the whole section.

`cell` and `span` must refer to the same axis: leave them at their defaults for a
vertical length, or pass `cell = ctx.dy, span = profile_span(ctx)` for a
horizontal one.
"""
function sample_length(ctx::SampleContext, spec::SizeSpec;
                       min_cells::Real = ctx.cfg.min_feature_cells,
                       max_frac::Real = 0.9, cell::Real = ctx.dz,
                       span::Real = depth_span(ctx))::Float64
    v = exp(spec.mu + spec.sigma * randn(ctx.rng)) * ctx.length_scale
    lo = spec.lo * ctx.length_scale
    hi = spec.hi * ctx.length_scale
    v = clamp(v, lo, hi)
    return clamp(v, min_cells * cell, max_frac * Float64(span))
end

# ─────────────────────────────────────────────────────────────────────────────
# Random fields
# ─────────────────────────────────────────────────────────────────────────────

"""
    random_field(rng, coords...; corr, kind, hurst, n_modes) -> Array

Zero-mean, unit-variance Gaussian random field by spectral synthesis: a sum of
`n_modes` random cosine modes,
`f(x) = √(2/M) Σ aₘ cos(kₘ·x + φₘ)`, empirically renormalised to unit standard
deviation so amplitudes are exact regardless of the mode count.

`coords` is one vector for a 1-D field (interface undulation) or two for a 2-D
field (textural noise). `corr` gives the correlation length per axis in metres,
which makes the field anisotropic by simply rescaling the coordinates.

`kind` selects the spectrum:
- `:gaussian` — squared-exponential covariance, `k ~ N(0, corr⁻²)`; smooth
- `:fractal` — power-law `|k|^-(hurst + d/2)` over a decade range; self-similar
  roughness, the usual choice for geological interfaces
"""
function random_field(rng::AbstractRNG, ys::AbstractVector{<:Real};
                      corr::Real, kind::Symbol = :fractal, hurst::Real = 0.75,
                      n_modes::Int = 96)::Vector{Float64}
    f = _spectral_sum(rng, (collect(Float64, ys),), (Float64(corr),), kind,
                      Float64(hurst), n_modes)
    return f
end

function random_field(rng::AbstractRNG, ys::AbstractVector{<:Real},
                      zs::AbstractVector{<:Real};
                      corr::Tuple{<:Real,<:Real}, kind::Symbol = :fractal,
                      hurst::Real = 0.75, n_modes::Int = 128)::Matrix{Float64}
    flat = _spectral_sum(rng, (collect(Float64, ys), collect(Float64, zs)),
                         (Float64(corr[1]), Float64(corr[2])), kind,
                         Float64(hurst), n_modes)
    return reshape(flat, length(ys), length(zs))
end

function _spectral_sum(rng::AbstractRNG, axes::Tuple, corr::Tuple,
                       kind::Symbol, hurst::Float64, n_modes::Int)
    d = length(axes)
    npts = prod(length.(axes))
    out = zeros(Float64, npts)
    scaled = ntuple(i -> collect(axes[i]) ./ max(corr[i], 1.0e-9), d)

    for _ in 1:n_modes
        k = if kind === :gaussian
            ntuple(_ -> randn(rng), d)
        elseif kind === :fractal
            # log-uniform wavenumber over ~2.5 decades, power-law amplitude
            mag = exp(_uniform(rng, log(0.05), log(12.0)))
            if d == 1
                (mag * (rand(rng) < 0.5 ? -1.0 : 1.0),)
            else
                θ = 2π * rand(rng)
                (mag * cos(θ), mag * sin(θ))
            end
        else
            error("Unknown random-field kind: $(kind) (use :gaussian or :fractal)")
        end
        amp = kind === :fractal ? _mode_norm(k)^(-(hurst + d / 2)) : 1.0
        φ = 2π * rand(rng)
        _accumulate_mode!(out, scaled, k, amp, φ)
    end

    σ = std(out)
    σ > 0 && (out ./= σ)
    out .-= mean(out)
    return out
end

_mode_norm(k::Tuple{Float64}) = max(abs(k[1]), 1.0e-9)
_mode_norm(k::Tuple{Float64,Float64}) = max(sqrt(k[1]^2 + k[2]^2), 1.0e-9)

function _accumulate_mode!(out::Vector{Float64}, scaled::Tuple{Vector{Float64}},
                           k::Tuple{Float64}, amp::Float64, φ::Float64)
    xs = scaled[1]
    @inbounds for i in eachindex(xs)
        out[i] += amp * cos(k[1] * xs[i] + φ)
    end
    return out
end

function _accumulate_mode!(out::Vector{Float64},
                           scaled::Tuple{Vector{Float64},Vector{Float64}},
                           k::Tuple{Float64,Float64}, amp::Float64, φ::Float64)
    xs, zs = scaled
    nx = length(xs)
    @inbounds for j in eachindex(zs)
        base = (j - 1) * nx
        kz = k[2] * zs[j] + φ
        for i in 1:nx
            out[base+i] += amp * cos(k[1] * xs[i] + kz)
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Rough polygons
# ─────────────────────────────────────────────────────────────────────────────

"""
    RoughPolygon

Closed polygon in the `(y, z)` cross-section, metres.
"""
struct RoughPolygon
    y::Vector{Float64}
    z::Vector{Float64}
end

"""
    random_polygon(rng; y0, z0, half_width, half_height, dip_deg, n_vertices, roughness)
        -> RoughPolygon

A star-shaped random polygon: vertices at jittered angles with radii modulated by
two low harmonics, scaled to the requested half-axes, rotated by `dip_deg` and
centred on `(y0, z0)`. `roughness` is the relative radius perturbation.
"""
function random_polygon(rng::AbstractRNG; y0::Real, z0::Real, half_width::Real,
                        half_height::Real, dip_deg::Real = 0.0,
                        n_vertices::Int = 9, roughness::Real = 0.25)
    n = max(n_vertices, 4)
    a1, a2 = roughness * randn(rng), 0.5 * roughness * randn(rng)
    p1, p2 = 2π * rand(rng), 2π * rand(rng)
    k2 = rand(rng, 2:4)
    θs = sort!([2π * (i - 1) / n + _uniform(rng, -0.35, 0.35) * 2π / n for i in 1:n])
    ys = Vector{Float64}(undef, n)
    zs = Vector{Float64}(undef, n)
    c, s = cosd(Float64(dip_deg)), sind(Float64(dip_deg))
    @inbounds for i in 1:n
        θ = θs[i]
        r = max(1 + a1 * cos(2θ + p1) + a2 * cos(k2 * θ + p2), 0.25)
        ly = r * half_width * cos(θ)
        lz = r * half_height * sin(θ)
        ys[i] = y0 + c * ly - s * lz
        zs[i] = z0 + s * ly + c * lz
    end
    return RoughPolygon(ys, zs)
end

"""
    inside(poly, y, z) -> Bool

Crossing-number point-in-polygon test.
"""
function inside(poly::RoughPolygon, y::Real, z::Real)::Bool
    py, pz = poly.y, poly.z
    n = length(py)
    result = false
    j = n
    @inbounds for i in 1:n
        if (pz[i] > z) != (pz[j] > z)
            yint = (py[j] - py[i]) * (z - pz[i]) / (pz[j] - pz[i]) + py[i]
            y < yint && (result = !result)
        end
        j = i
    end
    return result
end

"""
    scale_polygon(poly, factor) -> RoughPolygon

Dilate (or shrink) a polygon about its centroid.
"""
function scale_polygon(poly::RoughPolygon, factor::Real)::RoughPolygon
    cy, cz = mean(poly.y), mean(poly.z)
    f = Float64(factor)
    return RoughPolygon(cy .+ f .* (poly.y .- cy), cz .+ f .* (poly.z .- cz))
end

"""
    paint_polygon!(canvas, ctx, poly, value; halo_cells=0.0, halo_steps=3)

Stamp `value` (log10 Ω·m) onto every target cell inside `poly`.

`halo_cells > 0` adds an alteration-halo rim: nested dilations of the polygon
carry linearly decreasing blend weights outward, so the contact grades over
roughly that many cells instead of being mathematically sharp.
"""
function paint_polygon!(canvas::Matrix{Float64}, ctx::SampleContext,
                        poly::RoughPolygon, value::Real;
                        halo_cells::Real = 0.0, halo_steps::Int = 3)
    nzc, nyc = size(canvas)
    halo = Float64(halo_cells)

    rings = Tuple{RoughPolygon,Float64}[]
    if halo > 0
        extent = max(maximum(poly.y) - minimum(poly.y),
                     maximum(poly.z) - minimum(poly.z))
        reach = halo * max(ctx.dy, ctx.dz)
        grow = extent > 0 ? 1 + 2 * reach / extent : 1.0
        for s in halo_steps:-1:1
            f = 1 + (grow - 1) * s / halo_steps
            push!(rings, (scale_polygon(poly, f), 1 - s / (halo_steps + 1)))
        end
    end

    @inbounds for iy in 1:nyc, iz in 1:nzc
        y, z = ctx.ys[iy], ctx.zs[iz]
        if inside(poly, y, z)
            canvas[iz, iy] = value
            continue
        end
        for (ring, w) in rings
            if inside(ring, y, z)
                canvas[iz, iy] = (1 - w) * canvas[iz, iy] + w * value
                break
            end
        end
    end
    return canvas
end

# ─────────────────────────────────────────────────────────────────────────────
# Layered backgrounds
# ─────────────────────────────────────────────────────────────────────────────

"""
    LayerStack

A laterally varying layered sequence: `values[k]` is the log10 ρ of layer `k` and
`interfaces[k][iy]` the depth of its base beneath column `iy`.
"""
struct LayerStack
    values::Vector{Float64}
    interfaces::Vector{Vector{Float64}}
    transitions::Vector{Float64}
end

"""
    sample_layer_stack(ctx; n_layers=nothing, conductive_bias=nothing) -> LayerStack

Build a layered background. Thicknesses come from the prior layer-thickness
lognormal, each interface is undulated by a fractal random field whose amplitude
is a fraction of the layer thickness, and each interface gets its own transition
zone width. Layer resistivities are drawn from the mixture, with the chance of a
conductive layer set by the prior conductive metre fraction.
"""
function sample_layer_stack(ctx::SampleContext; n_layers::Union{Nothing,Int} = nothing,
                            conductive_bias::Union{Nothing,Float64} = nothing)::LayerStack
    nl = n_layers === nothing ? _randint(ctx.rng, ctx.cfg.n_layers_range) : n_layers
    nl = max(nl, 1)
    p_cond = conductive_bias === nothing ? ctx.priors.conductive_fraction : conductive_bias
    nyc = length(ctx.ys)
    zbot = ctx.zs[end] + ctx.dz

    values = Vector{Float64}(undef, nl)
    @inbounds for k in 1:nl
        r = rand(ctx.rng)
        values[k] = if r < p_cond
            sample_log10_rho(ctx, :conductive)
        elseif r < p_cond + 0.15
            sample_log10_rho(ctx, :resistive)
        else
            sample_log10_rho(ctx, :host)
        end
    end

    interfaces = Vector{Vector{Float64}}()
    transitions = Float64[]
    depth = ctx.zs[1]
    for k in 1:(nl-1)
        thickness = sample_length(ctx, ctx.priors.layer_thickness; max_frac = 0.6)
        depth += thickness
        depth >= zbot && (depth = zbot - 0.5 * ctx.dz)
        rough = _uniform(ctx.rng, ctx.cfg.interface_roughness_range) * thickness
        field = rough > 0 ?
                random_field(ctx.rng, ctx.ys;
                             corr = max(ctx.priors.corr_horizontal * ctx.length_scale,
                                        2 * ctx.dy),
                             kind = ctx.cfg.texture_kind, hurst = ctx.cfg.hurst,
                             n_modes = 48) : zeros(nyc)
        push!(interfaces, depth .+ rough .* field)
        push!(transitions, _uniform(ctx.rng, ctx.cfg.transition_cells_range) * ctx.dz)
    end
    return LayerStack(values, interfaces, transitions)
end

"""
    fill_layers!(canvas, ctx, stack)

Rasterise a [`LayerStack`](@ref) onto the target window. Contacts are blended
with a smoothstep over each interface's transition width, so gradational and
sharp boundaries coexist in the training set.
"""
function fill_layers!(canvas::Matrix{Float64}, ctx::SampleContext, stack::LayerStack)
    nzc, nyc = size(canvas)
    nl = length(stack.values)
    @inbounds for iy in 1:nyc
        for iz in 1:nzc
            z = ctx.zs[iz]
            v = stack.values[nl]
            for k in 1:(nl-1)
                zk = stack.interfaces[k][iy]
                t = stack.transitions[k]
                if z <= zk - t / 2
                    v = stack.values[k]
                    break
                elseif t > 0 && z < zk + t / 2
                    u = (z - (zk - t / 2)) / t
                    w = u * u * (3 - 2u)          # smoothstep
                    v = (1 - w) * stack.values[k] + w * stack.values[k+1]
                    break
                end
            end
            canvas[iz, iy] = v
        end
    end
    return canvas
end

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

"""
    body_axes(ctx) -> (half_width, half_height)

Half-axes for a discrete body: the horizontal extent is drawn from the prior
across-strike width and the vertical one from the prior thickness, then nudged
toward the prior thickness/width aspect ratio.
"""
function body_axes(ctx::SampleContext)
    w = sample_length(ctx, ctx.priors.body_width; cell = ctx.dy,
                      span = profile_span(ctx), max_frac = 0.8)
    h = sample_length(ctx, ctx.priors.body_thickness; max_frac = 0.7)
    target_h = w * ctx.priors.aspect_ratio * exp(0.5 * randn(ctx.rng))
    h = clamp(0.5 * (h + target_h), ctx.cfg.min_feature_cells * ctx.dz,
              0.7 * depth_span(ctx))
    return (0.5 * w, 0.5 * h)
end

"""
    add_body!(canvas, ctx, meta_list; kind=nothing) -> Dict

Insert one rough polygon body and return its metadata entry.
"""
function add_body!(canvas::Matrix{Float64}, ctx::SampleContext,
                   meta_list::Vector{Any}; kind::Union{Nothing,Symbol} = nothing)
    k = kind === nothing ? (rand(ctx.rng) < 0.7 ? :conductive : :resistive) : kind
    hw, hh = body_axes(ctx)
    y0 = _uniform(ctx.rng, ctx.ys[1] + hw, ctx.ys[end] - hw)
    z0 = _uniform(ctx.rng, ctx.zs[1] + hh, ctx.zs[end] - hh)
    dip = _uniform(ctx.rng, -40.0, 40.0)
    value = sample_log10_rho(ctx, k)
    poly = random_polygon(ctx.rng; y0 = y0, z0 = z0, half_width = hw,
                          half_height = hh, dip_deg = dip,
                          n_vertices = _randint(ctx.rng, (6, 12)),
                          roughness = _uniform(ctx.rng, 0.05, 0.35))
    paint_polygon!(canvas, ctx, poly, value;
                   halo_cells = _uniform(ctx.rng, 0.0, 2.0))
    entry = Dict{String,Any}(
        "kind" => String(k), "log10_rho" => round(value; digits = 3),
        "y_center_m" => round(y0; digits = 1), "z_center_m" => round(z0; digits = 1),
        "half_width_m" => round(hw; digits = 1), "half_height_m" => round(hh; digits = 1),
        "rotation_deg" => round(dip; digits = 1),
    )
    push!(meta_list, entry)
    return entry
end

"""
    build_halfspace!(canvas, ctx)

Uniform half-space. Kept in the library at low weight so the network also sees
the trivial case and does not hallucinate structure into flat data.
"""
function build_halfspace!(canvas::Matrix{Float64}, ctx::SampleContext)
    v = sample_log10_rho(ctx, :any)
    fill!(canvas, v)
    ctx.meta["background_log10_rho"] = round(v; digits = 3)
    return canvas
end

"""
    build_layered!(canvas, ctx)

Purely 1-D layered section with rough interfaces.
"""
function build_layered!(canvas::Matrix{Float64}, ctx::SampleContext)
    stack = sample_layer_stack(ctx)
    fill_layers!(canvas, ctx, stack)
    _record_stack!(ctx, stack)
    return canvas
end

"""
    build_layered_block!(canvas, ctx)

Layered background with one buried body — the classic MT test case.
"""
function build_layered_block!(canvas::Matrix{Float64}, ctx::SampleContext)
    stack = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (2, 4)))
    fill_layers!(canvas, ctx, stack)
    _record_stack!(ctx, stack)
    bodies = Any[]
    add_body!(canvas, ctx, bodies)
    ctx.meta["bodies"] = bodies
    return canvas
end

"""
    build_multi_anomaly!(canvas, ctx)

Layered background plus several independent bodies at random depths.
"""
function build_multi_anomaly!(canvas::Matrix{Float64}, ctx::SampleContext)
    stack = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (1, 3)))
    fill_layers!(canvas, ctx, stack)
    _record_stack!(ctx, stack)
    bodies = Any[]
    for _ in 1:_randint(ctx.rng, ctx.cfg.n_anomalies_range)
        add_body!(canvas, ctx, bodies)
    end
    ctx.meta["bodies"] = bodies
    return canvas
end

"""
    build_basin!(canvas, ctx)

Conductive basin over resistive basement: the basement top is a broad smooth
surface modulated by a long-wavelength random field, and the fill may be split
into two units. Generic sedimentary geology, unrelated to Keivitsa.
"""
function build_basin!(canvas::Matrix{Float64}, ctx::SampleContext)
    nzc, nyc = size(canvas)
    span = ctx.zs[end] - ctx.zs[1]
    depth0 = _uniform(ctx.rng, 0.15, 0.6) * span + ctx.zs[1]
    relief = _uniform(ctx.rng, 0.1, 0.5) * span
    field = random_field(ctx.rng, ctx.ys;
                         corr = max(0.4 * (ctx.ys[end] - ctx.ys[1]), 4 * ctx.dy),
                         kind = :gaussian, n_modes = 24)
    basement = depth0 .+ relief .* field
    fill_log = sample_log10_rho(ctx, :conductive)
    base_log = sample_log10_rho(ctx, :resistive)
    has_cover = rand(ctx.rng) < 0.6
    cover_log = has_cover ? sample_log10_rho(ctx, :host) : fill_log
    cover_frac = _uniform(ctx.rng, 0.15, 0.5)
    trans = _uniform(ctx.rng, ctx.cfg.transition_cells_range) * ctx.dz

    @inbounds for iy in 1:nyc
        zb = clamp(basement[iy], ctx.zs[1] + ctx.dz, ctx.zs[end])
        zc = ctx.zs[1] + cover_frac * (zb - ctx.zs[1])
        for iz in 1:nzc
            z = ctx.zs[iz]
            v = if z < zc
                cover_log
            elseif z < zb
                fill_log
            else
                base_log
            end
            if trans > 0 && abs(z - zb) < trans / 2
                u = (z - (zb - trans / 2)) / trans
                w = u * u * (3 - 2u)
                v = (1 - w) * fill_log + w * base_log
            end
            canvas[iz, iy] = v
        end
    end
    ctx.meta["basin"] = Dict{String,Any}(
        "basement_depth_m" => round(depth0; digits = 1),
        "relief_m" => round(relief; digits = 1),
        "fill_log10_rho" => round(fill_log; digits = 3),
        "cover_log10_rho" => round(cover_log; digits = 3),
        "basement_log10_rho" => round(base_log; digits = 3),
    )
    return canvas
end

"""
    build_dyke!(canvas, ctx)

Layered section cut by a steeply dipping tabular body of constant width, the
COMEMI-style dyke case.
"""
function build_dyke!(canvas::Matrix{Float64}, ctx::SampleContext)
    stack = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (1, 4)))
    fill_layers!(canvas, ctx, stack)
    _record_stack!(ctx, stack)

    nzc, nyc = size(canvas)
    dip = _uniform(ctx.rng, ctx.cfg.dip_range)
    width = sample_length(ctx, ctx.priors.body_width; cell = ctx.dy,
                          span = profile_span(ctx), max_frac = 0.5)
    y_top = _uniform(ctx.rng, ctx.ys[1] + width, ctx.ys[end] - width)
    z_top = _uniform(ctx.rng, ctx.zs[1], ctx.zs[1] + 0.3 * (ctx.zs[end] - ctx.zs[1]))
    z_bot = _uniform(ctx.rng, z_top + 0.3 * (ctx.zs[end] - z_top), ctx.zs[end])
    value = rand(ctx.rng) < 0.6 ? sample_log10_rho(ctx, :conductive) :
            sample_log10_rho(ctx, :resistive)
    slope = 1 / tand(clamp(dip, 1.0, 179.0))
    trace = random_field(ctx.rng, ctx.zs; corr = max(0.5 * (ctx.zs[end] - ctx.zs[1]),
                                                     4 * ctx.dz),
                         kind = :gaussian, n_modes = 16)
    wobble = _uniform(ctx.rng, 0.0, 0.4) * width

    @inbounds for iz in 1:nzc
        z = ctx.zs[iz]
        (z_top <= z <= z_bot) || continue
        yc = y_top + slope * (z - z_top) + wobble * trace[iz]
        for iy in 1:nyc
            abs(ctx.ys[iy] - yc) <= width / 2 && (canvas[iz, iy] = value)
        end
    end
    ctx.meta["dyke"] = Dict{String,Any}(
        "dip_deg" => round(dip; digits = 1), "width_m" => round(width; digits = 1),
        "y_top_m" => round(y_top; digits = 1),
        "z_range_m" => [round(z_top; digits = 1), round(z_bot; digits = 1)],
        "log10_rho" => round(value; digits = 3),
    )
    return canvas
end

"""
    offset_interfaces!(stack, ctx; y_surface, slope, throw_m, damage)

Displace a layer stack across a dipping fault plane
`y(z) = y_surface + slope·(z − z_top)`.

Each interface is offset where *its own* depth crosses the plane, so the
displaced boundary steps laterally as it deepens — the defining feature of a
dipping fault, which a single column-wise shift cannot reproduce. The trace
position is evaluated at the undisplaced depth, a first-order solution of the
implicit geometry that is well within a cell for realistic throws.
"""
function offset_interfaces!(stack::LayerStack, ctx::SampleContext;
                            y_surface::Real, slope::Real, throw_m::Real, damage::Real)
    nyc = length(ctx.ys)
    z_top = ctx.zs[1]
    @inbounds for iface in stack.interfaces
        for iy in 1:nyc
            y_trace = y_surface + slope * (iface[iy] - z_top)
            u = clamp((ctx.ys[iy] - (y_trace - damage / 2)) / max(damage, 1.0e-9),
                      0.0, 1.0)
            iface[iy] += throw_m * (u * u * (3 - 2u))
        end
    end
    return stack
end

"""
    build_dipping_fault!(canvas, ctx)

Layered section broken by a dipping normal/reverse fault: the hanging wall is
displaced by a vertical throw that tapers across a damage zone, and a conductive
gouge band may be draped along the fault plane.
"""
function build_dipping_fault!(canvas::Matrix{Float64}, ctx::SampleContext)
    stack = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (2, 6)))
    nzc, nyc = size(canvas)
    dip = _uniform(ctx.rng, ctx.cfg.dip_range)
    y_surface = _uniform(ctx.rng, ctx.ys[1] + 0.2 * (ctx.ys[end] - ctx.ys[1]),
                         ctx.ys[end] - 0.2 * (ctx.ys[end] - ctx.ys[1]))
    slope = 1 / tand(clamp(dip, 1.0, 179.0))
    throw_m = _uniform(ctx.rng, 1.0, 6.0) * ctx.dz *
              (rand(ctx.rng) < 0.5 ? -1.0 : 1.0)
    damage = _uniform(ctx.rng, 1.0, 5.0) * ctx.dy

    offset_interfaces!(stack, ctx; y_surface = y_surface, slope = slope,
                       throw_m = throw_m, damage = damage)
    fill_layers!(canvas, ctx, stack)
    _record_stack!(ctx, stack)

    gouge = rand(ctx.rng) < 0.7
    if gouge
        width = _uniform(ctx.rng, 1.0, 3.5) * ctx.dy
        value = sample_log10_rho(ctx, :conductive)
        @inbounds for iz in 1:nzc
            yc = y_surface + slope * (ctx.zs[iz] - ctx.zs[1])
            for iy in 1:nyc
                abs(ctx.ys[iy] - yc) <= width / 2 && (canvas[iz, iy] = value)
            end
        end
        ctx.meta["fault_gouge"] = Dict{String,Any}(
            "width_m" => round(width; digits = 1),
            "log10_rho" => round(value; digits = 3))
    end
    ctx.meta["fault"] = Dict{String,Any}(
        "dip_deg" => round(dip; digits = 1),
        "surface_intercept_m" => round(y_surface; digits = 1),
        "throw_m" => round(throw_m; digits = 1),
        "damage_zone_m" => round(damage; digits = 1),
        "has_gouge" => gouge,
    )
    return canvas
end

"""
    build_thrust_sheet!(canvas, ctx)

Shallow-dipping thrust: an allochthonous sheet with its own stratigraphy sits on
an undulating sole above the autochthonous sequence, so the section repeats — a
structure the network cannot get from Keivitsa's flat-lying logs.
"""
function build_thrust_sheet!(canvas::Matrix{Float64}, ctx::SampleContext)
    nzc, nyc = size(canvas)
    upper = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (1, 3)))
    lower = sample_layer_stack(ctx; n_layers = _randint(ctx.rng, (2, 4)))

    span = ctx.zs[end] - ctx.zs[1]
    dip = _uniform(ctx.rng, ctx.cfg.thrust_dip_range) * (rand(ctx.rng) < 0.5 ? -1 : 1)
    z_mid = _uniform(ctx.rng, 0.25, 0.7) * span + ctx.zs[1]
    slope = tand(dip)
    y_mid = 0.5 * (ctx.ys[1] + ctx.ys[end])
    undulation = random_field(ctx.rng, ctx.ys;
                              corr = max(0.3 * (ctx.ys[end] - ctx.ys[1]), 4 * ctx.dy),
                              kind = :fractal, hurst = ctx.cfg.hurst, n_modes = 32)
    amp = _uniform(ctx.rng, 0.02, 0.15) * span
    sole = [clamp(z_mid + slope * (ctx.ys[iy] - y_mid) + amp * undulation[iy],
                  ctx.zs[1] + ctx.dz, ctx.zs[end] - ctx.dz) for iy in 1:nyc]

    top = similar(canvas)
    bottom = similar(canvas)
    fill_layers!(top, ctx, upper)
    fill_layers!(bottom, ctx, lower)
    trans = _uniform(ctx.rng, ctx.cfg.transition_cells_range) * ctx.dz
    @inbounds for iy in 1:nyc, iz in 1:nzc
        z = ctx.zs[iz]
        zs_ = sole[iy]
        canvas[iz, iy] = if trans > 0 && abs(z - zs_) < trans / 2
            u = (z - (zs_ - trans / 2)) / trans
            w = u * u * (3 - 2u)
            (1 - w) * top[iz, iy] + w * bottom[iz, iy]
        else
            z < zs_ ? top[iz, iy] : bottom[iz, iy]
        end
    end

    ctx.meta["thrust"] = Dict{String,Any}(
        "dip_deg" => round(dip; digits = 1),
        "sole_depth_m" => round(z_mid; digits = 1),
        "undulation_amplitude_m" => round(amp; digits = 1),
        "upper_log10_rho" => round.(upper.values; digits = 3),
        "lower_log10_rho" => round.(lower.values; digits = 3),
    )
    return canvas
end

function _record_stack!(ctx::SampleContext, stack::LayerStack)
    ctx.meta["layers"] = Dict{String,Any}(
        "n" => length(stack.values),
        "log10_rho" => round.(stack.values; digits = 3),
        "mean_interface_depth_m" => [round(mean(iface); digits = 1)
                                     for iface in stack.interfaces],
        "transition_m" => round.(stack.transitions; digits = 2),
    )
    return ctx
end

const _BUILDERS = Dict{Symbol,Function}(
    :halfspace => build_halfspace!,
    :layered => build_layered!,
    :layered_block => build_layered_block!,
    :multi_anomaly => build_multi_anomaly!,
    :basin => build_basin!,
    :dyke => build_dyke!,
    :dipping_fault => build_dipping_fault!,
    :thrust_sheet => build_thrust_sheet!,
)

"""
    apply_texture!(canvas, ctx)

Superimpose an anisotropic random field on `log10(ρ)` using the prior vertical
and horizontal correlation lengths, so cells are never piecewise constant and the
network cannot rely on perfectly homogeneous units.
"""
function apply_texture!(canvas::Matrix{Float64}, ctx::SampleContext)
    amp = _uniform(ctx.rng, ctx.cfg.texture_log10_range)
    amp <= 0 && return canvas
    corr_z = max(ctx.priors.corr_vertical * ctx.length_scale, 1.5 * ctx.dz)
    corr_y = max(ctx.priors.corr_horizontal * ctx.length_scale, 1.5 * ctx.dy)
    field = random_field(ctx.rng, ctx.ys, ctx.zs; corr = (corr_y, corr_z),
                         kind = ctx.cfg.texture_kind, hurst = ctx.cfg.hurst)
    nzc, nyc = size(canvas)
    @inbounds for iy in 1:nyc, iz in 1:nzc
        canvas[iz, iy] += amp * field[iy, iz]
    end
    ctx.meta["texture"] = Dict{String,Any}(
        "amplitude_log10" => round(amp; digits = 3),
        "kind" => String(ctx.cfg.texture_kind),
        "corr_horizontal_m" => round(corr_y; digits = 1),
        "corr_vertical_m" => round(corr_z; digits = 1),
    )
    return canvas
end

# ─────────────────────────────────────────────────────────────────────────────
# Model assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    SyntheticModel

One synthetic section.

# Fields
- `log10_rho`: `(n_target_z, n_core_y)` `log10(Ω·m)`, depth-major like the solver
- `scenario`: which builder produced it
- `seed`: the per-model RNG seed, so the model is reproducible on its own
- `metadata`: every sampled parameter, JSON-serialisable
"""
struct SyntheticModel
    log10_rho::Matrix{Float32}
    scenario::Symbol
    seed::Int
    metadata::Dict{String,Any}
end

function Base.show(io::IO, m::SyntheticModel)
    v = m.log10_rho
    @printf(io, "SyntheticModel(%s, %d×%d, log10rho %.2f…%.2f, seed=%d)",
            m.scenario, size(v, 1), size(v, 2), minimum(v), maximum(v), m.seed)
end

"""
    sample_scenario(rng, cfg) -> Symbol

Draw a scenario according to `cfg.scenario_weights`.
"""
function sample_scenario(rng::AbstractRNG, cfg::GeneratorConfig)::Symbol
    w = cfg.scenario_weights
    length(w) == length(SCENARIOS) ||
        error("scenario_weights needs $(length(SCENARIOS)) entries, got $(length(w))")
    total = sum(max.(w, 0.0))
    total > 0 || error("scenario_weights must contain a positive entry")
    t = rand(rng) * total
    acc = 0.0
    for (i, wi) in enumerate(w)
        acc += max(wi, 0.0)
        t <= acc && return SCENARIOS[i]
    end
    return SCENARIOS[end]
end

"""
    generate_model(mesh, priors, cfg; rng, scenario=nothing, seed=nothing, index=0)
        -> SyntheticModel

Draw one synthetic section on the target window of `mesh`.

`seed` (or a draw from `rng`) is stored in the model so it can be regenerated in
isolation. `scenario` forces a specific builder; otherwise it is sampled from
`cfg.scenario_weights`.
"""
function generate_model(mesh::GeneratorMesh, priors::PriorSpec,
                        cfg::GeneratorConfig = GeneratorConfig();
                        rng::AbstractRNG = Random.default_rng(),
                        scenario::Union{Nothing,Symbol} = nothing,
                        seed::Union{Nothing,Integer} = nothing,
                        index::Integer = 0)::SyntheticModel
    model_seed = seed === nothing ? rand(rng, 1:typemax(Int32)) : Int(seed)
    mrng = MersenneTwister(model_seed)

    yc = y_centers(mesh)
    zc = z_centers(mesh)
    ys = yc[mesh.core_y]
    zs = zc[mesh.target_z]
    scale = resolve_length_scale(cfg, priors)
    mode = rand(mrng) < cfg.generic_fraction ? :generic : :keivitsa
    # Draw unconditionally, even when the scenario is forced, so the RNG stream
    # stays aligned and `(seed, scenario)` always regenerates the same model.
    sampled = sample_scenario(mrng, cfg)
    scen = scenario === nothing ? sampled : scenario
    haskey(_BUILDERS, scen) || error("Unknown scenario $(scen); expected one of $(SCENARIOS)")

    meta = Dict{String,Any}(
        "index" => Int(index),
        "scenario" => String(scen),
        "seed" => model_seed,
        "prior_mode" => String(mode),
        "length_scale" => round(scale; digits = 4),
        "prior_source" => priors.source,
    )
    ctx = SampleContext(mrng, cfg, priors, collect(ys), collect(zs),
                        cfg.y_core_cell, cfg.target_dz, scale, mode, meta)

    canvas = zeros(Float64, length(zs), length(ys))
    _BUILDERS[scen](canvas, ctx)
    apply_texture!(canvas, ctx)
    lo, hi = cfg.log10_clip
    clamp!(canvas, lo, hi)

    meta["stats"] = Dict{String,Any}(
        "log10_min" => round(minimum(canvas); digits = 3),
        "log10_median" => round(median(canvas); digits = 3),
        "log10_max" => round(maximum(canvas); digits = 3),
        "log10_std" => round(std(canvas); digits = 3),
        "conductive_cell_fraction" =>
            round(count(<(2.0), canvas) / length(canvas); digits = 4),
    )
    return SyntheticModel(Float32.(canvas), scen, model_seed, meta)
end

"""
    solver_resistivity(model, mesh; pad=:replicate, deep_background=nothing)
        -> Matrix{Float64}

Expand a target-window model into the full `(n_z, n_y)` linear-Ω·m array that
`MTGeophysics.run_mt2d_forward` expects.

- rows `1:n_air_cells` are set to `1e9` Ω·m (the solver's air value);
- lateral padding columns replicate the nearest core column (`:replicate`) or
  relax exponentially toward the section's median resistivity (`:decay`);
- the deep zone below the target window continues the bottom target row, blending
  toward `deep_background` (Ω·m) when given.

The padding and deep zone are modelling scaffolding, never training targets.
"""
function solver_resistivity(model::SyntheticModel, mesh::GeneratorMesh;
                            pad::Symbol = :replicate,
                            deep_background::Union{Nothing,Real} = nothing)::Matrix{Float64}
    return solver_resistivity(model.log10_rho, mesh; pad = pad,
                              deep_background = deep_background)
end

function solver_resistivity(log10_rho::AbstractMatrix{<:Real}, mesh::GeneratorMesh;
                            pad::Symbol = :replicate,
                            deep_background::Union{Nothing,Real} = nothing)::Matrix{Float64}
    nzt, nyc = size(log10_rho)
    (nzt, nyc) == target_size(mesh) ||
        error("model is $(size(log10_rho)) but the mesh target is $(target_size(mesh))")

    full = Matrix{Float64}(undef, n_z(mesh), n_y(mesh))
    med = median(log10_rho)

    # target window
    @inbounds for (jj, iy) in enumerate(mesh.core_y), (ii, iz) in enumerate(mesh.target_z)
        full[iz, iy] = Float64(log10_rho[ii, jj])
    end

    # deep zone: continue the bottom row, optionally relaxing to a deep background
    z_deep = (last(mesh.target_z)+1):n_z(mesh)
    deep_log = deep_background === nothing ? nothing : log10(Float64(deep_background))
    ndeep = length(z_deep)
    @inbounds for (jj, iy) in enumerate(mesh.core_y)
        base = Float64(log10_rho[end, jj])
        for (kk, iz) in enumerate(z_deep)
            full[iz, iy] = if deep_log === nothing
                base
            else
                w = ndeep == 1 ? 1.0 : (kk - 1) / (ndeep - 1)
                (1 - w) * base + w * deep_log
            end
        end
    end

    # lateral padding
    left, right = first(mesh.core_y), last(mesh.core_y)
    ground = (mesh.n_air_cells+1):n_z(mesh)
    @inbounds for iz in ground
        vl, vr = full[iz, left], full[iz, right]
        for iy in 1:(left-1)
            full[iz, iy] = pad === :decay ?
                           _pad_decay(vl, med, (left - iy) / max(left - 1, 1)) : vl
        end
        for iy in (right+1):n_y(mesh)
            full[iz, iy] = pad === :decay ?
                           _pad_decay(vr, med, (iy - right) / max(n_y(mesh) - right, 1)) : vr
        end
    end

    ρ = 10.0 .^ full
    ρ[1:mesh.n_air_cells, :] .= AIR_RESISTIVITY
    return ρ
end

_pad_decay(edge::Float64, background::Float64, u::Float64)::Float64 =
    edge + (background - edge) * (1 - exp(-3u))

"""
    forward_response(model, mesh; mode=:TETM, kwargs...)

Run MTGeophysics.jl's 2-D TE/TM solver on a generated model and return its
`MT2DResponse` (`rho_xy`/`phase_xy` for TE, `rho_yx`/`phase_yx` for TM, each
sized `(n_frequencies, n_stations)`).

MTGeophysics.jl is imported lazily here, so generating and storing a dataset does
not require it.
"""
function forward_response(model::SyntheticModel, mesh::GeneratorMesh;
                          mode::Symbol = :TETM, kwargs...)
    MTG = mtgeophysics()
    ρ = solver_resistivity(model, mesh; kwargs...)
    return _mtg_call(MTG.run_mt2d_forward, to_mt2d_mesh(mesh), ρ; mode = mode)
end

# ─────────────────────────────────────────────────────────────────────────────
# Batch generation and storage
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_dataset(path; n_models, cfg, priors, priors_path, seed, format,
                     compress, log_every) -> String

Generate `n_models` sections and stream them to disk. Memory stays flat: the
HDF5 dataset is preallocated and written one model at a time, chunked per model
and compressed.

# Layout (`format = :hdf5`)
- `log10_rho` — `(n_target_z, n_core_y, n_models)` `Float32`
- `scenario`, `prior_mode` — string vectors, one entry per model
- `seed`, `index` — `Int64` vectors
- `metadata_json` — the full per-model parameter record
- `mesh/…` — `y_nodes`, `z_nodes`, `y_cell_sizes`, `z_cell_sizes`,
  `receiver_positions`, `frequencies`, `core_y`, `target_z`
- `config_json`, and root attributes (`schema`, `n_air_cells`, units, …)

`format = :jld2` stores the same content as native Julia objects, including the
`GeneratorMesh` and `SyntheticModel` values themselves.
"""
function generate_dataset(path::AbstractString;
                          n_models::Integer = 1000,
                          cfg::GeneratorConfig = GeneratorConfig(),
                          priors::Union{Nothing,PriorSpec} = nothing,
                          priors_path::AbstractString = "",
                          mesh::Union{Nothing,GeneratorMesh} = nothing,
                          seed::Union{Nothing,Integer} = nothing,
                          format::Symbol = :hdf5,
                          compress::Int = 4,
                          log_every::Integer = 100)::String
    n_models > 0 || error("n_models must be positive")
    gmesh = mesh === nothing ? build_generator_mesh(cfg) : mesh
    pr = priors === nothing ? load_generator_priors(priors_path) : priors
    base_seed = seed === nothing ? cfg.seed : Int(seed)
    rng = MersenneTwister(base_seed)
    nzt, nyc = target_size(gmesh)
    scale = resolve_length_scale(cfg, pr)

    @info "Generating synthetic MT2D dataset" n_models target=(nzt, nyc) solver=(n_z(gmesh), n_y(gmesh)) length_scale=round(scale; digits=3) out=path
    mkpath(dirname(abspath(path)))

    if format === :jld2
        models = Vector{SyntheticModel}(undef, n_models)
        for i in 1:n_models
            models[i] = generate_model(gmesh, pr, cfg; rng = rng, index = i)
            _progress(i, n_models, log_every)
        end
        JLD2.jldsave(abspath(path);
                     schema = SCHEMA_VERSION, mesh = gmesh, config = cfg,
                     models = models, priors_source = pr.source,
                     length_scale = scale, base_seed = base_seed,
                     created_utc = _timestamp())
        @info "Dataset written" path=abspath(path) format=:jld2 n_models=n_models
        return abspath(path)
    end

    format === :hdf5 || error("format must be :hdf5 or :jld2, got $(format)")

    scenarios = Vector{String}(undef, n_models)
    modes = Vector{String}(undef, n_models)
    seeds = Vector{Int64}(undef, n_models)
    metas = Vector{String}(undef, n_models)

    h5open(abspath(path), "w") do fid
        dset = create_dataset(fid, "log10_rho", datatype(Float32),
                              dataspace((nzt, nyc, Int(n_models)));
                              chunk = (nzt, nyc, 1), shuffle = true, deflate = compress)
        for i in 1:n_models
            m = generate_model(gmesh, pr, cfg; rng = rng, index = i)
            dset[:, :, i] = m.log10_rho
            scenarios[i] = String(m.scenario)
            modes[i] = String(get(m.metadata, "prior_mode", "unknown"))
            seeds[i] = m.seed
            metas[i] = JSON3.write(_sanitize(m.metadata))
            _progress(i, n_models, log_every)
        end

        fid["scenario"] = scenarios
        fid["prior_mode"] = modes
        fid["seed"] = seeds
        fid["index"] = collect(Int64, 1:n_models)
        fid["metadata_json"] = metas
        fid["config_json"] = JSON3.write(_sanitize(_config_dict(cfg)))
        fid["priors_json"] = JSON3.write(_sanitize(_priors_dict(pr)))

        g = create_group(fid, "mesh")
        g["y_nodes"] = gmesh.y_nodes
        g["z_nodes"] = gmesh.z_nodes
        g["y_cell_sizes"] = gmesh.y_cell_sizes
        g["z_cell_sizes"] = gmesh.z_cell_sizes
        g["receiver_positions"] = gmesh.receiver_positions
        g["frequencies"] = gmesh.frequencies
        g["core_y"] = collect(Int64, gmesh.core_y)
        g["target_z"] = collect(Int64, gmesh.target_z)
        attributes(g)["n_air_cells"] = gmesh.n_air_cells
        attributes(g)["n_y"] = n_y(gmesh)
        attributes(g)["n_z"] = n_z(gmesh)
        attributes(g)["air_resistivity_ohm_m"] = AIR_RESISTIVITY
        attributes(g)["layout"] = "solver arrays are (n_z, n_y): depth first, profile second"

        attributes(fid)["schema"] = SCHEMA_VERSION
        attributes(fid)["created_utc"] = _timestamp()
        attributes(fid)["julia_version"] = string(VERSION)
        attributes(fid)["n_models"] = Int(n_models)
        attributes(fid)["base_seed"] = base_seed
        attributes(fid)["length_scale"] = scale
        attributes(fid)["priors_source"] = pr.source
        attributes(fid)["units"] = "log10(ohm.m); lengths m; frequency Hz"
        attributes(fid)["target_layout"] = "(n_target_z, n_core_y, n_models)"
    end

    @info "Dataset written" path=abspath(path) format=:hdf5 n_models=n_models scenarios=_tally(scenarios)
    return abspath(path)
end

"""
    save_dataset(path, models, mesh, cfg, priors; format=:hdf5) -> String

Persist an already-generated vector of models. Uses the same layout as
[`generate_dataset`](@ref).
"""
function save_dataset(path::AbstractString, models::Vector{SyntheticModel},
                      mesh::GeneratorMesh, cfg::GeneratorConfig, priors::PriorSpec;
                      format::Symbol = :hdf5, compress::Int = 4)::String
    isempty(models) && error("no models to save")
    nzt, nyc = size(models[1].log10_rho)
    mkpath(dirname(abspath(path)))
    if format === :jld2
        JLD2.jldsave(abspath(path); schema = SCHEMA_VERSION, mesh = mesh, config = cfg,
                     models = models, priors_source = priors.source,
                     created_utc = _timestamp())
        return abspath(path)
    end
    h5open(abspath(path), "w") do fid
        dset = create_dataset(fid, "log10_rho", datatype(Float32),
                              dataspace((nzt, nyc, length(models)));
                              chunk = (nzt, nyc, 1), shuffle = true, deflate = compress)
        for (i, m) in enumerate(models)
            dset[:, :, i] = m.log10_rho
        end
        fid["scenario"] = [String(m.scenario) for m in models]
        fid["prior_mode"] = [String(get(m.metadata, "prior_mode", "unknown")) for m in models]
        fid["seed"] = Int64[m.seed for m in models]
        fid["index"] = collect(Int64, 1:length(models))
        fid["metadata_json"] = [JSON3.write(_sanitize(m.metadata)) for m in models]
        fid["config_json"] = JSON3.write(_sanitize(_config_dict(cfg)))
        fid["priors_json"] = JSON3.write(_sanitize(_priors_dict(priors)))
        g = create_group(fid, "mesh")
        g["y_nodes"] = mesh.y_nodes
        g["z_nodes"] = mesh.z_nodes
        g["y_cell_sizes"] = mesh.y_cell_sizes
        g["z_cell_sizes"] = mesh.z_cell_sizes
        g["receiver_positions"] = mesh.receiver_positions
        g["frequencies"] = mesh.frequencies
        g["core_y"] = collect(Int64, mesh.core_y)
        g["target_z"] = collect(Int64, mesh.target_z)
        attributes(g)["n_air_cells"] = mesh.n_air_cells
        attributes(fid)["schema"] = SCHEMA_VERSION
        attributes(fid)["n_models"] = length(models)
        attributes(fid)["created_utc"] = _timestamp()
    end
    return abspath(path)
end

"""
    load_dataset(path) -> NamedTuple

Read a dataset back. Returns
`(log10_rho, mesh, scenario, prior_mode, seed, metadata, attrs)` where
`log10_rho` is `(n_target_z, n_core_y, n_models)` and `mesh` is a reconstructed
[`GeneratorMesh`](@ref) ready for [`solver_resistivity`](@ref).
"""
function load_dataset(path::AbstractString)
    p = abspath(path)
    isfile(p) || error("Dataset not found: $(p)")
    if endswith(lowercase(p), ".jld2")
        d = JLD2.load(p)
        return (log10_rho = cat((m.log10_rho for m in d["models"])...; dims = 3),
                mesh = d["mesh"],
                scenario = [String(m.scenario) for m in d["models"]],
                prior_mode = [String(get(m.metadata, "prior_mode", "unknown"))
                              for m in d["models"]],
                seed = Int64[m.seed for m in d["models"]],
                metadata = [m.metadata for m in d["models"]],
                attrs = Dict{String,Any}("schema" => d["schema"]))
    end
    h5open(p, "r") do fid
        g = fid["mesh"]
        core_y = read(g["core_y"])
        target_z = read(g["target_z"])
        mesh = GeneratorMesh(read(g["y_nodes"]), read(g["z_nodes"]),
                             read(g["y_cell_sizes"]), read(g["z_cell_sizes"]),
                             read(g["receiver_positions"]), read(g["frequencies"]),
                             read(attributes(g)["n_air_cells"]),
                             first(core_y):last(core_y),
                             first(target_z):last(target_z),
                             Dict{String,Any}())
        attrs = Dict{String,Any}(k => read(attributes(fid)[k])
                                 for k in keys(attributes(fid)))
        metas = [JSON3.read(s, Dict{String,Any}) for s in read(fid["metadata_json"])]
        return (log10_rho = read(fid["log10_rho"]), mesh = mesh,
                scenario = read(fid["scenario"]), prior_mode = read(fid["prior_mode"]),
                seed = read(fid["seed"]), metadata = metas, attrs = attrs)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_timestamp() = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")

function _progress(i::Integer, n::Integer, every::Integer)
    (every > 0 && (i % every == 0 || i == n)) || return nothing
    @info "  progress" done=i of=n
    return nothing
end

function _tally(xs::Vector{String})
    t = Dict{String,Int}()
    for x in xs
        t[x] = get(t, x, 0) + 1
    end
    return t
end

function _config_dict(cfg::GeneratorConfig)::Dict{String,Any}
    d = Dict{String,Any}()
    for f in fieldnames(GeneratorConfig)
        v = getfield(cfg, f)
        d[String(f)] = v isa Tuple ? collect(v) : (v isa Symbol ? String(v) : v)
    end
    return d
end

_pop_dict(p::PopulationSpec) = Dict{String,Any}(
    "name" => p.name, "log10_mean" => p.mean, "log10_std" => p.std,
    "weight" => p.weight, "log10_range" => [p.lo, p.hi])

_size_dict(s::SizeSpec) = Dict{String,Any}(
    "name" => s.name, "lognormal_mu" => s.mu, "lognormal_sigma" => s.sigma,
    "p05" => s.lo, "p95" => s.hi)

function _priors_dict(p::PriorSpec)::Dict{String,Any}
    return Dict{String,Any}(
        "source" => p.source,
        "populations" => [_pop_dict(x) for x in p.populations],
        "layer_thickness_m" => _size_dict(p.layer_thickness),
        "body_thickness_m" => _size_dict(p.body_thickness),
        "body_width_m" => _size_dict(p.body_width),
        "aspect_ratio" => p.aspect_ratio,
        "conductive_fraction" => p.conductive_fraction,
        "corr_vertical_m" => p.corr_vertical,
        "corr_horizontal_m" => p.corr_horizontal,
        "anisotropy" => p.anisotropy,
        "log10_clip" => collect(p.log10_clip),
    )
end

"""
    _sanitize(x)

Recursively make a value JSON-safe: non-finite floats become `nothing`, tuples
and symbols become arrays and strings.
"""
function _sanitize(x)
    if x isa AbstractDict
        return Dict{String,Any}(string(k) => _sanitize(v) for (k, v) in x)
    elseif x isa Symbol
        return String(x)
    elseif x isa Tuple
        return Any[_sanitize(v) for v in x]
    elseif x isa AbstractRange
        return collect(x)
    elseif x isa AbstractVector
        return Any[_sanitize(v) for v in x]
    elseif x isa AbstractFloat
        return isfinite(x) ? Float64(x) : nothing
    else
        return x
    end
end

end # module SyntheticGenerator
