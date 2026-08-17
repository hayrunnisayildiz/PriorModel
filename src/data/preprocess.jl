"""
    DataPreprocess

Geophysical + borehole preprocessing for the Smart Prior deep-learning pipeline.

Reads GTK `.xyz` / `.XYZ` surface surveys and collar / survey / petrophysics
tables, then produces training-ready tensors:

- **Input `X`**: `(N_x, N_y, C_in)` multi-channel surface anomaly maps
  (gravity, magnetic, EM/IP, …) on the active [`GridSpec`](@ref).
- **Target `Y`**: `(N_x, N_y, N_z, C_out)` subsurface volume with borehole
  petrophysics (`density`, `LUO_R`) and a well-path mask `M_sondaj`.

Surface workflow per channel:
1. Clip `(X, Y)` to the grid bounding box
2. Remove statistical outliers (IQR or z-score)
3. Interpolate scattered samples onto cell centres (`:rbf`, `:kriging`, or `:idw`)

Borehole workflow:
1. Merge collar + survey → 3-D trajectory (balanced tangential desurvey)
2. Assign `DSR_D` / `PTR_D` and `LUO_R` at traversed voxels
3. Build binary mask along the hole path

Persist with HDF5 (default) or JLD2; iterate mini-batches via [`PatchDataLoader`](@ref).
"""
module DataPreprocess

using Statistics
using Random
using HDF5
using JLD2
using LinearAlgebra
using NearestNeighbors
using StaticArrays

if !isdefined(@__MODULE__, :GridSpecs)
    include(joinpath(@__DIR__, "..", "fusion", "GridSpec.jl"))
end
if !isdefined(@__MODULE__, :Voxelizer)
    include(joinpath(@__DIR__, "..", "fusion", "Voxelizer.jl"))
end
using .GridSpecs
using .Voxelizer

const GridSpec = GridSpecs.GridSpec

export PreprocessOptions, PreprocessedDataset, PatchDataLoader, PatchBatch
export preprocess, preprocess!
export save_dataset, load_dataset
export input_channel_names, target_channel_names
export surface_channel_table, iterate, n_batches

const FILL = Voxelizer.FILL
const P2 = SVector{2,Float32}

"""Default output channels for `Y`: density, resistivity, well mask."""
const TARGET_NAMES = ("density", "resistivity", "M_sondaj")

# ─────────────────────────────────────────────────────────────────────────────
# Options & dataset container
# ─────────────────────────────────────────────────────────────────────────────

"""
    PreprocessOptions

Controls clipping, outlier rejection, interpolation, and trajectory sampling.

# Fields
- `outlier_method`: `:iqr` (default) or `:zscore`
- `outlier_k`: IQR multiplier or z-score threshold
- `interp_method`: `:rbf` (Gaussian kernel), `:kriging` (OK + exponential variogram), or `:idw`
- `rbf_sigma`: Gaussian RBF length scale (m)
- `variogram_range`, `variogram_sill`, `variogram_nugget`: kriging parameters (m, variance units)
- `idw_k`, `idw_power`, `max_radius`: neighbour IDW settings (m)
- `trajectory_step`: along-hole sampling interval for the mask (m)
"""
Base.@kwdef struct PreprocessOptions
    outlier_method::Symbol = :iqr
    outlier_k::Float32 = 3.0f0
    interp_method::Symbol = :rbf
    rbf_sigma::Float32 = 75.0f0
    variogram_range::Float32 = 150.0f0
    variogram_sill::Float32 = 1.0f0
    variogram_nugget::Float32 = 0.05f0
    idw_k::Int = 8
    idw_power::Float32 = 2.0f0
    max_radius::Float32 = 200.0f0
    trajectory_step::Float32 = 12.5f0
end

"""
    PreprocessedDataset

- `X`: `(N_x, N_y, C_in)` surface inputs (`Float32`, off-grid / empty → `NaN32`)
- `Y`: `(N_x, N_y, N_z, C_out)` targets (`Float32`; channels 1–2 petrophysics, 3 mask)
- `grid`: active [`GridSpec`](@ref)
- `input_names`, `target_names`: channel labels
"""
struct PreprocessedDataset
    X::Array{Float32,3}
    Y::Array{Float32,4}
    grid::GridSpec
    input_names::Vector{String}
    target_names::Vector{String}
    meta::Dict{String,Any}
end

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Surface-only `tensor_channels` entries (ground + airborne geophysics)."""
function surface_channel_table(cfg::AbstractDict)::Vector{Dict{String,Any}}
    return [ch for ch in Voxelizer.tensor_channel_table(cfg)
            if startswith(string(ch["source"]), "ground_geophysics.") ||
               startswith(string(ch["source"]), "aero_geophysics.")]
end

input_channel_names(cfg::AbstractDict)::Vector{String} =
    [string(ch["name"]) for ch in surface_channel_table(cfg)]

target_channel_names() = collect(String, TARGET_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# Point filtering
# ─────────────────────────────────────────────────────────────────────────────

"""Keep samples whose `(x,y)` lie inside the grid XY extent."""
function clip_to_bbox(px::Vector{Float32}, py::Vector{Float32}, pv::Vector{Float32},
                      grid::GridSpec)
    keep = BitVector(undef, length(px))
    @inbounds for i in eachindex(px)
        keep[i] = (grid.xmin <= px[i] < grid.xmax) &&
                  (grid.ymin <= py[i] < grid.ymax) &&
                  isfinite(px[i]) && isfinite(py[i]) && isfinite(pv[i])
    end
    return px[keep], py[keep], pv[keep]
end

"""IQR fences or z-score threshold on `pv`; returns filtered triplets."""
function filter_outliers(px::Vector{Float32}, py::Vector{Float32}, pv::Vector{Float32};
                         method::Symbol=:iqr, k::Float32=3.0f0)
    n = length(pv)
    n == 0 && return px, py, pv
    finite = .!isnan.(pv)
    count(finite) == 0 && return Float32[], Float32[], Float32[]

    if method == :iqr
        vals = pv[finite]
        q1, q3 = quantile(vals, (0.25, 0.75))
        iqr = q3 - q1
        lo = Float32(q1 - k * iqr)
        hi = Float32(q3 + k * iqr)
        keep = finite .& (pv .>= lo) .& (pv .<= hi)
    elseif method == :zscore
        vals = pv[finite]
        μ = mean(vals)
        σ = std(vals)
        σ <= 0 && return px, py, pv
        keep = finite .& (abs.((pv .- μ) ./ σ) .<= k)
    else
        error("Unknown outlier_method=$(repr(method)); use :iqr or :zscore")
    end
    return px[keep], py[keep], pv[keep]
end

# ─────────────────────────────────────────────────────────────────────────────
# 2-D interpolation (RBF / Kriging / IDW)
# ─────────────────────────────────────────────────────────────────────────────

"""Inverse-distance weighting onto a regular `(nx, ny)` mesh."""
function idw_interpolate_2d(xc::Vector{Float32}, yc::Vector{Float32},
                            px::Vector{Float32}, py::Vector{Float32}, pv::Vector{Float32};
                            k::Int=8, power::Float32=2.0f0,
                            max_radius::Float32=200.0f0)::Matrix{Float32}
    return Voxelizer.idw_kdtree_2d(xc, yc, px, py, pv;
                                   k=k, power=power, max_radius=max_radius)
end

"""
Gaussian RBF (Shepard) interpolation.

``f(x^*) = \\sum_i w_i v_i / \\sum_i w_i`` with ``w_i = \\exp(-(d_i/\\sigma)^2)``.
"""
function rbf_interpolate_2d(xc::Vector{Float32}, yc::Vector{Float32},
                            px::Vector{Float32}, py::Vector{Float32}, pv::Vector{Float32};
                            k::Int=12, sigma::Float32=75.0f0,
                            max_radius::Float32=200.0f0)::Matrix{Float32}
    nxx = length(xc)
    nyy = length(yc)
    out = fill(FILL, nxx, nyy)
    keep = .!isnan.(pv) .& .!isnan.(px) .& .!isnan.(py)
    count(keep) == 0 && return out

    pts = P2[P2(px[i], py[i]) for i in eachindex(px) if keep[i]]
    vals = Float32[pv[i] for i in eachindex(pv) if keep[i]]
    tree = KDTree(pts)
    k_use = min(k, length(pts))
    σ = max(sigma, 1.0f0)
    inv_2σ2 = 1.0f0 / (2.0f0 * σ * σ)

    @inbounds for j in 1:nyy, i in 1:nxx
        idxs, dists = knn(tree, P2(xc[i], yc[j]), k_use, true)
        d0 = Float32(dists[1])
        d0 > max_radius && continue
        if d0 <= 1.0f-3
            out[i, j] = vals[idxs[1]]
            continue
        end
        num = 0.0f0
        den = 0.0f0
        for t in 1:k_use
            d = Float32(dists[t])
            d > max_radius && break
            w = exp(-d * d * inv_2σ2)
            num += w * vals[idxs[t]]
            den += w
        end
        den > 0.0f0 && (out[i, j] = num / den)
    end
    return out
end

"""Exponential variogram ``γ(h) = nugget + sill·(1 - exp(-h/a))``."""
@inline function _exp_variogram(h::Float32, range::Float32, sill::Float32, nugget::Float32)
    return nugget + sill * (1.0f0 - exp(-h / max(range, 1.0f0)))
end

"""
Ordinary kriging with an exponential variogram (local `k` neighbours).

Solves the OK system for each target location on the grid mesh.
"""
function kriging_interpolate_2d(xc::Vector{Float32}, yc::Vector{Float32},
                                px::Vector{Float32}, py::Vector{Float32}, pv::Vector{Float32};
                                k::Int=12, max_radius::Float32=200.0f0,
                                range::Float32=150.0f0, sill::Float32=1.0f0,
                                nugget::Float32=0.05f0)::Matrix{Float32}
    nxx = length(xc)
    nyy = length(yc)
    out = fill(FILL, nxx, nyy)
    keep = .!isnan.(pv) .& .!isnan.(px) .& .!isnan.(py)
    count(keep) == 0 && return out

    pts = P2[P2(px[i], py[i]) for i in eachindex(px) if keep[i]]
    vals = Float32[pv[i] for i in eachindex(pv) if keep[i]]
    tree = KDTree(pts)
    k_use = min(k, length(pts))

    @inbounds for j in 1:nyy, i in 1:nxx
        idxs, dists = knn(tree, P2(xc[i], yc[j]), k_use, true)
        d0 = Float32(dists[1])
        d0 > max_radius && continue
        m = 0
        for t in 1:k_use
            Float32(dists[t]) <= max_radius && (m += 1)
        end
        m == 0 && continue
        n = m + 1
        K = Matrix{Float64}(undef, n, n)
        rhs = Vector{Float64}(undef, n)
        for a in 1:m, b in 1:m
            pa = pts[idxs[a]]
            pb = pts[idxs[b]]
            hab = hypot(pa[1] - pb[1], pa[2] - pb[2])
            K[a, b] = _exp_variogram(hab, range, sill, nugget)
        end
        for a in 1:m
            pa = pts[idxs[a]]
            da = hypot(pa[1] - xc[i], pa[2] - yc[j])
            K[a, a] = _exp_variogram(da, range, sill, nugget)
            K[a, n] = 1.0
            K[n, a] = 1.0
            rhs[a] = vals[idxs[a]]
        end
        K[n, n] = 0.0
        rhs[n] = 0.0
        λμ = K \ rhs
        out[i, j] = Float32(λμ[1:m]' * vals[idxs[1:m]])
    end
    return out
end

"""Dispatch surface interpolation according to `opts.interp_method`."""
function interpolate_surface_2d(xc, yc, px, py, pv, opts::PreprocessOptions)
    if opts.interp_method == :rbf
        return rbf_interpolate_2d(xc, yc, px, py, pv;
                                  k=opts.idw_k, sigma=opts.rbf_sigma,
                                  max_radius=opts.max_radius)
    elseif opts.interp_method == :kriging
        return kriging_interpolate_2d(xc, yc, px, py, pv;
                                      k=opts.idw_k, max_radius=opts.max_radius,
                                      range=opts.variogram_range,
                                      sill=opts.variogram_sill,
                                      nugget=opts.variogram_nugget)
    elseif opts.interp_method == :idw
        return idw_interpolate_2d(xc, yc, px, py, pv;
                                  k=opts.idw_k, power=opts.idw_power,
                                  max_radius=opts.max_radius)
    else
        error("Unknown interp_method=$(repr(opts.interp_method)); use :rbf, :kriging, or :idw")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface tensor  X  : (Nx, Ny, Cin)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_surface_tensor(grid, config_path, opts) -> (X, names)

Stack clipped / filtered / interpolated surface channels into `(N_x, N_y, C_in)`.
"""
function build_surface_tensor(grid::GridSpec, config_path::AbstractString,
                              opts::PreprocessOptions=PreprocessOptions())
    cfg = GridSpecs.load_config(config_path)
    root = GridSpecs.project_root(GridSpecs.resolve_config_path(config_path))
    channels = surface_channel_table(cfg)
    active = Voxelizer.active_grid_name(cfg)
    nxx, nyy = GridSpecs.nx(grid), GridSpecs.ny(grid)
    nc = length(channels)
    X = fill(FILL, nxx, nyy, nc)
    xc = GridSpecs.x_centers(grid)
    yc = GridSpecs.y_centers(grid)

    for (c, ch) in enumerate(channels)
        spec = Voxelizer.lookup_source(cfg, string(ch["source"]))
        px, py, pv = Voxelizer._load_surface_xy(root, spec, string(ch["name"]), active)
        n_raw = length(pv)
        px, py, pv = clip_to_bbox(px, py, pv, grid)
        px, py, pv = filter_outliers(px, py, pv;
                                     method=opts.outlier_method, k=opts.outlier_k)
        @info "Surface channel" name=ch["name"] raw=n_raw kept=length(pv) method=opts.interp_method
        if length(pv) > 0
            X[:, :, c] = interpolate_surface_2d(xc, yc, px, py, pv, opts)
        end
    end
    return X, [string(ch["name"]) for ch in channels]
end

# ─────────────────────────────────────────────────────────────────────────────
# Borehole target  Y  : (Nx, Ny, Nz, Cout)
# ─────────────────────────────────────────────────────────────────────────────

"""Read petrophysics table and return desurveyed `(px,py,pz,p_density,p_resist)`."""
function _load_petrophysics_desurveyed(root::AbstractString, cfg::AbstractDict,
                                       ctx::Voxelizer.BoreholeContext)
    petro_yaml = cfg["borehole"]["petrophysics"]["path"]
    path = Voxelizer.resolve_table_path(root, string(petro_yaml))
    df = Voxelizer.read_gtk_txt(path)

    idc = Voxelizer.find_column(df, ["Tunnus", "HOLE_ID"])
    depc = Voxelizer.find_column(df, ["Syvyys", "DEPTH"])
    dsrc = Voxelizer.find_column(df, ["DSR_D"])
    ptrc = Voxelizer.find_column(df, ["PTR_D"])
    resc = Voxelizer.find_column(df, ["LUO_R"])
    (idc === nothing || depc === nothing) && error("Petrophysics missing Tunnus/Syvyys")
    (dsrc === nothing && ptrc === nothing) && error("Petrophysics missing DSR_D/PTR_D")
    resc === nothing && @warn "LUO_R column not found; resistivity channel stays empty"

    ids = Voxelizer.col_str(df, idc)
    deps = Voxelizer.col_f32(df, depc)
    dsr = dsrc === nothing ? fill(FILL, length(ids)) : Voxelizer.col_f32(df, dsrc)
    ptr = ptrc === nothing ? fill(FILL, length(ids)) : Voxelizer.col_f32(df, ptrc)
    res = resc === nothing ? fill(FILL, length(ids)) : Voxelizer.col_f32(df, resc)

    px = Float32[]; py = Float32[]; pz = Float32[]
    pd = Float32[]; pr = Float32[]
    @inbounds for i in eachindex(ids)
        hid = ids[i]
        (isempty(hid) || isnan(deps[i]) || !haskey(ctx.collars, hid)) && continue
        dens = isnan(dsr[i]) ? ptr[i] : dsr[i]
        (isnan(dens) && isnan(res[i])) && continue
        x, y, z = Voxelizer.desurvey_md(ctx.collars[hid],
                                        Voxelizer._stations(ctx, hid), deps[i])
        push!(px, x); push!(py, y); push!(pz, z)
        push!(pd, dens); push!(pr, res[i])
    end
    @info "Petrophysics desurveyed" n=length(px) file=basename(path)
    return px, py, pz, pd, pr
end

"""Mark voxels along every surveyed hole (for the well mask)."""
function _rasterize_trajectory_mask!(mask::Array{Float32,3}, grid::GridSpec,
                                     ctx::Voxelizer.BoreholeContext,
                                     step::Float32)
    nxx, nyy, nzz = size(mask)
    @inbounds for (hid, collar) in ctx.collars
        stations = Voxelizer._stations(ctx, hid)
        md_max = collar.length
        if isnan(md_max) || md_max <= 0
            md_max = stations === nothing || isempty(stations) ? step :
                     stations[end].depth
        end
        md = 0.0f0
        while md <= md_max + 1.0f-3
            x, y, z = Voxelizer.desurvey_md(collar, stations, md)
            if GridSpecs.in_bounds(grid, x, y, z)
                i, j, k = GridSpecs.coord_to_index(grid, x, y, z)
                if 1 <= i <= nxx && 1 <= j <= nyy && 1 <= k <= nzz
                    mask[i, j, k] = 1.0f0
                end
            end
            md += max(step, 1.0f0)
        end
    end
    return mask
end

"""Splat petrophysics samples into nearest voxels (last finite write wins)."""
function _splat_petrophysics!(density::Array{Float32,3}, resistivity::Array{Float32,3},
                              mask::Array{Float32,3}, grid::GridSpec,
                              px, py, pz, pd, pr)
    nxx, nyy, nzz = size(density)
    @inbounds for p in eachindex(px)
        (isnan(px[p]) || isnan(py[p]) || isnan(pz[p])) && continue
        i, j, k = GridSpecs.coord_to_index(grid, px[p], py[p], pz[p])
        (1 <= i <= nxx && 1 <= j <= nyy && 1 <= k <= nzz) || continue
        if isfinite(pd[p])
            density[i, j, k] = pd[p]
            mask[i, j, k] = 1.0f0
        end
        if isfinite(pr[p])
            resistivity[i, j, k] = pr[p]
            mask[i, j, k] = 1.0f0
        end
    end
    return density, resistivity, mask
end

"""
    build_borehole_target(grid, config_path, opts) -> Y

`(N_x, N_y, N_z, 3)` volume: `[density, LUO_R, M_sondaj]`.
Empty cells → `NaN32` (petrophysics) or `0` (mask).
"""
function build_borehole_target(grid::GridSpec, config_path::AbstractString,
                               opts::PreprocessOptions=PreprocessOptions())
    cfg = GridSpecs.load_config(config_path)
    root = GridSpecs.project_root(GridSpecs.resolve_config_path(config_path))
    nxx, nyy, nzz = GridSpecs.nxyz(grid)

    density = fill(FILL, nxx, nyy, nzz)
    resistivity = fill(FILL, nxx, nyy, nzz)
    mask = zeros(Float32, nxx, nyy, nzz)

    ctx = Voxelizer.load_borehole_context(root, cfg)
    px, py, pz, pd, pr = _load_petrophysics_desurveyed(root, cfg, ctx)
    _splat_petrophysics!(density, resistivity, mask, grid, px, py, pz, pd, pr)
    _rasterize_trajectory_mask!(mask, grid, ctx, opts.trajectory_step)

    Y = cat(density, resistivity, mask; dims=4)
    return Y
end

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

"""
    preprocess(config_path; block=nothing, opts=PreprocessOptions()) -> PreprocessedDataset

Run surface + borehole preprocessing and return an in-memory dataset.
"""
function preprocess(config_path::AbstractString;
                    block::Union{Nothing,AbstractString}=nothing,
                    opts::PreprocessOptions=PreprocessOptions())::PreprocessedDataset
    grid = GridSpecs.load_gridspec(config_path; block=block)
    cfg = GridSpecs.load_config(config_path)
    X, in_names = build_surface_tensor(grid, config_path, opts)
    Y = build_borehole_target(grid, config_path, opts)
    meta = Dict{String,Any}(
        "config_path" => abspath(GridSpecs.resolve_config_path(config_path)),
        "grid_block" => block === nothing ? Voxelizer.active_grid_name(cfg) : string(block),
        "interp_method" => string(opts.interp_method),
        "outlier_method" => string(opts.outlier_method),
        "epsg" => grid.epsg_code,
    )
    return PreprocessedDataset(X, Y, grid, in_names, target_channel_names(), meta)
end

"""In-place alias for [`preprocess`](@ref)."""
preprocess!(config_path::AbstractString; kwargs...) = preprocess(config_path; kwargs...)

# ─────────────────────────────────────────────────────────────────────────────
# Persistence  (.h5 / .jld2)
# ─────────────────────────────────────────────────────────────────────────────

"""Write grid metadata as HDF5 attributes on group `g`."""
function _write_grid_attrs(g, grid::GridSpec)
    attrs = attributes(g)
    for (k, v) in (
        ("xmin", grid.xmin), ("xmax", grid.xmax),
        ("ymin", grid.ymin), ("ymax", grid.ymax),
        ("zmin", grid.zmin), ("zmax", grid.zmax),
        ("dx", grid.dx), ("dy", grid.dy), ("dz", grid.dz),
        ("epsg", grid.epsg_code),
        ("nx", GridSpecs.nx(grid)), ("ny", GridSpecs.ny(grid)), ("nz", GridSpecs.nz(grid)),
    )
        attrs[k] = v
    end
end

"""
    save_dataset(path, dataset; format=:auto)

Persist `X`, `Y`, channel names, and grid metadata.

`format`: `:auto` (from extension), `:h5`, or `:jld2`.
"""
function save_dataset(path::AbstractString, ds::PreprocessedDataset;
                      format::Symbol=:auto)
    mkpath(dirname(abspath(path)))
    fmt = format
    if fmt == :auto
        ext = lowercase(splitext(path)[2])
        fmt = ext == ".jld2" ? :jld2 : :h5
    end
    if fmt == :h5
        h5open(path, "w") do f
            write(f, "X", ds.X)
            write(f, "Y", ds.Y)
            _write_grid_attrs(f, ds.grid)
            attrs = attributes(f)
            attrs["input_channel_names"] = ds.input_names
            attrs["target_channel_names"] = ds.target_names
            for (k, v) in ds.meta
                attrs["meta/" * k] = v
            end
        end
    elseif fmt == :jld2
        jldsave(path;
                X=ds.X, Y=ds.Y, grid=ds.grid,
                input_names=ds.input_names, target_names=ds.target_names,
                meta=ds.meta)
    else
        error("Unknown format=$(repr(format)); use :h5 or :jld2")
    end
    @info "Saved preprocessed dataset" path=path format=fmt size_X=size(ds.X) size_Y=size(ds.Y)
    return path
end

"""
    load_dataset(path) -> PreprocessedDataset

Load from HDF5 or JLD2 (detected from extension).
"""
function load_dataset(path::AbstractString)::PreprocessedDataset
    ext = lowercase(splitext(path)[2])
    if ext in (".h5", ".hdf5")
        h5open(path, "r") do f
            X = read(f, "X")
            Y = read(f, "Y")
            attrs = attributes(f)
            grid = GridSpec(
                Float32(read(attrs["xmin"])), Float32(read(attrs["xmax"])),
                Float32(read(attrs["ymin"])), Float32(read(attrs["ymax"])),
                Float32(read(attrs["zmin"])), Float32(read(attrs["zmax"])),
                Float32(read(attrs["dx"])), Float32(read(attrs["dy"])), Float32(read(attrs["dz"])),
                Int32(read(attrs["epsg"])),
            )
            in_names = Vector{String}(read(attrs["input_channel_names"]))
            tgt_names = Vector{String}(read(attrs["target_channel_names"]))
            meta = Dict{String,Any}()
            for k in keys(attrs)
                sk = string(k)
                startswith(sk, "meta/") && (meta[sk[6:end]] = read(attrs[k]))
            end
            return PreprocessedDataset(X, Y, grid, in_names, tgt_names, meta)
        end
    elseif ext == ".jld2"
        data = jldopen(path, "r") do file
            (; X=read(file, "X"), Y=read(file, "Y"), grid=read(file, "grid"),
               input_names=read(file, "input_names"), target_names=read(file, "target_names"),
               meta=haskey(file, "meta") ? read(file, "meta") : Dict{String,Any}())
        end
        return PreprocessedDataset(data.X, data.Y, data.grid,
                                   Vector{String}(data.input_names),
                                   Vector{String}(data.target_names),
                                   data.meta)
    else
        error("Unsupported dataset extension $(ext); use .h5 or .jld2")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch DataLoader (spatial patches)
# ─────────────────────────────────────────────────────────────────────────────

"""
    PatchBatch

One mini-batch: `X` is `(px, py, C_in, B)`, `Y` is `(px, py, pz, C_out, B)`.
"""
struct PatchBatch
    X::Array{Float32,4}
    Y::Array{Float32,5}
    origins::Vector{NTuple{3,Int}}   # (i0, j0, k0) top-left of each patch
end

"""
    PatchDataLoader(dataset; patch_xy, patch_z, batch_size, n_batches, shuffle, rng)

Random spatial crop iterator for single-domain grids.

Each batch draws `batch_size` patches of size `(patch_xy, patch_xy, patch_z)`.
"""
struct PatchDataLoader
    dataset::PreprocessedDataset
    patch_xy::Int
    patch_z::Int
    batch_size::Int
    n_batches::Int
    shuffle::Bool
    rng::Random.AbstractRNG
end

function PatchDataLoader(ds::PreprocessedDataset;
                         patch_xy::Int=64, patch_z::Int=16,
                         batch_size::Int=4, n_batches::Int=32,
                         shuffle::Bool=true,
                         rng::Random.AbstractRNG=Random.default_rng())
    nxx, nyy, nzz = GridSpecs.nxyz(ds.grid)
    patch_xy <= nxx && patch_xy <= nyy && patch_z <= nzz ||
        error("Patch size ($patch_xy,$patch_xy,$patch_z) exceeds grid ($nxx,$nyy,$nzz)")
    return PatchDataLoader(ds, patch_xy, patch_z, batch_size, n_batches, shuffle, rng)
end

n_batches(loader::PatchDataLoader)::Int = loader.n_batches

"""Random `(i0, j0, k0)` origin for a valid patch."""
function _random_origin(n::Int, patch::Int, rng::AbstractRNG)::Int
    return patch >= n ? 1 : rand(rng, 1:(n - patch + 1))
end

"""Collect one mini-batch of spatial patches."""
function _next_patch_batch!(loader::PatchDataLoader)
    ds = loader.dataset
    px = loader.patch_xy
    pz = loader.patch_z
    B = loader.batch_size
    cin = size(ds.X, 3)
    cout = size(ds.Y, 4)
    nxx, nyy, nzz = GridSpecs.nxyz(ds.grid)

    Xb = Array{Float32,4}(undef, px, px, cin, B)
    Yb = Array{Float32,5}(undef, px, px, pz, cout, B)
    origins = NTuple{3,Int}[]

    for b in 1:B
        i0 = _random_origin(nxx, px, loader.rng)
        j0 = _random_origin(nyy, px, loader.rng)
        k0 = _random_origin(nzz, pz, loader.rng)
        push!(origins, (i0, j0, k0))
        Xb[:, :, :, b] = ds.X[i0:(i0 + px - 1), j0:(j0 + px - 1), :]
        Yb[:, :, :, :, b] = ds.Y[i0:(i0 + px - 1), j0:(j0 + px - 1),
                                  k0:(k0 + pz - 1), :]
    end
    return PatchBatch(Xb, Yb, origins)
end

Base.iterate(loader::PatchDataLoader, state=0) =
    state >= loader.n_batches ? nothing : (_next_patch_batch!(loader), state + 1)

end # module DataPreprocess
