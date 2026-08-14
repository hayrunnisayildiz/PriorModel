"""
    ForwardGravity

Analytic 3-D prism gravity forward operator (Nagy 1966; Okabe 1979;
Nagy, Papp & Benedek 2000).

Maps a voxel density model ``m`` (``g/cm^3``) onto predicted vertical
gravity anomalies ``g_{pred}`` (mGal, positive downward) at surface
observation points.

Each cell of the [`GridSpec`](@ref GridSpecs.GridSpec) is treated as a
homogeneous rectangular prism. The vertical attraction is the 8-corner
closed-form kernel; the hot loop is SIMD-vectorised with
`LoopVectorization.@turbo` and writes only into a preallocated output
(zero allocations on the voxel path).

# Coordinates
World coordinates are metres, `z` positive upward (metres RL), matching
`GridSpec`. Internally the Nagy kernel is evaluated in a local frame
with ``z`` positive **downward** (``z_\\mathrm{down} = z_\\mathrm{obs} - z_\\mathrm{prism}``)
so a positive density contrast below the station yields a positive mGal
anomaly.

# Units
- density: ``g/cm^3``
- coordinates: metres
- gravity: mGal (``1\\,\\mathrm{mGal} = 10^{-5}\\,\\mathrm{m}/s^2``)
"""
module ForwardGravity

using LoopVectorization
using Base.Threads: @threads

if !isdefined(@__MODULE__, :GridSpecs)
    include(joinpath(@__DIR__, "..", "fusion", "GridSpec.jl"))
end
using .GridSpecs

export GravityPrism, prism_gz, prism_gz_kernel
export forward_gravity, forward_gravity!
export gravity_kernel_matrix, apply_gravity_kernel
export G_SI, G_MGAL, G_CM3_TO_KG_M3, M_S2_TO_MGAL

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

"""Newtonian constant ``G`` (CODATA 2018), ``m^3 kg^{-1} s^{-2}``."""
const G_SI::Float64 = 6.67430e-11

"""``1\\,g/cm^3 = 1000\\,kg/m^3``."""
const G_CM3_TO_KG_M3::Float64 = 1.0e3

"""``1\\,m/s^2 = 10^5\\,mGal``."""
const M_S2_TO_MGAL::Float64 = 1.0e5

"""
    G_MGAL

Scale from (density in ``g/cm^3``) × (Nagy kernel in metres) to mGal:

```
G_MGAL = G_SI × 1000 × 1e5 = 6.67430e-3
```
"""
const G_MGAL::Float64 = G_SI * G_CM3_TO_KG_M3 * M_S2_TO_MGAL

"""Floor for ``r`` and ``x+r`` / ``y+r`` so `log` / `atan` stay defined at a corner."""
const _EPS::Float64 = 1.0e-30

# ─────────────────────────────────────────────────────────────────────────────
# Nagy / Okabe kernel (one corner + 8-corner prism)
# ─────────────────────────────────────────────────────────────────────────────

"""
    nagy_gz_corner(x, y, z) -> Float64

Nagy vertical-attraction term at one prism corner (metres).

Local frame: origin at the station, ``z`` positive downward.

```
K(x,y,z) = x log(y + r) + y log(x + r) − z arctan(xy / (z r))
```

with ``r = \\sqrt{x^2 + y^2 + z^2}``.
"""
@inline function nagy_gz_corner(x::Float64, y::Float64, z::Float64)::Float64
    r = sqrt(x * x + y * y + z * z)
    r = max(r, _EPS)
    return x * log(max(y + r, _EPS)) +
           y * log(max(x + r, _EPS)) -
           z * atan(x * y, z * r)
end

"""
    prism_gz_kernel(x1, x2, y1, y2, z1, z2) -> Float64

8-corner inclusion-exclusion of [`nagy_gz_corner`](@ref) (metres).

Corners are **local** coordinates (prism vertex minus station in ``x,y``;
``z_\\mathrm{obs} - z_\\mathrm{vertex}`` in ``z``). Sign of each corner is
``μ_{ijk} = (-1)^{i+j+k}`` with 1-based ``i,j,k ∈ \\{1,2\\}``.
"""
@inline function prism_gz_kernel(x1::Float64, x2::Float64,
                                 y1::Float64, y2::Float64,
                                 z1::Float64, z2::Float64)::Float64
    return (
        -nagy_gz_corner(x1, y1, z1) + nagy_gz_corner(x2, y1, z1) +
         nagy_gz_corner(x1, y2, z1) - nagy_gz_corner(x2, y2, z1) +
         nagy_gz_corner(x1, y1, z2) - nagy_gz_corner(x2, y1, z2) -
         nagy_gz_corner(x1, y2, z2) + nagy_gz_corner(x2, y2, z2)
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# GravityPrism
# ─────────────────────────────────────────────────────────────────────────────

"""
    GravityPrism

Axis-aligned rectangular prism in world coordinates (metres, `z` up).

# Fields
- `x1, x2`: easting faces, ``x1 < x2`` (m)
- `y1, y2`: northing faces, ``y1 < y2`` (m)
- `z1, z2`: RL faces, ``z1 < z2`` (`z1` deepest, `z2` shallowest) (m)
"""
struct GravityPrism
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
    z1::Float64
    z2::Float64
    function GravityPrism(x1::Real, x2::Real, y1::Real, y2::Real, z1::Real, z2::Real)
        xa, xb = extrema((Float64(x1), Float64(x2)))
        ya, yb = extrema((Float64(y1), Float64(y2)))
        za, zb = extrema((Float64(z1), Float64(z2)))
        return new(xa, xb, ya, yb, za, zb)
    end
end

"""
    GravityPrism(grid::GridSpec, i, j, k) -> GravityPrism

Prism occupying voxel `(i, j, k)` (1-based) of `grid`.
Matches the `GridSpec` cell convention: `k = 1` is the deepest slice.
"""
function GravityPrism(grid::GridSpec, i::Integer, j::Integer, k::Integer)
    dx = Float64(grid.dx)
    dy = Float64(grid.dy)
    dz = Float64(grid.dz)
    x1 = Float64(grid.xmin) + Float64(i - 1) * dx
    y1 = Float64(grid.ymin) + Float64(j - 1) * dy
    z1 = Float64(grid.zmin) + Float64(k - 1) * dz
    return GravityPrism(x1, x1 + dx, y1, y1 + dy, z1, z1 + dz)
end

"""
    prism_gz(prism, ox, oy, oz, density=1) -> Float64

Vertical gravity (mGal, positive downward) of `prism` at station
`(ox, oy, oz)` (m, `z` up). `density` is ``g/cm^3``.
"""
function prism_gz(prism::GravityPrism, ox::Real, oy::Real, oz::Real,
                  density::Real=1.0)::Float64
    ox64 = Float64(ox)
    oy64 = Float64(oy)
    oz64 = Float64(oz)
    x1 = prism.x1 - ox64
    x2 = prism.x2 - ox64
    y1 = prism.y1 - oy64
    y2 = prism.y2 - oy64
    z1 = oz64 - prism.z1
    z2 = oz64 - prism.z2
    return G_MGAL * Float64(density) * prism_gz_kernel(x1, x2, y1, y2, z1, z2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Grid forward operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    _check_forward_args(density_grid, grid_spec, obs_x, obs_y, obs_z, g_pred)

Type-stable shape / length checks. No allocations.
"""
function _check_forward_args(density_grid::Array{Float32,3},
                             grid_spec::GridSpec,
                             obs_x::Vector{Float32},
                             obs_y::Vector{Float32},
                             obs_z::Vector{Float32},
                             g_pred::Vector{Float32})
    nxx, nyy, nzz = nxyz(grid_spec)
    size(density_grid, 1) == nxx && size(density_grid, 2) == nyy &&
        size(density_grid, 3) == nzz ||
        error("density_grid size $(size(density_grid)) ≠ GridSpec $(nxyz(grid_spec))")
    nobs = length(obs_x)
    length(obs_y) == nobs && length(obs_z) == nobs ||
        error("obs_x / obs_y / obs_z lengths differ: $(length(obs_x)), $(length(obs_y)), $(length(obs_z))")
    length(g_pred) == nobs ||
        error("g_pred length $(length(g_pred)) ≠ nobs $(nobs)")
    return nobs, nxx, nyy, nzz
end

"""
    _accumulate_prism_gz(...)

Internal helper: sum Nagy prism contributions for one station (no allocations).
"""
@inline function _accumulate_prism_gz(density_grid::Array{Float32,3},
                                      xmin::Float64, ymin::Float64, zmin::Float64,
                                      dx::Float64, dy::Float64, dz::Float64,
                                      ox::Float64, oy::Float64, oz::Float64,
                                      nxx::Int, nyy::Int, nzz::Int)::Float64
    acc = 0.0
    eps = 1.0e-30
    # Column-major: `i` (X) is the fastest index for cache-friendly access.
    # Nagy corner kernel is expanded inline so `@turbo` can SIMD-vectorise.
    @turbo for k in 1:nzz, j in 1:nyy, i in 1:nxx
        ρ = Float64(density_grid[i, j, k])
        ρ = ifelse(isfinite(ρ), ρ, 0.0)

        x1 = xmin + (Float64(i) - 1.0) * dx - ox
        x2 = x1 + dx
        y1 = ymin + (Float64(j) - 1.0) * dy - oy
        y2 = y1 + dy
        zw1 = zmin + (Float64(k) - 1.0) * dz
        z1 = oz - zw1
        z2 = oz - (zw1 + dz)

        r111 = sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        r111 = max(r111, eps)
        k111 = x1 * log(max(y1 + r111, eps)) + y1 * log(max(x1 + r111, eps)) - z1 * atan(x1 * y1, z1 * r111)

        r211 = sqrt(x2 * x2 + y1 * y1 + z1 * z1)
        r211 = max(r211, eps)
        k211 = x2 * log(max(y1 + r211, eps)) + y1 * log(max(x2 + r211, eps)) - z1 * atan(x2 * y1, z1 * r211)

        r121 = sqrt(x1 * x1 + y2 * y2 + z1 * z1)
        r121 = max(r121, eps)
        k121 = x1 * log(max(y2 + r121, eps)) + y2 * log(max(x1 + r121, eps)) - z1 * atan(x1 * y2, z1 * r121)

        r221 = sqrt(x2 * x2 + y2 * y2 + z1 * z1)
        r221 = max(r221, eps)
        k221 = x2 * log(max(y2 + r221, eps)) + y2 * log(max(x2 + r221, eps)) - z1 * atan(x2 * y2, z1 * r221)

        r112 = sqrt(x1 * x1 + y1 * y1 + z2 * z2)
        r112 = max(r112, eps)
        k112 = x1 * log(max(y1 + r112, eps)) + y1 * log(max(x1 + r112, eps)) - z2 * atan(x1 * y1, z2 * r112)

        r212 = sqrt(x2 * x2 + y1 * y1 + z2 * z2)
        r212 = max(r212, eps)
        k212 = x2 * log(max(y1 + r212, eps)) + y1 * log(max(x2 + r212, eps)) - z2 * atan(x2 * y1, z2 * r212)

        r122 = sqrt(x1 * x1 + y2 * y2 + z2 * z2)
        r122 = max(r122, eps)
        k122 = x1 * log(max(y2 + r122, eps)) + y2 * log(max(x1 + r122, eps)) - z2 * atan(x1 * y2, z2 * r122)

        r222 = sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        r222 = max(r222, eps)
        k222 = x2 * log(max(y2 + r222, eps)) + y2 * log(max(x2 + r222, eps)) - z2 * atan(x2 * y2, z2 * r222)

        acc += ρ * (-k111 + k211 + k121 - k221 + k112 - k212 - k122 + k222)
    end
    return acc
end

"""
    forward_gravity!(g_pred, density_grid, grid_spec, obs_x, obs_y, obs_z) -> g_pred

In-place Nagy–Okabe prism forward model. Zero allocations on the voxel
loop (`@inbounds`, `@views`, `@turbo`).

# Arguments
- `g_pred::Vector{Float32}`: output, length `nobs`, filled with mGal
- `density_grid::Array{Float32,3}`: `(X, Y, Z)` density, ``g/cm^3``
  (non-finite cells contribute 0)
- `grid_spec::GridSpec`: voxel geometry (m, `z` up)
- `obs_x, obs_y, obs_z::Vector{Float32}`: station coordinates (m)

# Returns
`g_pred`, the vertical gravity anomaly (mGal, positive downward).
"""
@views function forward_gravity!(g_pred::Vector{Float32},
                                 density_grid::Array{Float32,3},
                                 grid_spec::GridSpec,
                                 obs_x::Vector{Float32},
                                 obs_y::Vector{Float32},
                                 obs_z::Vector{Float32})::Vector{Float32}
    nobs, nxx, nyy, nzz = _check_forward_args(
        density_grid, grid_spec, obs_x, obs_y, obs_z, g_pred)

    xmin = Float64(grid_spec.xmin)
    ymin = Float64(grid_spec.ymin)
    zmin = Float64(grid_spec.zmin)
    dx   = Float64(grid_spec.dx)
    dy   = Float64(grid_spec.dy)
    dz   = Float64(grid_spec.dz)
    gmin = G_MGAL

    @inbounds @threads for p in 1:nobs
        ox = Float64(obs_x[p])
        oy = Float64(obs_y[p])
        oz = Float64(obs_z[p])
        acc = _accumulate_prism_gz(
            density_grid, xmin, ymin, zmin, dx, dy, dz,
            ox, oy, oz, nxx, nyy, nzz)
        g_pred[p] = Float32(gmin * acc)
    end
    return g_pred
end

"""
    forward_gravity(density_grid, grid_spec, obs_x, obs_y, obs_z) -> Vector{Float32}

Predicted gravity anomaly ``g_{pred}`` (mGal) at each observation point.

# Arguments
- `density_grid::Array{Float32,3}`: `(X, Y, Z)` density model, ``g/cm^3``
- `grid_spec::GridSpec`: regular voxel grid (m, `z` positive upward)
- `obs_x, obs_y, obs_z::Vector{Float32}`: station easting, northing, RL (m)

# Returns
`Vector{Float32}` of length `length(obs_x)`, mGal, positive downward.

See [`forward_gravity!`](@ref) for the zero-allocation mutating form.
"""
function forward_gravity(density_grid::Array{Float32,3},
                         grid_spec::GridSpec,
                         obs_x::Vector{Float32},
                         obs_y::Vector{Float32},
                         obs_z::Vector{Float32})::Vector{Float32}
    g_pred = Vector{Float32}(undef, length(obs_x))
    return forward_gravity!(g_pred, density_grid, grid_spec, obs_x, obs_y, obs_z)
end

# ─────────────────────────────────────────────────────────────────────────────
# Linear operator  g = G * vec(m)  (Nagy kernel is linear in density)
# ─────────────────────────────────────────────────────────────────────────────

"""
    gravity_kernel_matrix(grid_spec, obs_x, obs_y, obs_z) -> Matrix{Float32}

Assemble the dense sensitivity matrix ``G`` of the Nagy prism operator:

```
g_pred = G * vec(m)     # column-major vec, layout (X, Y, Z)
```

# Size
`(n_obs, n_voxels)` with `n_voxels = nx·ny·nz`.

# Units
mGal per ``g/cm^3``. Precompute once per geometry; each later forward /
adjoint is a BLAS matvec (not a prism loop).
"""
function gravity_kernel_matrix(grid_spec::GridSpec,
                               obs_x::Vector{Float32},
                               obs_y::Vector{Float32},
                               obs_z::Vector{Float32})::Matrix{Float32}
    nobs = length(obs_x)
    length(obs_y) == nobs && length(obs_z) == nobs ||
        error("obs_x / obs_y / obs_z lengths differ: $(length(obs_x)), $(length(obs_y)), $(length(obs_z))")

    nxx, nyy, nzz = nxyz(grid_spec)
    nvox = nxx * nyy * nzz
    Gmat = Matrix{Float32}(undef, nobs, nvox)

    xmin = Float64(grid_spec.xmin)
    ymin = Float64(grid_spec.ymin)
    zmin = Float64(grid_spec.zmin)
    dx   = Float64(grid_spec.dx)
    dy   = Float64(grid_spec.dy)
    dz   = Float64(grid_spec.dz)
    gmin = G_MGAL

    # One row per station. Voxel index `t` matches Julia `vec` (i fastest).
    @inbounds @threads for p in 1:nobs
        ox = Float64(obs_x[p])
        oy = Float64(obs_y[p])
        oz = Float64(obs_z[p])
        t = 0
        for k in 1:nzz, j in 1:nyy, i in 1:nxx
            t += 1
            x1 = xmin + (Float64(i) - 1.0) * dx - ox
            x2 = x1 + dx
            y1 = ymin + (Float64(j) - 1.0) * dy - oy
            y2 = y1 + dy
            zw1 = zmin + (Float64(k) - 1.0) * dz
            z1 = oz - zw1
            z2 = oz - (zw1 + dz)
            Gmat[p, t] = Float32(gmin * prism_gz_kernel(x1, x2, y1, y2, z1, z2))
        end
    end
    return Gmat
end

"""
    apply_gravity_kernel(G, density_grid) -> Vector{Float32}

``g_{pred} = G \\, \\mathrm{vec}(m)`` in mGal. `G` from
[`gravity_kernel_matrix`](@ref); `density_grid` in ``g/cm^3``.
"""
function apply_gravity_kernel(G::AbstractMatrix{<:Real},
                              density_grid::Array{Float32,3})::Vector{Float32}
    nvox = length(density_grid)
    size(G, 2) == nvox ||
        error("kernel columns $(size(G, 2)) ≠ n_voxels $nvox")
    return Float32.(G * vec(density_grid))
end

end # module ForwardGravity
