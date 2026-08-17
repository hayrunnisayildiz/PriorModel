"""
    ForwardMagnetic

Analytic 3-D prism total-field magnetic forward operator for **induced**
magnetization in a vertical regional field (Nagy–Okabe geometry; Blakely 1996,
§6.2).

Maps a voxel susceptibility model ``\\chi`` (SI, dimensionless) onto predicted
TMI anomalies ``\\Delta T`` (nT) at surface observation points.  The geometric
kernel matches [`ForwardGravity.prism_gz_kernel`](@ref); the physical scale is
``(\\mu_0/4\\pi)\\,H_0`` with ``H_0 = B_0/\\mu_0``.

Each forward / adjoint step is a BLAS matvec via a precomputed sensitivity
matrix (Zygote-friendly).

# Units
- susceptibility: SI (dimensionless)
- coordinates: metres, `z` up (RL)
- TMI anomaly: nT
"""
module ForwardMagnetic

using LoopVectorization
using Base.Threads: @threads

if !isdefined(@__MODULE__, :ForwardGravity)
    include(joinpath(@__DIR__, "ForwardGravity.jl"))
end
using .ForwardGravity
using .ForwardGravity.GridSpecs
using .ForwardGravity: prism_gz_kernel, _check_forward_args

export forward_magnetic, forward_magnetic!
export magnetic_kernel_matrix, apply_magnetic_kernel
export MAG_NT_PER_SI, DEFAULT_B0_NT

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

"""Default total-field intensity at Keivitsa latitude (nT)."""
const DEFAULT_B0_NT::Float64 = 50_000.0

"""``\\mu_0/(4\\pi)`` in SI."""
const MU0_4PI::Float64 = 1.0e-7

"""
    MAG_NT_PER_SI

Scale from (SI susceptibility) × (Nagy kernel, metres) to nT:

``\\Delta T = \\mathrm{MAG\\_NT\\_PER\\_SI} \\cdot \\chi \\cdot K_\\mathrm{geom}``

with ``H_0 = B_0/\\mu_0`` and ``B_0 =`` [`DEFAULT_B0_NT`](@ref) nT.
"""
const H0_A_M::Float64 = (DEFAULT_B0_NT * 1.0e-9) / (4π * MU0_4PI)
const MAG_NT_PER_SI::Float64 = MU0_4PI * H0_A_M * 1.0e9

# ─────────────────────────────────────────────────────────────────────────────
# Grid forward operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    forward_magnetic!(t_pred, chi_grid, grid_spec, obs_x, obs_y, obs_z) -> t_pred

In-place prism TMI forward model (nT).  Non-finite susceptibility cells
contribute zero.
"""
@views function forward_magnetic!(t_pred::Vector{Float32},
                                  chi_grid::Array{Float32,3},
                                  grid_spec::GridSpec,
                                  obs_x::Vector{Float32},
                                  obs_y::Vector{Float32},
                                  obs_z::Vector{Float32})::Vector{Float32}
    nobs, nxx, nyy, nzz = _check_forward_args(
        chi_grid, grid_spec, obs_x, obs_y, obs_z, t_pred)

    xmin = Float64(grid_spec.xmin)
    ymin = Float64(grid_spec.ymin)
    zmin = Float64(grid_spec.zmin)
    dx   = Float64(grid_spec.dx)
    dy   = Float64(grid_spec.dy)
    dz   = Float64(grid_spec.dz)
    scale = MAG_NT_PER_SI

    @inbounds @threads for p in 1:nobs
        ox = Float64(obs_x[p])
        oy = Float64(obs_y[p])
        oz = Float64(obs_z[p])
        acc = 0.0
        @turbo for k in 1:nzz, j in 1:nyy, i in 1:nxx
            χ = Float64(chi_grid[i, j, k])
            χ = ifelse(isfinite(χ), χ, 0.0)
            x1 = xmin + (Float64(i) - 1.0) * dx - ox
            x2 = x1 + dx
            y1 = ymin + (Float64(j) - 1.0) * dy - oy
            y2 = y1 + dy
            zw1 = zmin + (Float64(k) - 1.0) * dz
            z1 = oz - zw1
            z2 = oz - (zw1 + dz)
            acc += χ * prism_gz_kernel(x1, x2, y1, y2, z1, z2)
        end
        t_pred[p] = Float32(scale * acc)
    end
    return t_pred
end

"""
    forward_magnetic(chi_grid, grid_spec, obs_x, obs_y, obs_z) -> Vector{Float32}

Predicted TMI anomaly (nT) at each observation point.
"""
function forward_magnetic(chi_grid::Array{Float32,3},
                          grid_spec::GridSpec,
                          obs_x::Vector{Float32},
                          obs_y::Vector{Float32},
                          obs_z::Vector{Float32})::Vector{Float32}
    t_pred = Vector{Float32}(undef, length(obs_x))
    return forward_magnetic!(t_pred, chi_grid, grid_spec, obs_x, obs_y, obs_z)
end

"""
    magnetic_kernel_matrix(grid_spec, obs_x, obs_y, obs_z) -> Matrix{Float32}

Dense sensitivity matrix ``M`` with ``\\Delta T = M \\, \\mathrm{vec}(\\chi)`` (nT).
"""
function magnetic_kernel_matrix(grid_spec::GridSpec,
                                obs_x::Vector{Float32},
                                obs_y::Vector{Float32},
                                obs_z::Vector{Float32})::Matrix{Float32}
    nobs = length(obs_x)
    length(obs_y) == nobs && length(obs_z) == nobs ||
        error("obs_x / obs_y / obs_z lengths differ")

    nxx, nyy, nzz = nxyz(grid_spec)
    nvox = nxx * nyy * nzz
    Mmat = Matrix{Float32}(undef, nobs, nvox)

    xmin = Float64(grid_spec.xmin)
    ymin = Float64(grid_spec.ymin)
    zmin = Float64(grid_spec.zmin)
    dx   = Float64(grid_spec.dx)
    dy   = Float64(grid_spec.dy)
    dz   = Float64(grid_spec.dz)
    scale = MAG_NT_PER_SI

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
            Mmat[p, t] = Float32(scale * prism_gz_kernel(x1, x2, y1, y2, z1, z2))
        end
    end
    return Mmat
end

"""
    apply_magnetic_kernel(M, chi_grid) -> Vector{Float32}

``\\Delta T = M \\, \\mathrm{vec}(\\chi)`` in nT.
"""
function apply_magnetic_kernel(M::AbstractMatrix{<:Real},
                               chi_grid::Array{Float32,3})::Vector{Float32}
    nvox = length(chi_grid)
    size(M, 2) == nvox ||
        error("kernel columns $(size(M, 2)) ≠ n_voxels $nvox")
    return Float32.(M * vec(chi_grid))
end

end # module ForwardMagnetic
