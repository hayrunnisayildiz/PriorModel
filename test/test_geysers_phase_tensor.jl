# Verifies scripts/geysers_dimensionality.jl against Caldwell, Bibby & Brown
# (2004), "The magnetotelluric phase tensor", Geophys. J. Int. 158, 457-469
# (doi:10.1111/j.1365-246X.2004.02281.x). Every @testset name cites the
# equation it reproduces. The analytic 1-D / 2-D / rotated cases are taken
# straight from the paper, so a sign or branch error in the implementation
# fails here rather than silently biasing the strike in results/.
#
# Not part of test/runtests.jl (that file lists its includes explicitly).
# Run standalone:
#     julia --project=. test/test_geysers_phase_tensor.jl

using Test
using LinearAlgebra
using Statistics

get!(ENV, "GKSwstype", "nul")

if !isdefined(Main, :pt_invariants)
    include(joinpath(@__DIR__, "..", "scripts", "geysers_dimensionality.jl"))
end

"""CBB04 eq. (21): R(θ) = [cosθ sinθ; −sinθ cosθ], θ clockwise from x₁."""
cbb_rot(θ_deg::Real) = [cosd(θ_deg) sind(θ_deg); -sind(θ_deg) cosd(θ_deg)]

"""
Meridian arc from the equator to `φ`, by direct numerical quadrature of
`M(φ) = ∫₀^φ a(1−e²)(1−e²sin²t)^{-3/2} dt`. Independent of the truncated
series (Snyder eq. 3-21) used inside `wgs84_to_utm10n`, so it actually tests
those coefficients. Composite Simpson with `n` intervals; at n = 20 000 the
quadrature error is far below a micrometre.
"""
function meridian_arc_quadrature(lat_deg::Real; n::Int = 20_000)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    g(t) = a * (1 - e2) / (1 - e2 * sin(t)^2)^1.5
    φ = deg2rad(lat_deg)
    h = φ / n
    s = g(0.0) + g(φ)
    for k in 1:(n - 1)
        s += (isodd(k) ? 4 : 2) * g(k * h)
    end
    return s * h / 3
end

const GEYSERS_EDI_DIR = joinpath(@__DIR__, "..", "data", "geysers", "mt", "edi")

@testset "Geysers phase tensor (Caldwell, Bibby & Brown 2004)" begin

    @testset "Φ = X⁻¹Y — eqs. (13) and (15)" begin
        Z = ComplexF64[1+2im 3-1im; -0.5+0.7im 2+4im]
        X, Y = real.(Z), imag.(Z)
        Φ = phase_tensor(Z)

        @test Φ ≈ inv(X) * Y                                    # eq. (13)
        @test Φ ≈ [X[2,2]*Y[1,1]-X[1,2]*Y[2,1]  X[2,2]*Y[1,2]-X[1,2]*Y[2,2];
                   X[1,1]*Y[2,1]-X[2,1]*Y[1,1]  X[1,1]*Y[2,2]-X[2,1]*Y[1,2]] / det(X)

        # eq. (14): a real galvanic distortion tensor D leaves Φ unchanged.
        for D in ([1.7 0.4; -0.3 0.9], [0.2 0.0; 0.0 5.0], [1.0 -2.5; 3.1 0.4])
            @test phase_tensor(D * Z) ≈ Φ
        end

        @test all(isnan, phase_tensor(ComplexF64[NaN+0im 1; 1 1]))
        @test all(isnan, phase_tensor(ComplexF64[1+1im 2+1im; 2+3im 4+1im]))  # det X = 0
    end

    @testset "1-D: Φ = tan(φ)·I — eq. (18)" begin
        for φ in (20.0, 45.0, 70.0)
            z0 = 10 * cis(deg2rad(φ))
            v = pt_invariants(ComplexF64[0 z0; -z0 0])
            @test phase_tensor(ComplexF64[0 z0; -z0 0]) ≈ tand(φ) * I(2)
            @test v.beta ≈ 0 atol = 1e-10
            @test v.phimax ≈ tand(φ)
            @test v.phimin ≈ tand(φ)
            @test v.phimax_deg ≈ φ
            @test v.phimin_deg ≈ φ
            @test v.ellipticity ≈ 0 atol = 1e-10
        end
    end

    @testset "2-D in strike coordinates: Φ diagonal, β = 0 — eq. (26)" begin
        # Z_R = [0 Z∥; −Z⊥ 0] (eq. 24). Φ = diag(Y⊥/X⊥, Y∥/X∥) = diag(tan30°, tan55°).
        z_par = 40 * cis(deg2rad(55))
        z_perp = 12 * cis(deg2rad(30))
        Z = ComplexF64[0 z_par; -z_perp 0]
        Φ = phase_tensor(Z)

        @test Φ[1, 2] ≈ 0 atol = 1e-12
        @test Φ[2, 1] ≈ 0 atol = 1e-12
        @test Φ[1, 1] ≈ tand(30)
        @test Φ[2, 2] ≈ tand(55)

        v = pt_invariants(Φ)
        @test v.beta ≈ 0 atol = 1e-10
        @test v.phimax ≈ tand(55)
        @test v.phimin ≈ tand(30)
        @test v.phimax_deg ≈ 55
        @test v.phimin_deg ≈ 30
        # Φ11 < Φ22, so α = 90°: the major axis lies along x₂, i.e. perpendicular
        # to the strike axis x₁. This is the 90° ambiguity CBB04 describe.
        @test mod(v.azimuth, 180) ≈ 90 atol = 1e-8
    end

    @testset "2-D rotated to an arbitrary strike: azimuth tracks strike" begin
        Z_strike = ComplexF64[0 40*cis(deg2rad(55)); -12*cis(deg2rad(30)) 0]
        for θ in (0.0, 17.0, 40.0, 115.0, 168.0)
            # Observation frame: strike sits at azimuth θ, so Z_strike = R(θ) Z Rᵀ(θ).
            Z_obs = transpose(cbb_rot(θ)) * Z_strike * cbb_rot(θ)
            v = pt_invariants(Z_obs)
            @test v.beta ≈ 0 atol = 1e-10
            @test v.phimax ≈ tand(55)
            @test v.phimin ≈ tand(30)
            @test mod(v.azimuth, 180) ≈ mod(90 + θ, 180) atol = 1e-8
        end
    end

    @testset "Coordinate invariance and α′ = α − θ — eq. (23)" begin
        for Z in (ComplexF64[1+2im 3-1im; -0.5+0.7im 2+4im],
                  ComplexF64[3-1im 55+40im; -30-25im -2+6im],
                  ComplexF64[-0.4+0.9im 12+7im; -9-11im 0.8-0.2im])
            base = pt_invariants(Z)
            for θ in (13.0, 47.0, -80.0, 200.0)
                v = pt_invariants(cbb_rot(θ) * Z * transpose(cbb_rot(θ)))
                @test v.beta ≈ base.beta atol = 1e-10
                @test v.phimax ≈ base.phimax
                @test v.phimin ≈ base.phimin
                @test v.ellipticity ≈ base.ellipticity
                @test mod(v.azimuth, 180) ≈ mod(base.azimuth - θ, 180) atol = 1e-8
            end
        end
    end

    @testset "Singular-value decomposition — eq. (20)" begin
        for Z in (ComplexF64[1+2im 3-1im; -0.5+0.7im 2+4im],
                  ComplexF64[0 40*cis(deg2rad(55)); -12*cis(deg2rad(30)) 0],
                  ComplexF64[3-1im 55+40im; -30-25im -2+6im])
            Φ = phase_tensor(Z)
            v = pt_invariants(Φ)
            recomposed = transpose(cbb_rot(v.alpha - v.beta)) *
                         Diagonal([v.phimax, v.phimin]) *
                         cbb_rot(v.alpha + v.beta)
            @test recomposed ≈ Φ atol = 1e-12
        end
    end

    @testset "Coordinate invariants — Appendix eqs. (A2)–(A10)" begin
        for Z in (ComplexF64[1+2im 3-1im; -0.5+0.7im 2+4im],
                  ComplexF64[3-1im 55+40im; -30-25im -2+6im])
            Φ = phase_tensor(Z)
            v = pt_invariants(Φ)
            Φ1 = tr(Φ) / 2                       # (A2), (A5)
            Φ3 = (Φ[1, 2] - Φ[2, 1]) / 2         # (A3), (A7)
            detΦ = det(Φ)                        # (A4)

            @test v.phimax * v.phimin ≈ detΦ
            @test v.phimax + v.phimin ≈ 2 * sqrt(Φ1^2 + Φ3^2)
            @test v.beta ≈ rad2deg(0.5 * atan(Φ3 / Φ1))   # (A10), valid for tr Φ > 0

            if detΦ >= 0
                Φ2 = sqrt(detΦ)                  # (A6)
                root = sqrt(Φ1^2 + Φ3^2 - Φ2^2)
                @test v.phimax ≈ sqrt(Φ1^2 + Φ3^2) + root    # (A9)
                @test v.phimin ≈ sqrt(Φ1^2 + Φ3^2) - root    # (A8)
            end
        end
    end

    @testset "det(Φ) < 0 gives Φmin < 0 — Appendix" begin
        Φ = [0.5 0.9; 0.9 0.5]                   # det = −0.56
        v = pt_invariants(Φ)
        @test det(Φ) < 0
        @test v.phimin < 0
        @test v.phimax * v.phimin ≈ det(Φ)
        @test v.phimin_deg < 0
    end

    @testset "NaN in, NaN out" begin
        v = pt_invariants(fill(NaN, 2, 2))
        @test isnan(v.beta) && isnan(v.alpha) && isnan(v.azimuth)
        @test isnan(v.phimax) && isnan(v.phimin)
    end
end

@testset "Circular statistics" begin
    @test all(isnan, axial_mean(Float64[], 180.0))

    μ, R, σ = axial_mean([37.0, 37.0, 37.0], 180.0)
    @test μ ≈ 37
    @test R ≈ 1
    @test σ ≈ 0 atol = 1e-12

    # A line direction is 180°-periodic: 190° is the same axis as 10°.
    μ, R, _ = axial_mean([10.0, 190.0], 180.0)
    @test μ ≈ 10 atol = 1e-9
    @test R ≈ 1

    # A geoelectric strike is 90°-periodic on top of that.
    μ, R, _ = axial_mean([10.0, 100.0, 190.0, 280.0], 90.0)
    @test μ ≈ 10 atol = 1e-9
    @test R ≈ 1

    # Wrap-around through 0°.
    μ, _, _ = axial_mean([175.0, 5.0], 180.0)
    @test μ ≈ 0 atol = 1e-9

    # Perfectly opposed axes cancel: no preferred direction.
    _, R, _ = axial_mean([0.0, 45.0, 90.0, 135.0], 180.0)
    @test R ≈ 0 atol = 1e-12

    @test axial_mean([0.0, 40.0], 180.0)[1] ≈ 20 atol = 1e-9
end

@testset "Period clustering" begin
    @test cluster_periods([1.0, 1.005, 2.0, 2.01]) ≈ [sqrt(1.0 * 1.005), sqrt(2.0 * 2.01)]
    @test length(cluster_periods([1.0, 1.5, 2.25])) == 3
    @test isempty(cluster_periods([-1.0, 0.0, NaN]))
    grid = cluster_periods([0.01, 0.1, 1.0, 10.0])
    @test nearest_period_index(grid, 0.09) == 2
    @test nearest_period_index(grid, 1e6) == 4
end

@testset "WGS84 → UTM Zone 10N" begin
    # On the central meridian the easting is the false easting, exactly.
    for lat in (0.0, 38.8, 60.0)
        e, _ = wgs84_to_utm10n(lat, -123.0)
        @test e ≈ 500_000.0 atol = 1e-6
    end
    @test wgs84_to_utm10n(0.0, -123.0)[2] ≈ 0.0 atol = 1e-6

    # Northing on the central meridian is k0 · M(φ). Compare the truncated
    # Snyder series against direct quadrature of the meridian-arc integral.
    for lat in (10.0, 38.8228, 55.0)
        _, n = wgs84_to_utm10n(lat, -123.0)
        @test n ≈ 0.9996 * meridian_arc_quadrature(lat) atol = 1e-3
    end

    # Northwest Geysers, gz01. Zone 10N covers 126°W–120°W, so the easting must
    # sit east of the central meridian by roughly 0.228° × 86.7 km/° ≈ 19.8 km.
    e, n = wgs84_to_utm10n(38.822778, -122.771944)
    @test 518_000 < e < 521_000
    @test 4_296_000 < n < 4_300_000
end

@testset "EDI parser" begin
    if !isdir(GEYSERS_EDI_DIR) || isempty(filter(endswith(".edi"), readdir(GEYSERS_EDI_DIR)))
        @info "Geysers EDIs not present; skipping parser tests" dir = GEYSERS_EDI_DIR
    else
        st = parse_edi(joinpath(GEYSERS_EDI_DIR, "gz01.edi"))

        @test st.site == "GZ01"
        @test st.lat ≈ 38 + 49 / 60 + 22 / 3600            # LAT=38:49:22.00
        @test st.lon ≈ -(122 + 46 / 60 + 19 / 3600)        # LON=-122:46:19.00
        @test st.elev ≈ 2113.2
        @test st.declination ≈ 13.880
        @test length(st.periods) == 40
        @test issorted(st.periods)
        @test st.periods[1] ≈ 1 / 7.679902e+02             # highest FREQ
        @test st.periods[end] ≈ 1 / 9.766579e-04           # lowest FREQ
        @test all(≈(346.12), st.zrot)

        # Row 1 of every impedance block, verbatim from gz01.edi.
        Z = st.Z[1]
        @test Z[1, 1] ≈ complex(-8.443681e+01, -1.012529e+02)
        @test Z[1, 2] ≈ complex(4.668080e+02, 5.462381e+02)
        @test Z[2, 1] ≈ complex(-1.237429e+02, -1.504845e+02)
        @test Z[2, 2] ≈ complex(-5.504812e+00, 5.667029e+01)
        @test st.Zvar[1] ≈ [2.192539e+00 3.024178e+01; 6.728379e+00 1.740238e+01]

        # Whole survey: 42 stations, full tensor everywhere, one declination.
        files = sort(filter(endswith(".edi"), readdir(GEYSERS_EDI_DIR)))
        stations = [parse_edi(joinpath(GEYSERS_EDI_DIR, f)) for f in files]
        @test length(stations) == 42
        @test length(unique(s.site for s in stations)) == 42
        @test all(s -> all(≈(346.12), s.zrot), stations)
        @test all(s -> 38.7 < s.lat < 39.0 && -123.0 < s.lon < -122.6, stations)

        # `EMPTY=1e+32` must become NaN, never leak through as a huge number.
        # A handful of pairs really are blank (gz03, gz21), so require most of
        # them to be complete rather than all of them.
        allZ = vcat([s.Z for s in stations]...)
        @test !any(z -> any(c -> abs(c) > 1e29, z), allZ)
        @test any(z -> any(isnan, z), allZ)
        @test count(z -> all(isfinite, z), allZ) / length(allZ) > 0.95

        Ts = vcat([s.periods for s in stations]...)
        @test minimum(Ts) > 1.29e-3 && minimum(Ts) < 1.31e-3
        @test maximum(Ts) > 1020 && maximum(Ts) < 1030
    end
end

@testset "Monte-Carlo β uncertainty" begin
    rng = MersenneTwister(1234)
    Z = ComplexF64[3-1im 55+40im; -30-25im -2+6im]
    @test isnan(beta_uncertainty(Z, fill(NaN, 2, 2), 100, rng))
    @test isnan(beta_uncertainty(Z, [-1.0 1.0; 1.0 1.0], 100, rng))

    # Zero variance ⇒ zero spread, up to the floating-point noise in R → 1.
    @test beta_uncertainty(Z, zeros(2, 2), 200, rng) ≈ 0 atol = 1e-4
    small = beta_uncertainty(Z, fill(0.25, 2, 2), 4000, MersenneTwister(7))
    large = beta_uncertainty(Z, fill(25.0, 2, 2), 4000, MersenneTwister(7))
    @test 0 < small < large
end
