# COMMEMI 11×7 → UNET 30×20 resample + kanonik round-trip.
# Bu geçiş kırılırsa model DimensionMismatch verir veya T_ood clamp ile yanlış prior üretir.

if !isdefined(Main, :MTInputStandardizer)
    include(joinpath(@__DIR__, "..", "src", "synthetic", "MTInputStandardizer.jl"))
end
using .MTInputStandardizer
using MTGeophysics
using Statistics: mean

# Nested copy from the standardizer include — do not `using` Main.MTMeshParams
# here; that is a different type after the MeshParams test's include.
const _MP = MTInputStandardizer.MTMeshParams

const COMMEMI_OBS = joinpath(@__DIR__, "..", "examples", "0COMEMI2D-I", "Comemi2D1.obs")

# COMMEMI receivers are -8000:1600:8000 m; this duck-typed survey makes
# `station_positions(nx, dx, n_stations)` reproduce those x-coordinates.
_commemi_query_mesh(periods) = (nx=11, dx=1600.0, n_stations=11, periods=Float64.(collect(periods)))

function _period_ood(src_T, tgt_T)
    smin, smax = extrema(Float64.(src_T))
    tmin, tmax = extrema(Float64.(tgt_T))
    return smin < tmin - 1e-12 || smax > tmax + 1e-12
end

function _n_stations_ood(src_x, tgt_x)
    tmin, tmax = extrema(Float64.(tgt_x))
    return count(x -> x < tmin - 1e-12 || x > tmax + 1e-12, Float64.(src_x))
end

@testset "MTInputStandardizer round-trip" begin
    @testset "CANONICAL_* stays locked to UNET_MESH" begin
        @test CANONICAL_MESH === _MP.UNET_MESH
        @test CANONICAL_N_STATIONS == _MP.UNET_MESH.n_stations == 30
        @test length(CANONICAL_PERIODS) == _MP.n_periods(_MP.UNET_MESH) == 20
        @test CANONICAL_PERIODS == _MP.UNET_MESH.periods
        @test canonical_station_positions() == _MP.station_positions(_MP.UNET_MESH)
        @test first(CANONICAL_PERIODS) ≈ 1e-3
        @test last(CANONICAL_PERIODS) ≈ 1e3
    end

    @testset "canonical → canonical identity (lossless round-trip)" begin
        xs = _MP.station_positions(_MP.UNET_MESH)
        Ts = _MP.UNET_MESH.periods
        synth = Array{Float32,3}(undef, 30, 20, 2)
        for is in eachindex(xs), ip in eachindex(Ts)
            synth[is, ip, 1] = Float32(1e-4 * xs[is] + log10(Ts[ip]))
            synth[is, ip, 2] = Float32(45.0 + 1e-5 * xs[is])
        end
        restored = standardize_mt_input(xs, Ts, synth; mp=_MP.UNET_MESH, method=:bilinear)
        @test size(restored) == (30, 20, 2)
        @test restored ≈ synth atol=2.0f-5 rtol=2.0f-5
    end

    @testset "COMMEMI 11×7 → 30×20 and reverse onto the same survey" begin
        @test isfile(COMMEMI_OBS)
        data = MTGeophysics.load_data2d(COMMEMI_OBS)
        raw = pack_te_response(data.rho_xy, data.phase_xy)

        @test size(raw) == (_MP.DEFAULT_MESH.n_stations, _MP.n_periods(_MP.DEFAULT_MESH), 2)
        @test size(raw, 1) == length(data.receivers) == 11
        @test size(raw, 2) == length(data.periods) == 7
        @test all(isfinite, raw)

        tgt_x = _MP.station_positions(_MP.UNET_MESH)
        @test _period_ood(data.periods, CANONICAL_PERIODS) == false
        @test _n_stations_ood(data.receivers, tgt_x) == 0

        x_bi = standardize_mt_input(data; method=:bilinear)
        x_nn = standardize_mt_input(data; method=:nearest)
        @test size(x_bi) == (CANONICAL_N_STATIONS, length(CANONICAL_PERIODS), 2)
        @test size(x_nn) == size(x_bi)
        @test all(isfinite, x_bi)
        @test all(isfinite, x_nn)

        # Linear field in (x, log10 T): bilinear upsample then downsample is lossless
        # interior to COMMEMI, so a reverse query must recover the source tensor.
        src_x = Float64.(collect(data.receivers))
        src_T = Float64.(collect(data.periods))
        linear = Array{Float32,3}(undef, 11, 7, 2)
        for is in 1:11, ip in 1:7
            linear[is, ip, 1] = Float32(1e-4 * src_x[is] + 0.25 * log10(src_T[ip]))
            linear[is, ip, 2] = Float32(30.0 + 1e-4 * src_x[is])
        end
        up = standardize_mt_input(src_x, src_T, linear; mp=_MP.UNET_MESH, method=:bilinear)
        @test size(up) == (30, 20, 2)

        qmesh = _commemi_query_mesh(src_T)
        @test _MP.station_positions(qmesh.nx, qmesh.dx, qmesh.n_stations) ≈ src_x
        down = standardize_mt_input(tgt_x, _MP.UNET_MESH.periods, up; mp=qmesh, method=:bilinear)
        @test size(down) == (11, 7, 2)
        @test all(isfinite, down)
        @test down ≈ linear atol=5e-4 rtol=5e-4

        # Real COMMEMI reverse: shape + no NaN; interpolation of a non-linear
        # field is not bit-exact, but RMS must stay well below the data scale.
        down_real = standardize_mt_input(tgt_x, _MP.UNET_MESH.periods, x_bi; mp=qmesh, method=:bilinear)
        @test size(down_real) == size(raw)
        @test all(isfinite, down_real)
        rms = sqrt(mean((down_real .- raw) .^ 2))
        @test rms < 1.0
    end

    @testset "4-channel TE+TM layout is interpolated independently" begin
        # Next architecture experiment stacks TM as channels 3–4; a shape drift
        # here would silently drop TM or feed the U-Net the wrong C_in.
        xs = _MP.station_positions(_MP.DEFAULT_MESH)
        Ts = _MP.DEFAULT_MESH.periods
        tetm = rand(Float32, 11, 7, 4)
        out = standardize_mt_input(xs, Ts, tetm; mp=_MP.UNET_MESH, method=:nearest)
        @test size(out) == (30, 20, 4)
        @test all(isfinite, out)
    end

    @testset "pack_tetm_response folds TM phase to [0, 90]" begin
        @test fold_tm_phase_to_0_90(-135.0) ≈ 45.0
        @test fold_tm_phase_to_0_90(45.0) ≈ 45.0
        @test isnan(fold_tm_phase_to_0_90(NaN))

        rho_xy   = Float64[10.0 100.0; 20.0 200.0]
        phase_xy = Float64[45.0  50.0; 40.0  55.0]
        rho_yx   = Float64[30.0 300.0; 40.0 400.0]
        phase_yx = Float64[-140.0 -130.0; -150.0 -120.0]
        te = pack_te_response(rho_xy, phase_xy)
        tetm = pack_tetm_response(rho_xy, phase_xy, rho_yx, phase_yx)
        @test size(tetm) == (2, 2, 4)
        @test tetm[:, :, 1:2] == te
        @test te[2, 1, 2] == Float32(50.0)
        @test all(0 .<= tetm[:, :, 4] .<= 90)
        @test tetm[1, 1, 4] ≈ Float32(fold_tm_phase_to_0_90(-140.0))

        data = MTGeophysics.load_data2d(COMMEMI_OBS)
        @test maximum(data.phase_yx) < 0
        packed = pack_tetm_response(data.rho_xy, data.phase_xy, data.rho_yx, data.phase_yx)
        @test size(packed, 3) == 4
        @test all(0 .<= packed[:, :, 2] .<= 90)
        @test all(0 .<= packed[:, :, 4] .<= 90)
        rms_raw = sqrt(mean(abs2, data.phase_yx .- data.phase_xy))
        rms_fold = sqrt(mean(abs2, packed[:, :, 4] .- packed[:, :, 2]))
        @test rms_raw > 90
        @test rms_fold < 45
    end
end
