# UNET_MESH (240×80 m, 30×20) ve DEFAULT_MESH (COMMEMI solver) sözleşmesini kilitler.
# Bu grid kayarsa v7 ağırlıkları sessizce uyumsuz boyutta yüklenir / COMMEMI 11×7 ağ girdisi sanılır.

if !isdefined(Main, :MTMeshParams)
    include(joinpath(@__DIR__, "..", "src", "synthetic", "MeshParams.jl"))
end
using .MTMeshParams

function _mesh_fields_match(a::MeshParams, b::MeshParams; rtol=1e-12, atol=1e-15)
    return a.nx == b.nx && a.nz == b.nz &&
           isapprox(a.dx, b.dx; rtol=rtol, atol=atol) &&
           isapprox(a.dz, b.dz; rtol=rtol, atol=atol) &&
           a.n_stations == b.n_stations &&
           length(a.periods) == length(b.periods) &&
           all(isapprox.(a.periods, b.periods; rtol=rtol, atol=atol))
end

@testset "MeshParams round-trip" begin
    @testset "UNET_MESH is the 240×80 m / 30×20 network survey" begin
        @test UNET_MESH.nx == 240
        @test UNET_MESH.dx == 80.0
        @test UNET_MESH.nz == 48
        @test UNET_MESH.dz == 25.0
        @test UNET_MESH.n_stations == 30
        @test n_periods(UNET_MESH) == 20
        @test n_periods(UNET_MESH) == UNET_N_PERIODS
        @test n_components(UNET_MESH) == 2
        @test n_components(UNET_MESH; tetm=true) == 4
        @test profile_length(UNET_MESH) == 240 * 80.0
        @test depth_extent(UNET_MESH) == 48 * 25.0
        @test UNET_MESH.periods == unet_log_periods()
        @test first(UNET_MESH.periods) ≈ 1e-3
        @test last(UNET_MESH.periods) ≈ 1e3
        @test all(diff(UNET_MESH.periods) .> 0)  # increasing T (log-spaced)

        xs = station_positions(UNET_MESH)
        @test length(xs) == UNET_MESH.n_stations
        @test first(xs) <= -9000.0
        @test last(xs) >= 9000.0
        @test issorted(xs)
        @test validate_mesh_params(UNET_MESH) === UNET_MESH
    end

    @testset "DEFAULT_MESH is the COMMEMI solver mesh (11×7)" begin
        @test DEFAULT_MESH.nx == 45
        @test DEFAULT_MESH.dx == 400.0
        @test DEFAULT_MESH.nz == 68
        @test DEFAULT_MESH.dz == 200.0
        @test DEFAULT_MESH.n_stations == 11
        @test n_periods(DEFAULT_MESH) == 7
        @test profile_length(DEFAULT_MESH) == 45 * 400.0
        @test depth_extent(DEFAULT_MESH) == 68 * 200.0
        expected_T = [1.0 / f for f in collect(10 .^ range(-2, 2; length=7))]
        @test DEFAULT_MESH.periods ≈ expected_T
        @test frequencies_hz(DEFAULT_MESH) ≈ 10 .^ range(-2, 2; length=7)

        xs = station_positions(DEFAULT_MESH)
        @test length(xs) == DEFAULT_MESH.n_stations
        @test issorted(xs)
        @test validate_mesh_params(DEFAULT_MESH) === DEFAULT_MESH
    end

    @testset "UNET_MESH and DEFAULT_MESH must not silently alias" begin
        @test UNET_MESH.nx != DEFAULT_MESH.nx
        @test UNET_MESH.n_stations != DEFAULT_MESH.n_stations
        @test n_periods(UNET_MESH) != n_periods(DEFAULT_MESH)
        @test UNET_MESH.dx != DEFAULT_MESH.dx
    end

    @testset "JSON / JLD2 serialization round-trip" begin
        mktempdir() do tmp
            for (label, mp) in (("unet", UNET_MESH), ("default", DEFAULT_MESH))
                json_path = joinpath(tmp, "$(label).json")
                jld_path = joinpath(tmp, "$(label).jld2")
                @test save_mesh_params(mp, json_path) == abspath(json_path)
                @test save_mesh_params(mp, jld_path) == abspath(jld_path)
                @test _mesh_fields_match(load_mesh_params(json_path), mp)
                @test _mesh_fields_match(load_mesh_params(jld_path), mp)
            end
        end
    end
end
