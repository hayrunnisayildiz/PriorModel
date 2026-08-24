# write_ini_prior çıktısının MTGeophysics.load_model2d ile parse edilebilir olduğunu doğrular.
# Anahtar/satır sözleşmesi kayarsa VFSA2DMT start_model_path sessizce yanlış mesh okur.

if !isdefined(Main, :ExportPriorToMTGeophysics)
    include(joinpath(@__DIR__, "..", "src", "inference", "export_prior_to_mtgeophysics.jl"))
end
using .ExportPriorToMTGeophysics: write_ini_prior
using MTGeophysics
using Logging

# Nested MeshParams from the export include — distinct from Main.MTMeshParams.
const _Emp = ExportPriorToMTGeophysics.MTResistivityUNet2DLayers.MTMeshParams

function _lateral_pads(dx::Float64, y_padding::Float64, pad_factor::Float64)
    left_pad = Float64[]
    Δy = dx
    accum = 0.0
    while accum < y_padding - 1e-9
        Δy *= pad_factor
        push!(left_pad, Δy)
        accum += Δy
    end
    right_pad = copy(left_pad)
    reverse!(left_pad)
    return left_pad, right_pad
end

_n_vector_lines(n::Int; per_line::Int=12) = n == 0 ? 0 : cld(n, per_line)

@testset ".ini export" begin
    # Tiny synthetic prior — not the 1000-sample training set. Geometry still
    # exercises core + air + lateral padding so VFSA can parse the file.
    mp = _Emp.MeshParams(6, 4, 80.0, 25.0, 2, [0.1, 1.0, 10.0])
    _Emp.validate_mesh_params(mp)
    logres = [Float32(1.0 + 0.05 * iz + 0.02 * ix) for iz in 1:mp.nz, ix in 1:mp.nx]

    title = "unit-test prior"
    n_air = 2
    y_padding = 200.0
    pad_factor = 1.5
    air_top = -100.0
    air_rho = 1e9
    bg_rho = 100.0

    left_pad, right_pad = _lateral_pads(mp.dx, y_padding, pad_factor)
    n_y = length(left_pad) + mp.nx + length(right_pad)
    n_z = n_air + mp.nz
    expected_lines = 2 + 1 +
                     _n_vector_lines(1) +
                     _n_vector_lines(n_y) +
                     _n_vector_lines(n_z) +
                     _n_vector_lines(n_y * n_z) +
                     1 + 1

    mktempdir() do tmp
        ini_path = joinpath(tmp, "prior_start.ini")
        with_logger(NullLogger()) do
            write_ini_prior(logres, ini_path, mp;
                            title=title,
                            n_air_cells=n_air,
                            air_resistivity=air_rho,
                            background_resistivity=bg_rho,
                            y_padding=y_padding,
                            pad_factor=pad_factor,
                            air_top=air_top)
        end
        @test isfile(ini_path)

        text = read(ini_path, String)
        lines = readlines(ini_path)
        @test length(lines) == expected_lines
        @test occursin("# $title", text)
        @test occursin("NZA=$n_air", text)
        @test occursin("LOGE", text)
        @test startswith(strip(lines[3]), "1 $n_y $n_z 0 LOGE")

        model = MTGeophysics.load_model2d(ini_path)
        @test model.format == "LOGE"
        @test model.n_air_cells == n_air
        @test occursin(title, model.title)
        @test length(model.x_cell_sizes) == 1
        @test model.x_cell_sizes[1] ≈ 1.0
        @test length(model.y_cell_sizes) == n_y
        @test length(model.z_cell_sizes) == n_z
        @test size(model.resistivity) == (n_z, n_y)
        @test model.rotation ≈ 0.0
        @test length(model.origin) == 3
        @test model.origin[1] ≈ 0.0
        @test model.origin[3] ≈ air_top

        @test model.y_cell_sizes ≈ vcat(left_pad, fill(mp.dx, mp.nx), right_pad)
        @test model.z_cell_sizes[n_air + 1:end] ≈ fill(mp.dz, mp.nz)

        n_pad_left = length(left_pad)
        for iz in 1:mp.nz, ix in 1:mp.nx
            ρ = model.resistivity[n_air + iz, n_pad_left + ix]
            @test ρ ≈ 10.0^Float64(logres[iz, ix]) rtol=1e-6 atol=1e-6
        end
        @test all(x -> isapprox(x, air_rho; rtol=1e-6), model.resistivity[1:n_air, :])
        @test all(x -> isapprox(x, bg_rho; rtol=1e-6),
                  model.resistivity[n_air + 1:end, 1:n_pad_left])
    end
end
