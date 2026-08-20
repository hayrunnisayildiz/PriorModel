#!/usr/bin/env julia
#=
Copy training artifacts from Google Drive, or build synthetic pairs in-place.

Looks under `--drive-root` (default `/content/drive/MyDrive/PriorModel`) for:

    data/synthetic/<dataset>.h5
    models/*.jld2

If the HDF5 dataset is found, it is copied into the repo. Otherwise
`scripts/build_train_pairs.jl` is invoked with `--n` (default 50).

Usage (Colab — never `--project=.` from `/content`):
    julia --project=/content/PriorModel /content/PriorModel/scripts/colab_fetch_or_build.jl
    julia --project=/content/PriorModel /content/PriorModel/scripts/colab_fetch_or_build.jl --n 100
    julia --project=/content/PriorModel /content/PriorModel/scripts/colab_fetch_or_build.jl \
        --drive-root /content/drive/MyDrive/PriorModel --n 50
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf

const DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/PriorModel"
const DEFAULT_DATASET = "train_pairs.h5"

function parse_args(argv::Vector{String})
    opts = Dict{Symbol,Any}(
        :n => 50,
        :drive_root => DEFAULT_DRIVE_ROOT,
        :dataset => DEFAULT_DATASET,
        :out => joinpath(ROOT, "data", "synthetic", DEFAULT_DATASET),
        :skip_build => false,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        need() = (i += 1; i <= length(argv) ? argv[i] : error("$a needs a value"))
        if a == "--n"
            opts[:n] = parse(Int, need())
        elseif a == "--drive-root"
            opts[:drive_root] = abspath(need())
        elseif a == "--dataset"
            opts[:dataset] = need()
        elseif a == "--out"
            opts[:out] = abspath(need())
        elseif a == "--skip-build"
            opts[:skip_build] = true
        else
            error("unknown argument: $a")
        end
        i += 1
    end
    if opts[:out] == joinpath(ROOT, "data", "synthetic", DEFAULT_DATASET) &&
       opts[:dataset] != DEFAULT_DATASET
        opts[:out] = joinpath(ROOT, "data", "synthetic", opts[:dataset])
    end
    return opts
end

function copy_file!(src::AbstractString, dst::AbstractString)::Bool
    isfile(src) || return false
    mkpath(dirname(dst))
    cp(src, dst; force=true)
    @info "Copied from Drive" src dst bytes=filesize(dst)
    return true
end

function copy_models!(drive_root::AbstractString)
    src_dir = joinpath(drive_root, "models")
    dst_dir = joinpath(ROOT, "models")
    isdir(src_dir) || return 0
    n = 0
    for name in readdir(src_dir)
        endswith(name, ".jld2") || continue
        if copy_file!(joinpath(src_dir, name), joinpath(dst_dir, name))
            n += 1
        end
    end
    return n
end

function build_pairs!(n::Int, out::AbstractString)
    script = joinpath(ROOT, "scripts", "build_train_pairs.jl")
    isfile(script) || error("build script missing: $script")
    @info "Building synthetic pairs in Colab" n out
    run(`$(Base.julia_cmd()) --project=$(ROOT) $script --n $n --out $out`)
    return out
end

function main(argv::Vector{String}=ARGS)
    opts = parse_args(argv)
    drive_root = opts[:drive_root]
    dataset = String(opts[:dataset])
    out = String(opts[:out])
    n = Int(opts[:n])

    println("═" ^ 60)
    println(" colab_fetch_or_build")
    println("═" ^ 60)
    @printf("  drive-root : %s  (exists=%s)\n", drive_root, string(isdir(drive_root)))
    @printf("  dataset    : %s\n", dataset)
    @printf("  dest       : %s\n", out)
    @printf("  n (build)  : %d\n", n)

    drive_h5 = joinpath(drive_root, "data", "synthetic", dataset)
    fetched = false
    if isfile(out)
        @info "Dataset already present; skipping copy/build" path=out
        fetched = true
    elseif copy_file!(drive_h5, out)
        fetched = true
    end

    n_models = copy_models!(drive_root)
    n_models > 0 && @info "Copied checkpoints from Drive" count=n_models

    if !fetched
        if opts[:skip_build]
            error("dataset not found at $out or $drive_h5 and --skip-build is set")
        end
        if !isdir(drive_root)
            @warn "Drive root missing; building synthetic data instead" drive_root
        else
            @info "HDF5 not on Drive; building synthetic data" expected=drive_h5
        end
        build_pairs!(n, out)
    end

    isfile(out) || error("dataset still missing after fetch/build: $out")
    println("\nReady: $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
