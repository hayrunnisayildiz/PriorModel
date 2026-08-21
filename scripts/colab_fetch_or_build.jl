#!/usr/bin/env julia
#=
Copy training artifacts from Google Drive. This script must not instantiate
the project or import MTGeophysics: both hang a headless Colab %%bash cell
(Pkg has no TTY output; GLMakie waits for an OpenGL context).

Looks under `--drive-root` (default `/content/drive/MyDrive/PriorModel`) for:

    data/synthetic/<dataset>.h5
    models/*.jld2

If the HDF5 is missing, Colab stops with an error instead of calling
`build_train_pairs.jl`. Pass `--force-build` only on a machine that can load
MTGeophysics (locally, or Colab + `xvfb-run`).

Usage (Colab — never `--project=.` from `/content`):
    julia --project=/content/PriorModel /content/PriorModel/scripts/colab_fetch_or_build.jl
    julia --project=/content/PriorModel /content/PriorModel/scripts/colab_fetch_or_build.jl \
        --drive-root /content/drive/MyDrive/PriorModel
=#

const ROOT = dirname(@__DIR__)

using Printf

const DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/PriorModel"
const DEFAULT_DATASET = "train_pairs.h5"

_is_colab()::Bool =
    isdir("/content") || haskey(ENV, "COLAB_RELEASE_TAG") || haskey(ENV, "COLAB_GPU")

"""Print and flush so Colab `%%bash` shows progress instead of looking frozen."""
function say(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
    return nothing
end

function parse_args(argv::Vector{String})
    opts = Dict{Symbol,Any}(
        :n => 50,
        :drive_root => DEFAULT_DRIVE_ROOT,
        :dataset => DEFAULT_DATASET,
        :out => joinpath(ROOT, "data", "synthetic", DEFAULT_DATASET),
        :skip_build => false,
        :force_build => false,
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
        elseif a == "--force-build"
            opts[:force_build] = true
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
    nbytes = filesize(src)
    say(@sprintf("  copying %.1f MB  %s", nbytes / 1e6, src))
    mkpath(dirname(dst))
    cp(src, dst; force=true)
    say(@sprintf("  wrote     %.1f MB  %s", filesize(dst) / 1e6, dst))
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
    say("Building synthetic pairs (loads MTGeophysics / GLMakie) n=$n")
    julia = Base.julia_cmd()
    cmd = `$julia --project=$(ROOT) $script --n $n --out $out`
    if _is_colab()
        xvfb = Sys.which("xvfb-run")
        xvfb === nothing && error(
            "Colab build needs xvfb (GLMakie). Install with: apt-get install -y xvfb"
        )
        cmd = `$xvfb -a $julia --project=$(ROOT) $script --n $n --out $out`
    end
    run(cmd)
    return out
end

function _missing_dataset_error(out::AbstractString, drive_h5::AbstractString, drive_root::AbstractString)
    return """
dataset not found:
  dest : $out
  Drive: $drive_h5
  Drive root exists: $(isdir(drive_root))

On Colab this script only copies files. It will not call build_train_pairs.jl
because that loads MTGeophysics → GLMakie and hangs a headless VM.

Put the HDF5 here, then re-run:
  /content/drive/MyDrive/PriorModel/data/synthetic/train_pairs.h5

To synthesize locally (or Colab + xvfb): pass --force-build --n 50
"""
end

function main(argv::Vector{String}=ARGS)
    opts = parse_args(argv)
    drive_root = opts[:drive_root]
    dataset = String(opts[:dataset])
    out = String(opts[:out])
    n = Int(opts[:n])

    say("═"^60)
    say(" colab_fetch_or_build  (copy only; no Pkg, no MTGeophysics)")
    say("═"^60)
    say(@sprintf("  drive-root : %s  (exists=%s)", drive_root, string(isdir(drive_root))))
    say(@sprintf("  dataset    : %s", dataset))
    say(@sprintf("  dest       : %s", out))

    drive_h5 = joinpath(drive_root, "data", "synthetic", dataset)
    say(@sprintf("  Drive HDF5 : %s  (exists=%s)", drive_h5, string(isfile(drive_h5))))

    fetched = false
    if isfile(out)
        say("Dataset already in the clone; skipping copy/build")
        say(@sprintf("  %s  (%.1f MB)", out, filesize(out) / 1e6))
        fetched = true
    elseif copy_file!(drive_h5, out)
        fetched = true
    end

    n_models = copy_models!(drive_root)
    n_models > 0 && say("Copied $n_models checkpoint(s) from Drive")

    if !fetched
        allow_build = opts[:force_build] && !opts[:skip_build]
        if !allow_build
            error(_missing_dataset_error(out, drive_h5, drive_root))
        end
        say("HDF5 missing; --force-build requested")
        build_pairs!(n, out)
    end

    isfile(out) || error("dataset still missing after fetch/build: $out")
    say("")
    say("Ready: $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
