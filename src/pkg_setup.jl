# Shared environment bootstrap.
#
# This file lives in `src/`, so the repository root is the parent directory.
# Google Colab's cwd is `/content`, which is not this repo. `Pkg.activate(".")`
# or `--project=.` from that directory creates an empty project at `/content`
# and then `using HDF5` / `using LuxCUDA` fail with "not found in current path".
#
# Include this file from any script; it defines `ROOT` and activates the project:
#   scripts/…  →  include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))
#   src/…      →  include(joinpath(@__DIR__, "..", "pkg_setup.jl"))
#   examples/  →  include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))
# Always start Julia with `--project=/path/to/PriorModel` (never `--project=.`
# from `/content`).

const ROOT = dirname(@__DIR__)

using Pkg

function _is_colab()::Bool
    return isdir("/content") ||
           haskey(ENV, "COLAB_RELEASE_TAG") ||
           haskey(ENV, "COLAB_GPU")
end

"""
    activate_project!(root=ROOT)

Activate `root/Project.toml`, align `pwd` and `JULIA_PROJECT`, then instantiate.
On Colab, skip auto-precompile so GLMakie/GLFW cannot hang a headless instantiate.
"""
function activate_project!(root::AbstractString=ROOT)
    root = abspath(normpath(root))
    project_toml = joinpath(root, "Project.toml")
    isfile(project_toml) || error("No Project.toml found at $root")

    parent_toml = joinpath(dirname(root), "Project.toml")
    if isfile(parent_toml) && abspath(parent_toml) != abspath(project_toml)
        @warn "Parent directory has $(parent_toml). That environment can hide this project's packages. Delete it, then start Julia with --project=$root"
    end

    # Keep pwd and JULIA_PROJECT aligned with the repo so later Pkg calls cannot
    # recreate an empty project at `/content`.
    cd(root)
    ENV["JULIA_PROJECT"] = root

    # Headless Colab: Plots/GR must not open a display. GLMakie is a hard
    # dependency of MTGeophysics.jl (cannot be removed here); skipping
    # auto-precompile lets `Pkg.instantiate()` finish without an OpenGL context.
    get!(ENV, "GKSwstype", "nul")
    if _is_colab()
        get!(ENV, "JULIA_PKG_PRECOMPILE_AUTO", "0")
    end

    Pkg.activate(root)
    Pkg.resolve()
    Pkg.instantiate()
    Pkg.activate(root)

    active = Base.active_project()
    active_dir = active === nothing ? nothing : rstrip(normpath(dirname(abspath(active))), '/')
    root_cmp = rstrip(normpath(root), '/')
    if active_dir != root_cmp
        error("Active project is $(repr(active)), expected $project_toml (pwd=$(pwd())). Start Julia with: julia --project=$root")
    end
    @info "Project ready" project=active pwd=pwd() julia=string(VERSION)
    return root
end

activate_project!(ROOT)
