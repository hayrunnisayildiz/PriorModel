# Shared environment bootstrap.
# Google Colab's cwd is `/content`, which is not this repo. `Pkg.activate(".")`
# or `--project=.` from that directory creates an empty project and then
# `using HDF5` / `using LuxCUDA` fail with "not found in current path".

using Pkg

function activate_project!(root::AbstractString)
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
    Pkg.activate(root)
    Pkg.resolve()
    Pkg.instantiate()
    Pkg.activate(root)

    active = Base.active_project()
    active_dir = active === nothing ? nothing : dirname(abspath(active))
    if active_dir != root
        error("Active project is $(repr(active)), expected $project_toml (pwd=$(pwd())). Start Julia with: julia --project=$root")
    end
    @info "Project ready" project=active pwd=pwd() julia=string(VERSION)
    return root
end
