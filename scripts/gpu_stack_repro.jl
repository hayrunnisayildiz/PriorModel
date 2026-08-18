#!/usr/bin/env julia
# Minimal GPU stack reproduction for Lux/CUDA/cuDNN paths.
# Run on Colab:
#   julia --project=/content/PriorModel /content/PriorModel/scripts/gpu_stack_repro.jl

using Pkg
include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))
activate_project!(joinpath(@__DIR__, ".."))

function section(title)
    println("\n", "=" ^ 60)
    println(title)
    println("=" ^ 60)
end

function loaded_pkg(name::String)
    for (id, mod) in Base.loaded_modules
        id.name == name && return mod
    end
    return nothing
end

function pkg_version(name)
    mod = isdefined(Main, Symbol(name)) ? getfield(Main, Symbol(name)) : loaded_pkg(name)
    if mod !== nothing
        try
            return string(pkgversion(mod))
        catch
        end
    end
    try
        proj = Pkg.project()
        for (_, pkg) in proj.dependencies
            pkg.name == name && return string(pkg.version)
        end
        deps = Pkg.dependencies()
        for (_, pkg) in deps
            pkg.name == name && return string(pkg.version)
        end
    catch
    end
    return "not loaded"
end

function ext_loaded(parent_mod, ext_name)
    return try
        Base.get_extension(parent_mod, ext_name) !== nothing
    catch
        false
    end
end

section("Environment")
println("Julia: ", VERSION)
println("Project: ", Base.active_project())

try
    using LuxCUDA
catch err
    @warn "LuxCUDA not available" exception = err
end
using Lux, Random, Statistics

# LuxLib / NNlib are transitive deps; bind them if loaded.
for name in ("CUDA", "cuDNN", "NNlib", "LuxLib")
    mod = loaded_pkg(name)
    if mod !== nothing && !isdefined(Main, Symbol(name))
        @eval const $(Symbol(name)) = $mod
    end
end

for pkg in (
    "CUDA", "cuDNN", "LuxCUDA", "NNlib", "Lux", "LuxLib", "GPUArraysCore",
    "GPUArrays", "Reactant", "ReactantCore", "MLDataDevices",
)
    println(lpad(pkg, 16), ": ", pkg_version(pkg))
end

function probe(mod, names...)
    for name in names
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        return try
            f isa Function ? f() : f
        catch err
            sprint(showerror, err)
        end
    end
    return "n/a"
end

section("Extension loading state")
if isdefined(Main, :CUDA)
    try
        CUDA.versioninfo()
    catch err
        println("CUDA.versioninfo() failed: ", sprint(showerror, err))
    end
    println("CUDA.functional(): ", probe(CUDA, :functional))
    if try CUDA.functional() catch; false end
        try
            dev = CUDA.device()
            println("GPU: ", CUDA.name(dev))
            cap = try
                CUDA.capability(dev)
            catch
                probe(CUDA, :capability)
            end
            println("Capability: ", cap)
            println("Driver: ", probe(CUDA, :driver_version, :driverversion))
            println("Runtime: ", probe(CUDA, :runtime_version, :runtimeversion))
        catch err
            println("GPU query failed: ", sprint(showerror, err))
        end
        if isdefined(Main, :cuDNN)
            println("cuDNN functional: ", probe(cuDNN, :functional, :has_cudnn))
        else
            println("cuDNN functional: n/a (module not loaded)")
        end
    end
else
    println("CUDA.jl not loaded")
end

for (parent_name, ext) in (
    ("NNlib", :NNlibCUDAExt),
    ("NNlib", :NNlibCUDACUDNNExt),
    ("LuxLib", :LuxLibCUDAExt),
    ("LuxLib", :LuxLibcuDNNExt),
)
    parent = isdefined(Main, Symbol(parent_name)) ? getfield(Main, Symbol(parent_name)) : loaded_pkg(parent_name)
    if parent === nothing
        println(rpad(string(ext), 24), "PARENT NOT LOADED")
    else
        println(rpad(string(ext), 24), ext_loaded(parent, ext) ? "LOADED" : "NOT LOADED")
    end
end

section("Test 1: CuArray matmul (LinearAlgebra / CUBLAS)")
try
    CUDA.functional() || error("CUDA not functional")
    A = CUDA.rand(Float32, 8, 16)
    B = CUDA.rand(Float32, 16, 4)
    C = A * B
    println("OK: plain CuArray matmul -> ", typeof(C), " ", size(C))
catch err
    println("FAIL: ", sprint(showerror, err))
end

section("Test 2: Lux Dense on GPU")
try
    CUDA.functional() || error("CUDA not functional")
    rng = Random.default_rng()
    layer = Dense(32, 16, relu)
    ps, st = Lux.setup(rng, layer)
    ps = Lux.recursive_map(cu, ps)
    x = CUDA.rand(Float32, 32, 4)
    y, _ = layer(x, ps, st)
    println("OK: Dense GPU forward -> ", typeof(y), " ", size(y))
catch err
    println("FAIL: ", sprint(showerror, err))
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
        break
    end
end

section("Test 3: CBAM-style reshape + Dense on GPU")
try
    CUDA.functional() || error("CUDA not functional")
    rng = Random.default_rng()
    channels, batch = 32, 4
    mlp = Chain(Dense(channels, 8, relu), Dense(8, channels))
    ps, st = Lux.setup(rng, mlp)
    ps = Lux.recursive_map(cu, ps)
    x = CUDA.rand(Float32, 4, 4, 4, channels, batch)
    avg = mean(x; dims=(1, 2, 3))
    xc = reshape(avg, channels, batch)
    println("  input type: ", typeof(xc))
    y, _ = mlp(xc, ps, st)
    println("OK: reshape+Dense GPU -> ", typeof(y), " ", size(y))
catch err
    println("FAIL: ", sprint(showerror, err))
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
        break
    end
end

section("Test 4: Conv3D on GPU (cuDNN path when LuxCUDA loaded)")
try
    CUDA.functional() || error("CUDA not functional")
    rng = Random.default_rng()
    layer = Conv((3, 3, 3), 8 => 16, relu; pad=SamePad())
    ps, st = Lux.setup(rng, layer)
    ps = Lux.recursive_map(cu, ps)
    x = CUDA.rand(Float32, 8, 8, 8, 8, 2)
    y, _ = layer(x, ps, st)
    println("OK: Conv3D GPU -> ", typeof(y), " ", size(y))
catch err
    println("FAIL: ", sprint(showerror, err))
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
        break
    end
end

section("Test 5: NNlib.conv on GPU (same backend as Test 4)")
try
    CUDA.functional() || error("CUDA not functional")
    x = CUDA.rand(Float32, 8, 8, 8, 4, 2)
    w = CUDA.rand(Float32, 3, 3, 3, 4, 8)
    nnlib = isdefined(Main, :NNlib) ? NNlib : loaded_pkg("NNlib")
    nnlib === nothing && error("NNlib not loaded")
    y = nnlib.conv(x, w; pad=1)
    println("OK: NNlib.conv GPU -> ", typeof(y), " ", size(y))
catch err
    println("FAIL: ", sprint(showerror, err))
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
        break
    end
end

println("\nDone.")
