"""
    GPUUtils

Optional CUDA detection and device transfer helpers.

Call `init_cuda!()` from `Main` immediately after `using LuxCUDA` (or `CUDA`) and
**before** including Lux network modules (Julia 1.12 world-age requirement).
"""
module GPUUtils

using Lux

export init_cuda!, cuda_available, to_device, device_label, cuda_diagnostics, gpu_forward

const _cuda_mod = Ref{Any}(nothing)
const _cuda_ok = Ref(false)

"""
    init_cuda!()

Register the loaded `CUDA` module. Must be called from `Main` after `using CUDA`.
"""
function init_cuda!()
    if isdefined(Main, :CUDA)
        _cuda_mod[] = Main.CUDA
        _cuda_ok[] = Main.CUDA.functional()
    else
        _cuda_mod[] = nothing
        _cuda_ok[] = false
    end
    return _cuda_ok[]
end

"""Return `true` when CUDA.jl is registered and a GPU is usable."""
function cuda_available(; verbose::Bool=false)::Bool
    if _cuda_mod[] === nothing
        # Fallback: CUDA loaded in Main but init_cuda! not called yet
        isdefined(Main, :CUDA) && init_cuda!()
    end
    if _cuda_mod[] === nothing
        verbose && @warn "CUDA.jl not loaded; running on CPU"
        return false
    end
    ok = _cuda_ok[]
    if verbose && !ok
        status = try
            _cuda_mod[].status()
        catch
            "CUDA.functional() returned false"
        end
        @warn "CUDA is installed but no usable GPU was found" status
    end
    return ok
end

"""Move arrays (and nested Lux parameter trees) to GPU."""
function to_device(x, ::Val{true})
    mod = _cuda_mod[]
    mod === nothing && error("CUDA requested but CUDA.jl is not initialized")
    return Lux.recursive_map(mod.cu, x)
end
to_device(x, ::Val{false}) = x

"""
    gpu_forward(f, use_gpu)

Run `f()` on GPU, temporarily enabling scalar indexing when NNlib's im2col conv
path needs it (CUDA disallows scalar GPU indexing by default).

Supports do-block syntax: `gpu_forward(use_gpu) do ... end`.
"""
function gpu_forward(f, use_gpu::Bool)
    use_gpu || return f()
    mod = _cuda_mod[]
    mod === nothing && return f()
    return mod.allowscalar(f)
end

device_label(use_gpu::Bool) = use_gpu ? "CUDA GPU" : "CPU"

"""Print CUDA driver / runtime diagnostics (useful on Colab)."""
function cuda_diagnostics()
    println("-- CUDA diagnostics --")
    try
        run(`nvidia-smi -L`)
    catch
        println("  nvidia-smi: not found (GPU runtime not selected?)")
    end
    mod = _cuda_mod[]
    if mod === nothing && isdefined(Main, :CUDA)
        init_cuda!()
        mod = _cuda_mod[]
    end
    if mod === nothing
        println("  CUDA.jl: not loaded")
    else
        try
            mod.versioninfo()
            println("  CUDA.functional(): ", mod.functional())
        catch err
            println("  CUDA.jl error: ", sprint(showerror, err))
        end
    end
    println("------------------------")
end

end # module
