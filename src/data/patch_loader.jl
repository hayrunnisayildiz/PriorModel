"""
    PatchLoader

Memory-efficient patch sampler for large 3-D fusion tensors stored in HDF5.

Reads `(N_x, N_y, N_z, C)` volumes **directly from disk** via HDF5 hyperslabs —
the full tensor is never loaded into RAM.  Each mini-batch is a stack of random
`P_x × P_y × P_z` crops with NaN32 holes zero-filled and an appended mask channel.

# Output layout
`(P_x, P_y, P_z, C + 1, B)` `Float32` array:
- channels `1:C` — cleaned fusion data (NaN32 → 0)
- channel `C + 1` — validity mask (1.0 = at least one finite sample, 0.0 = empty)

# Example
```julia
include("src/data/patch_loader.jl")
using .PatchLoader

sampler = PatchSampler("data/processed/keivitsa_preprocessed.h5";
                       dataset_key="fusion", px=32, py=32, pz=16,
                       batch_size=4, n_batches=100)

for batch in sampler
    X = batch.data   # (32, 32, 16, 12, 4) for C=11
    # ...
end
close(sampler)
```
"""
module PatchLoader

using HDF5
using Random

export PatchSampler, PatchBatch
export volume_shape, n_channels_in, n_channels_out, n_batches
export close, sample_batch!

const DEFAULT_PATCH = (32, 32, 16)

"""
    PatchBatch

One mini-batch of spatial patches.

- `data`: `(P_x, P_y, P_z, C + 1, B)` cleaned fusion + mask
- `origins`: `(i0, j0, k0)` top-left voxel index of each patch in the full volume
"""
struct PatchBatch
    data::Array{Float32,5}
    origins::Vector{NTuple{3,Int}}
end

"""
    PatchSampler

Lazy HDF5 patch iterator.  Keeps the file handle open for repeated hyperslab reads.

# Fields (keyword constructor)
- `path`: HDF5 file path
- `dataset_key`: 4-D dataset name (default `"fusion"`)
- `px`, `py`, `pz`: patch extent (default `32, 32, 16`)
- `batch_size`: patches per batch (default `4`)
- `n_batches`: iterator length (default `32`)
- `rng`: random origin sampler
"""
mutable struct PatchSampler
    path::String
    dataset_key::String
    file::Union{HDF5.File,Nothing}
    dataset::Union{HDF5.Dataset,Nothing}
    nx::Int
    ny::Int
    nz::Int
    nc::Int
    px::Int
    py::Int
    pz::Int
    batch_size::Int
    n_batches::Int
    rng::Random.AbstractRNG
end

"""Resolve a 4-D fusion dataset inside an open HDF5 file."""
function _resolve_dataset(f::HDF5.File, key::AbstractString)::HDF5.Dataset
    if haskey(f, key)
        d = f[key]
        d isa HDF5.Dataset || error("HDF5 entry $(repr(key)) is not a dataset")
        return d
    end
    available = filter(k -> f[k] isa HDF5.Dataset, collect(keys(f)))
    error("Dataset $(repr(key)) not found in $(repr(f.filename)); available: $(available)")
end

"""Infer `(nx, ny, nz, nc)` from a 4-D dataset."""
function _volume_shape(d::HDF5.Dataset)::NTuple{4,Int}
    sz = size(d)
    length(sz) == 4 ||
        error("Expected 4-D fusion tensor, got shape $(sz) for dataset $(repr(d.name))")
    return (Int(sz[1]), Int(sz[2]), Int(sz[3]), Int(sz[4]))
end

function PatchSampler(path::AbstractString;
                      dataset_key::AbstractString="fusion",
                      px::Int=DEFAULT_PATCH[1],
                      py::Int=DEFAULT_PATCH[2],
                      pz::Int=DEFAULT_PATCH[3],
                      batch_size::Int=4,
                      n_batches::Int=32,
                      rng::Random.AbstractRNG=Random.default_rng())
    abspath_path = abspath(path)
    isfile(abspath_path) || error("HDF5 file not found: $(abspath_path)")
    px > 0 && py > 0 && pz > 0 ||
        error("Patch sizes must be positive; got ($px, $py, $pz)")
    batch_size > 0 || error("batch_size must be positive")
    n_batches > 0 || error("n_batches must be positive")

    file = h5open(abspath_path, "r")
    dataset = _resolve_dataset(file, dataset_key)
    nx, ny, nz, nc = _volume_shape(dataset)

    px <= nx && py <= ny && pz <= nz ||
        (close(file);
         error("Patch size ($px, $py, $pz) exceeds volume ($nx, $ny, $nz)"))

    sampler = PatchSampler(string(abspath_path), string(dataset_key),
                           file, dataset,
                           nx, ny, nz, nc,
                           px, py, pz,
                           batch_size, n_batches, rng)
    finalizer(close, sampler)
    return sampler
end

volume_shape(s::PatchSampler)::NTuple{3,Int} = (s.nx, s.ny, s.nz)
n_channels_in(s::PatchSampler)::Int = s.nc
n_channels_out(s::PatchSampler)::Int = s.nc + 1
n_batches(s::PatchSampler)::Int = s.n_batches

function Base.close(s::PatchSampler)
    if s.file !== nothing
        try
            isopen(s.file) && close(s.file)
        catch
        end
        s.file = nothing
        s.dataset = nothing
    end
    return nothing
end

"""Uniform random valid origin so a `patch`-sized crop fits inside length `n`."""
function _random_origin(n::Int, patch::Int, rng::AbstractRNG)::Int
    return patch >= n ? 1 : rand(rng, 1:(n - patch + 1))
end

"""
    _read_patch_raw!(sampler, i0, j0, k0, buf) -> buf

Read one fusion patch from disk into `buf` `(px, py, pz, nc)` via an HDF5 hyperslab.
"""
function _read_patch_raw!(s::PatchSampler, i0::Int, j0::Int, k0::Int,
                          buf::Array{Float32,4})::Array{Float32,4}
    s.dataset === nothing && error("PatchSampler is closed")
    i1 = i0 + s.px - 1
    j1 = j0 + s.py - 1
    k1 = k0 + s.pz - 1
    raw = s.dataset[i0:i1, j0:j1, k0:k1, 1:s.nc]
    if raw === buf
        return buf
    end
    copyto!(buf, raw)
    return buf
end

"""
    _fill_and_mask!(dest, raw)

Replace non-finite samples with 0 and write a mask channel (`nc + 1`).
"""
function _fill_and_mask!(dest::Array{Float32,4}, raw::Array{Float32,4})
    px, py, pz, nc = size(raw)
    size(dest) == (px, py, pz, nc + 1) ||
        error("dest size $(size(dest)) incompatible with raw $(size(raw))")

    @inbounds for k in 1:pz, j in 1:py, i in 1:px
        valid = false
        for c in 1:nc
            v = raw[i, j, k, c]
            if isfinite(v)
                dest[i, j, k, c] = v
                valid = true
            else
                dest[i, j, k, c] = 0.0f0
            end
        end
        dest[i, j, k, nc + 1] = valid ? 1.0f0 : 0.0f0
    end
    return dest
end

"""
    sample_batch!(sampler) -> PatchBatch

Draw one mini-batch of random patches.  Reuses small scratch buffers to avoid
allocating a full-volume array.
"""
function sample_batch!(s::PatchSampler)::PatchBatch
    px, py, pz, nc = s.px, s.py, s.pz, s.nc
    B = s.batch_size
    batch = Array{Float32,5}(undef, px, py, pz, nc + 1, B)
    patch_buf = Array{Float32,4}(undef, px, py, pz, nc)
    patch_out = Array{Float32,4}(undef, px, py, pz, nc + 1)
    origins = Vector{NTuple{3,Int}}(undef, B)

    for b in 1:B
        i0 = _random_origin(s.nx, px, s.rng)
        j0 = _random_origin(s.ny, py, s.rng)
        k0 = _random_origin(s.nz, pz, s.rng)
        origins[b] = (i0, j0, k0)
        _read_patch_raw!(s, i0, j0, k0, patch_buf)
        _fill_and_mask!(patch_out, patch_buf)
        batch[:, :, :, :, b] = patch_out
    end
    return PatchBatch(batch, origins)
end

Base.iterate(s::PatchSampler, state::Int=0) =
    state >= s.n_batches ? nothing : (sample_batch!(s), state + 1)

end # module PatchLoader
