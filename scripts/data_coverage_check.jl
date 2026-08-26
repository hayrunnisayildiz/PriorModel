#!/usr/bin/env julia
#=
data_coverage_check.jl — Does the synthetic training set actually cover
COMMEMI's MT response, or does COMMEMI sit outside the training
distribution?

This is the natural follow-up to smoothing_ablation.jl: that test showed
smoothing never helps (Hypothesis B rejected) and consistently hurts,
supporting Hypothesis A (the prior's structure is inaccurate because the
training data doesn't represent structures like COMMEMI well enough).
This script tests Hypothesis A directly and quantitatively, without any
retraining.

Method (PCA-based analogue of the t-SNE check in Wang et al. 2023,
Fig. 11 — no new dependency needed, just LinearAlgebra + Statistics which
are already in Project.toml):

  1. Load the packed MT-response tensor X for every synthetic training
     sample (n_stations x n_periods x n_channels x n_models), flatten each
     sample to a feature vector.
  2. Pack COMMEMI's MT response the same way (same mesh, same TE/TE+TM
     channel layout as the training set — read from the H5 attributes).
  3. PCA (via SVD) on the training features. Project both the training
     cloud and COMMEMI onto the top principal components.
  4. Report:
       - variance explained by top components
       - COMMEMI's position vs. the training distribution on each PC
         (z-score: how many training standard deviations away)
       - COMMEMI's nearest-neighbor distance to the training set, vs. the
         training set's own typical (median) nearest-neighbor distance
         -> a ratio >> 1 means COMMEMI sits in a region of MT-response
            space the training set barely samples: an OUT-OF-DISTRIBUTION
            signal, independent of downstream VFSA behavior.

Usage (from project root):
    julia --project=. scripts/data_coverage_check.jl \
        --train-h5 data/synthetic/train_pairs_v5.h5 \
        --data examples/0COMEMI2D-I/Comemi2D1.obs

    julia --project=. scripts/data_coverage_check.jl \
        --train-h5 data/synthetic/train_pairs_v8_tetm_n200_dx160.h5 \
        --data examples/0COMEMI2D-I/Comemi2D1.obs
=#

include(joinpath(@__DIR__, "..", "src", "pkg_setup.jl"))

using Printf
using Statistics
using LinearAlgebra
using HDF5

include(joinpath(ROOT, "src", "synthetic", "MeshParams.jl"))
include(joinpath(ROOT, "src", "synthetic", "MTInputStandardizer.jl"))
using .MTMeshParams: MeshParams
using .MTInputStandardizer: standardize_mt_input, pack_te_response, pack_tetm_response

# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

Base.@kwdef struct Config
    train_h5::String = joinpath(ROOT, "data", "synthetic", "train_pairs_v5.h5")
    data_path::String = joinpath(ROOT, "examples", "0COMEMI2D-I", "Comemi2D1.obs")
    out_dir::String    = joinpath(ROOT, "results", "data_coverage_check")
    n_pcs::Int          = 5
end

function default_out_dir(train_h5::String)::String
    stem = splitext(basename(train_h5))[1]
    stem = replace(stem, r"^train_pairs_?" => "")
    isempty(stem) && (stem = "default")
    return joinpath(ROOT, "results", "data_coverage_check_" * stem)
end

function parse_args(args::Vector{String})::Config
    cfg = Config()
    train_h5, data, outdir, n_pcs = cfg.train_h5, cfg.data_path, cfg.out_dir, cfg.n_pcs
    out_explicit = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--train-h5" && i + 1 <= length(args)
            train_h5 = args[i+1]; i += 2; continue
        elseif a == "--data" && i + 1 <= length(args)
            data = args[i+1]; i += 2; continue
        elseif a == "--out" && i + 1 <= length(args)
            outdir = args[i+1]; out_explicit = true; i += 2; continue
        elseif a == "--n-pcs" && i + 1 <= length(args)
            n_pcs = parse(Int, args[i+1]); i += 2; continue
        end
        i += 1
    end
    if !out_explicit
        outdir = default_out_dir(train_h5)
    end
    return Config(; train_h5=train_h5, data_path=data, out_dir=outdir, n_pcs=n_pcs)
end

# ─────────────────────────────────────────────────────────────────────────
# Load training tensor + mesh from H5 attributes
# ─────────────────────────────────────────────────────────────────────────

function _h5_attr(a, name::String, default)
    return haskey(a, name) ? read(a[name]) : default
end

function load_training_set(path::String)
    isfile(path) || error("Training H5 not found: $path")
    h5open(path, "r") do f
        X = Float64.(read(f["X"]))  # (n_st, n_freq, n_comp, n_models)
        a = HDF5.attributes(f)
        nx = Int(read(a["nx"]))
        nz = Int(read(a["nz"]))
        dx = Float64(read(a["dx"]))
        n_stations = Int(read(a["n_stations"]))
        periods = Float64.(read(a["periods"]))
        n_components = Int(_h5_attr(a, "n_components", size(X, 3)))
        schema = string(_h5_attr(a, "schema", ""))
        tetm_flag = _h5_attr(a, "tetm", nothing)
        tetm = if tetm_flag !== nothing
            Int(tetm_flag) == 1
        else
            n_components == 4 || occursin("tetm", lowercase(schema))
        end
        mp = MeshParams(nx, nz, dx, Float64(read(a["dz"])), n_stations, periods)
        return X, mp, n_components, tetm
    end
end

function commemi_tensor(obs_path::String, mp, n_channels::Int)
    @eval using MTGeophysics
    return Base.invokelatest() do
        data = MTGeophysics.load_data2d(obs_path)
        y = Float64.(collect(data.receivers))
        periods = Float64.(collect(data.periods))
        raw = if n_channels == 4
            pack_tetm_response(data.rho_xy, data.phase_xy, data.rho_yx, data.phase_yx)
        elseif n_channels == 2
            pack_te_response(data.rho_xy, data.phase_xy)
        else
            error("Unsupported n_channels=$n_channels")
        end
        return standardize_mt_input(y, periods, raw; mp=mp, method=:bilinear)
    end
end

# ─────────────────────────────────────────────────────────────────────────
# PCA via SVD (no MultivariateStats dependency needed)
# ─────────────────────────────────────────────────────────────────────────

"""
    pca_fit(X_flat) -> (mean_vec, components, singular_values, explained_var_ratio)

X_flat: (n_models, n_features). Returns PCA basis fit on X_flat.
"""
function pca_fit(X_flat::Matrix{Float64})
    mu = vec(mean(X_flat; dims=1))
    Xc = X_flat .- mu'
    # thin SVD: Xc = U * diagm(S) * V'
    # Materialize V: on Julia 1.11+ F.V may be an Adjoint view.
    F = svd(Xc; full=false)
    V = Matrix{Float64}(F.V)
    var_explained = (F.S .^ 2) ./ sum(F.S .^ 2)
    return mu, V, F.S, var_explained
end

pca_project(x::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
            V::AbstractMatrix{<:Real}, k::Int) =
    (V[:, 1:k])' * (Float64.(x) .- Float64.(mu))

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

function main(cfg::Config)
    @info "Loading training set" cfg.train_h5
    X, mp, n_components, tetm = load_training_set(cfg.train_h5)
    n_st, n_freq, n_comp, n_models = size(X)
    @info "Training set shape" n_st n_freq n_comp n_models tetm mesh_nx=mp.nx mesh_dx=mp.dx n_periods=length(mp.periods)

    # Flatten: (n_models, n_st*n_freq*n_comp)
    feat_dim = n_st * n_freq * n_comp
    X_flat = Matrix{Float64}(undef, n_models, feat_dim)
    for i in 1:n_models
        X_flat[i, :] = vec(X[:, :, :, i])
    end

    @info "Loading COMMEMI and packing with matching mesh/channels" cfg.data_path n_comp
    commemi = commemi_tensor(cfg.data_path, mp, n_comp)
    size(commemi) == (n_st, n_freq, n_comp) ||
        error("COMMEMI tensor $(size(commemi)) != expected ($n_st, $n_freq, $n_comp)")
    commemi_flat = vec(Float64.(commemi))

    # ── PCA ─────────────────────────────────────────────────────────────
    mu, V, S, var_ratio = pca_fit(X_flat)
    k = min(cfg.n_pcs, size(V, 2))
    proj_train = Matrix{Float64}(undef, n_models, k)
    for i in 1:n_models
        proj_train[i, :] = pca_project(X_flat[i, :], mu, V, k)
    end
    proj_commemi = pca_project(commemi_flat, mu, V, k)

    # ── Nearest-neighbor distance check (vectorized via Gram matrix:
    #    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b — avoids an O(n^2 * dim)
    #    Julia loop, which gets slow once n_models is in the hundreds+) ──
    sq_norms = vec(sum(X_flat .^ 2; dims=2))                       # (n_models,)
    gram = X_flat * X_flat'                                        # (n_models, n_models)
    sq_dists = sq_norms .+ sq_norms' .- 2 .* gram
    sq_dists[sq_dists .< 0] .= 0.0                                 # clip fp noise
    for i in 1:n_models
        sq_dists[i, i] = Inf                                       # exclude self
    end
    train_nn_dists = vec(minimum(sq_dists; dims=2)) .|> sqrt
    median_train_nn = median(train_nn_dists)

    commemi_sq_dists = sq_norms .+ sum(commemi_flat .^ 2) .- 2 .* (X_flat * commemi_flat)
    commemi_sq_dists[commemi_sq_dists .< 0] .= 0.0
    commemi_nn = sqrt(minimum(commemi_sq_dists))
    nn_ratio = commemi_nn / median_train_nn

    # ── Report ──────────────────────────────────────────────────────────
    mkpath(cfg.out_dir)
    report_path = joinpath(cfg.out_dir, "coverage_report.txt")
    open(report_path, "w") do io
        println(io, "Data coverage check")
        println(io, "  training set : $(cfg.train_h5)  (n_models=$n_models, tetm=$tetm)")
        println(io, "  COMMEMI      : $(cfg.data_path)")
        println(io, "")
        println(io, "PCA variance explained (top $k components):")
        for c in 1:k
            @printf(io, "  PC%d: %.2f%%\n", c, 100 * var_ratio[c])
        end
        @printf(io, "  cumulative  : %.2f%%\n", 100 * sum(var_ratio[1:k]))
        println(io, "")
        println(io, "COMMEMI position on each PC (train mean=0, train std=1 by definition of z-score):")
        for c in 1:k
            train_std_c = std(proj_train[:, c])
            z = proj_commemi[c] / (train_std_c + 1e-12)
            @printf(io, "  PC%d: commemi=%.3f   train_std=%.3f   z-score=%.2f\n", c, proj_commemi[c], train_std_c, z)
        end
        println(io, "")
        @printf(io, "Median training nearest-neighbor distance : %.4f\n", median_train_nn)
        @printf(io, "COMMEMI nearest-neighbor distance to train : %.4f\n", commemi_nn)
        @printf(io, "Ratio (COMMEMI / median train NN)          : %.2f\n", nn_ratio)
        println(io, "")
        if nn_ratio > 3.0 || any(abs(pca_project(commemi_flat, mu, V, k)[c] / (std(proj_train[:, c]) + 1e-12)) > 3.0 for c in 1:k)
            println(io, "=> COMMEMI is a clear OUTLIER relative to the training distribution")
            println(io, "   (nn_ratio > 3x typical spacing, and/or |z| > 3 on a leading PC).")
            println(io, "   This directly supports Hypothesis A: the synthetic scenario")
            println(io, "   library does not adequately cover COMMEMI-like structures.")
            println(io, "   Priority: expand scenario diversity / scale (see Wang et al. 2023's")
            println(io, "   400k/50k samples), not further loss/regularization tuning.")
        elseif nn_ratio > 1.5
            println(io, "=> COMMEMI is at the EDGE of the training distribution (nn_ratio 1.5-3x).")
            println(io, "   Partial coverage: some representative scenarios help, but density")
            println(io, "   around COMMEMI-like structures should be increased.")
        else
            println(io, "=> COMMEMI falls WITHIN the training distribution's typical spacing.")
            println(io, "   Coverage looks adequate by this metric; the accuracy gap likely")
            println(io, "   comes from elsewhere (network capacity, resolution, or the")
            println(io, "   VFSA/prior interaction itself) — worth re-examining assumptions.")
        end
    end

    println(read(report_path, String))
    @info "Report written" report_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse_args(ARGS))
end
