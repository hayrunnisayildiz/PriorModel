# Pixel-weighted L1: unweighted path stays identical; anomaly pixels get w > 1;
# Zygote can differentiate the weighted term w.r.t. the prediction.

if !isdefined(Main, :MTPriorLoss)
    include(joinpath(@__DIR__, "..", "src", "training", "MTPriorLoss.jl"))
end
using .MTPriorLoss
using Zygote
using Random
using Statistics: mean, median

@testset "MTPriorLoss unweighted identity" begin
    pred = Float32[1.0 2.0; 3.0 4.0]
    target = Float32[1.5 2.5; 2.5 3.5]
    terms = mt_prior_loss_terms(pred, target; λ_tv=0.0f0)
    @test terms.data ≈ mean(abs, pred .- target)
    @test mt_prior_loss_terms(pred, target; λ_tv=0.0f0, weighted=false).data ≈ terms.data
    @test mt_prior_loss_terms(pred, target; λ_tv=0.0f0, weighted=false).total ≈ terms.total
end

@testset "MTPriorLoss anomaly pixels have w > 1" begin
    target = fill(2.0f0, 8, 8, 2)
    target[3:5, 3:5, 1] .= 4.0f0   # +2 decades vs host
    target[2:3, 6:7, 2] .= 0.5f0   # conductive block
    w = MTPriorLoss._anomaly_l1_weights(target; λ_boundary=5.0f0)
    @test size(w) == size(target)
    @test all(w[1, 1, :] .≈ 1.0f0)                 # host pixel
    @test w[4, 4, 1] ≈ 1.0f0 + 5.0f0 * 2.0f0       # resistive anomaly
    @test w[2, 6, 2] ≈ 1.0f0 + 5.0f0 * 1.5f0       # conductive anomaly
    @test w[4, 4, 1] > 1.0f0
    @test w[2, 6, 2] > 1.0f0

    pred = copy(target)
    pred[4, 4, 1] = 2.0f0   # miss the resistive block
    unweighted = mt_prior_loss_terms(pred, target; λ_tv=0.0f0, weighted=false).data
    weighted = mt_prior_loss_terms(pred, target; λ_tv=0.0f0, weighted=true).data
    @test weighted > unweighted
end

@testset "MTPriorLoss Zygote gradient through weighted L1" begin
    rng = MersenneTwister(0)
    pred = randn(rng, Float32, 6, 6, 2)
    target = fill(2.0f0, 6, 6, 2)
    target[2:4, 2:4, 1] .= 3.5f0
    g = Zygote.gradient(p -> mt_prior_loss(p, target; weighted=true, λ_tv=0.0f0), pred)
    @test g[1] !== nothing
    @test size(g[1]) == size(pred)
    @test all(isfinite, g[1])
    @test any(!iszero, g[1])
    # Host median is taken from target only; pred-median must not enter the graph.
    host = median(target[:, :, 1])
    @test host ≈ 2.0f0
end
