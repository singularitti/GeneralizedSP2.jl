using AffineScaler: rescale_one_zero
using DifferentiationInterface
using Mooncake
using Enzyme
using GeneralizedSP2
using GeneralizedSP2: manualdiff_model!
using LinearAlgebra
using ToyHamiltonians

β = 1.25
μ = 100
H = diagonalhamil(1000, 235)
E = eigen(H)
𝛌, V = E.values, E.vectors
εₘᵢₙ, εₘₐₓ = floor(minimum(𝛌)), ceil(maximum(𝛌))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

lower_bound, upper_bound = 0, 1
𝐱′ = chebyshevnodes_1st(300, (lower_bound, upper_bound))
benchmark_fitted = fit_fermi_dirac(
    𝐱′, μ′, β′, init_model(μ′, 18); strategy=Manual(), max_iter=1_000_000
);
ad_fitted = fit_fermi_dirac(
    𝐱′,
    μ′,
    β′,
    init_model(μ′, 18);
    strategy=Auto(AutoMooncake(; config=nothing)),
    max_iter=1_000_000,
);
benchmark_model = benchmark_fitted.model
ad_model = ad_fitted.model
benchmark_model ≈ ad_model

x = 0.6
derivatives = similar(benchmark_model)
benchmark = manualdiff_model!(one, derivatives, benchmark_model, x)
autodiff_model(identity, benchmark_model, x, AutoMooncake(; config=nothing)) ≈ benchmark
