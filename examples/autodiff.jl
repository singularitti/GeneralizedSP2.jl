using AffineScaler: rescale_one_zero
using DifferentiationInterface
# using FastDifferentiation
# using FiniteDiff
# using Mooncake
using Enzyme
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_jac!
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
    𝐱′, μ′, β′, init_model(μ′, 18); diff=Manual(), maxiters=1000
);
benchmark_model = benchmark_fitted.model

benchmark_derivatives = Array{Float64}(undef, length(𝐱′), length(benchmark_model))
ad_derivatives = Array{Float64}(undef, length(𝐱′), length(benchmark_model))
fermi_dirac_jac!(benchmark_derivatives, benchmark_model, 𝐱′, Manual())
backend = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
# backend = AutoZygote()
# backend = AutoFastDifferentiation()
# backend = AutoMooncake(; config=nothing)
ad_fitted = fit_fermi_dirac(
    𝐱′, μ′, β′, init_model(μ′, 18); diff=Auto(backend), maxiters=1000
);
ad_model = ad_fitted.model
@show benchmark_model ≈ ad_model
fermi_dirac_jac!(ad_derivatives, ad_model, 𝐱′, Auto(backend))
@show norm(benchmark_derivatives - ad_derivatives, Inf)
