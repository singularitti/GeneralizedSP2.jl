using AffineScaler: rescale_one_zero
using GeneralizedSP2
using LinearAlgebra
using Symbolics: expand, @variables
using Latexify: latexify
using ToyHamiltonians

β = 1.25
μ = 100
H = diagonalhamil(1000, 235)
E = eigen(H)
𝛌, V = E.values, E.vectors
εₘᵢₙ, εₘₐₓ = floor(minimum(𝛌)), ceil(maximum(𝛌))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)

lower_bound, upper_bound = 0, 1
𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, init_model(μ′, 4); maxiters=1000);
M = fitted.model

@variables x
ex = M(x)
println(latexify(ex))
println(latexify(expand(ex)))
