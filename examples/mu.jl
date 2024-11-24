using GeneralizedSP2
using GershgorinDiscs
using LinearAlgebra
using ToyHamiltonians

β = 1.25
μ = 100
H = tridiagonalhamil(1000, 235, 235)
εₘᵢₙ, εₘₐₓ = eigvals_extrema(H)
β′ = rescale_beta(β, (εₘᵢₙ, εₘₐₓ))
μ′ = rescale_mu(μ, (εₘᵢₙ, εₘₐₓ))
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

lower_bound, upper_bound = 0, 1
𝛆′ = chebyshevnodes_1st(500, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, 18; max_iter=1_00_000);
dm = fermi_dirac(fitted.model)(H_scaled)
N = tr(dm)

N_target = N + 100
μ_init = μ + 50
μ′_final = estimate_mu(N_target, H, β, 𝛆′, (εₘᵢₙ, εₘₐₓ), μ_init; fitting_max_iter=1_000_000)
fitted_final = fit_fermi_dirac(𝛆′, μ′_final, β′, 18; max_iter=1_000_000);
dm_final = fermi_dirac(fitted_final.model)(H_scaled)
N_final = tr(dm_final)
@show diff = N_final - N_target
