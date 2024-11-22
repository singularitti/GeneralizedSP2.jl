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
𝐱′ = chebyshevnodes_1st(500, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝐱′, μ′, β′, 18; max_iter=1_00_000);
dm = fermi_dirac(fitted.model)(H_scaled)
N = tr(dm)

target_occupation = N + 30
μ_init = μ + 10
μ′_final = estimate_mu(H, 𝐱′, β, target_occupation, μ_init, (εₘᵢₙ, εₘₐₓ); max_iter=10000)
fitted_final = fit_fermi_dirac(𝐱′, μ′_final, β′, 18; max_iter=1_00_000);
dm_final = fermi_dirac(fitted_final.model)(H_scaled)
N_final = tr(dm_final)
