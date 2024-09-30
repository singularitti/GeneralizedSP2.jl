using GeneralizedSP2
using GershgorinDiscs
using LinearAlgebra

function setup_hamiltonian(N, a=0.01)
    𝐇 = diagm(10 * rand(N))
    foreach(1:size(𝐇, 1)) do i
        foreach((i + 1):size(𝐇, 2)) do j
            𝐇[i, j] = exp(-a * (i - j)^2)  # Mimic a non-metallic system or a metallic system at ﬁnite temperature
        end
    end
    return Symmetric(𝐇)
end

β = 4
μ = 0.1
nlayers = 4
lower_bound, upper_bound = 0, 1
𝐇 = setup_hamiltonian(10)
λₘᵢₙ, λₘₐₓ = eigvals_extrema(𝐇)
evals_exact = eigvals(𝐇)
rescaled_evals = rescale_zero_one(λₘᵢₙ, λₘₐₓ).(evals_exact)
𝝷FD, 𝝷ₛ = fit_model(rescaled_evals, μ, β, nlayers)
dm = iterate_fermi_dirac(𝐇, 𝝷FD)
