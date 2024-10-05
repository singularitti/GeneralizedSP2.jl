using GershgorinDiscs
using GeneralizedSP2
using LinearAlgebra
using Roots: Newton, find_zero

function setup_hamiltonian(N, a=0.01)
    𝐇 = diagm(10.0 * rand(N))
    foreach(1:size(𝐇, 1)) do i
        foreach((i + 1):size(𝐇, 2)) do j
            𝐇[i, j] = exp(-a * (i - j)^2)  # Mimic a non-metallic system or a metallic system at ﬁnite temperature
        end
    end
    return Symmetric(𝐇)
end
function setup_hamiltonian2(N)
    𝐇 = zeros(N, N)
    foreach(1:size(𝐇, 1)) do i
        foreach((i + 1):size(𝐇, 2)) do j
            𝐇[i, j] = exp(-0.0005abs(i - j) / 2) * sin(i + j)
        end
    end
    return Symmetric(𝐇)
end
function setup_hamiltonian3(N)
    return 100 * diagm(sort(rand(N)))
end

function fermi_dirac_derivative(ε, μ, β)
    fd = fermi_dirac(ε, μ, β)
    return -β * fd * (oneunit(fd) - fd)
end

function estimate_mu(𝐇, nocc)
    nocc = floor(Int, nocc)
    diagonal = sort(diag(𝐇))
    HOMO, LUMO = diagonal[nocc], diagonal[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(diagonal, μ, β))
    g′(μ) = sum(fermi_dirac_derivative.(diagonal, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

function compute_mu(𝐇, nocc)
    nocc = floor(Int, nocc)
    evals = eigvals(𝐇)
    HOMO, LUMO = evals[nocc], evals[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_derivative.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

H = setup_hamiltonian(1000)

emin, emax = eigvals_extrema(H)
x = rescale_zero_one(emin, emax).(eigvals(H))
β = 4
μ = 0.8
ŷ = fermi_dirac.(x, μ, β)
𝝷FD, 𝝷ₛ = fit_model(x, μ, β, 10)
a = -1 / (emax - emin)
b = emax / (emax - emin)
Hinput = b * I + a * H

dm = iterate_fermi_dirac(Hinput, 𝝷FD)
N = tr(dm)

@show estimate_mu(Hinput, N)
@show compute_mu(Hinput, N)
