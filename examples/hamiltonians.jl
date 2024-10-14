using GershgorinDiscs
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_prime
using LinearAlgebra
# using Plots
using Roots: Newton, find_zero
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :size => (400, 300),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 1,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 9,
    :plot_titlefontsize => 9,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :left_margin => (0, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function estimate_mu(𝐇, nocc)
    nocc = floor(Int, nocc)
    diagonal = sort(diag(𝐇))
    HOMO, LUMO = diagonal[nocc], diagonal[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(diagonal, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(diagonal, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

function compute_mu(𝐇, nocc)
    nocc = floor(Int, nocc)
    evals = eigvals(𝐇)
    HOMO, LUMO = evals[nocc], evals[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

β = 4
μ = 0.8
H = diagonalhamil(1000, 100)

emin, emax = eigvals_extrema(H)
𝐱 = rescale_zero_one(emin, emax).(sort(eigvals(H)))  # Cannot do `sort(eigvals(Hinput))` because it is reversed!
𝐲̂ = fermi_dirac.(𝐱, μ, β)
𝛉 = fit_fermi_dirac(𝐱, μ, β, 10)
H_scaled = rescale_zero_one(emin, emax)(H)

dm = fermi_dirac_model(H_scaled, 𝛉)
N = tr(dm)

dm_exact = fermi_dirac(H_scaled, μ, β)
N_exact = tr(dm_exact)

@show estimate_mu(H_scaled, N)
@show compute_mu(H_scaled, N)
