using GershgorinDiscs
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_prime, rescaled_fermi_dirac
using LinearAlgebra
using Plots
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

function estimate_mu(H, nocc)
    nocc = floor(Int, nocc)
    diagonal = sort(diag(H))
    HOMO, LUMO = diagonal[nocc], diagonal[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(diagonal, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(diagonal, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

function compute_mu(H, nocc)
    nocc = floor(Int, nocc)
    evals = eigvals(H)
    HOMO, LUMO = evals[nocc], evals[nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

β = 10
μ = 0.45
H = diagonalhamil(1000, 235)

emin, emax = eigvals_extrema(H)
lower_bound, upper_bound = 0, 1
𝐱 = sample_by_pdf(bell_distribution(μ, β, 10), μ, (lower_bound, upper_bound))
𝛉, _, _ = fit_fermi_dirac(𝐱, μ, β, 10)
H_scaled = rescale_one_zero(emin, emax)(H)

dm = fermi_dirac_model(H_scaled, 𝛉)
N = tr(dm)

rescaled_fermi_dirac(H, μ, β) ≈ fermi_dirac(H_scaled, μ, β)
dm_exact = rescaled_fermi_dirac(H, μ, β)
N_exact = tr(dm_exact)

@show estimate_mu(H_scaled, N)
@show compute_mu(H_scaled, N)

scatter(eigvals(H), eigvals(dm_exact); label="target Fermi–Dirac", PLOT_DEFAULTS...)
scatter!(eigvals(H), eigvals(dm); label="MLSP2 model", PLOT_DEFAULTS...)
xlabel!("eigenvalues of H")
ylabel!("Fermi–Dirac distribution")
