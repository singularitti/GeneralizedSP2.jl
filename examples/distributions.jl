using Distributions
using GershgorinDiscs
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_prime, transform_fermi_dirac_derivative
using LinearAlgebra
using Roots: Newton, find_zero
using Plots
using StatsPlots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :size => (1210, 700),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 1,
    :markerstrokewidth => 0,
    :titlefontsize => 9,
    :plot_titlefontsize => 9,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :left_margin => (3, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :right,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function estimate_mu(𝐇, Nocc)
    Nocc = floor(Int, Nocc)
    diagonal = sort(diag(𝐇))
    HOMO, LUMO = diagonal[Nocc], diagonal[Nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = Nocc - sum(fermi_dirac.(diagonal, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(diagonal, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

function compute_mu(𝐇, Nocc)
    Nocc = floor(Int, Nocc)
    evals = eigvals(𝐇)
    HOMO, LUMO = evals[Nocc], evals[Nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = Nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

set_isapprox_rtol(1e-13)
β = 4
μ = 0.8
matsize = 1024
dist = Cauchy(0.35, 0.2)
# dist = Arcsine(0.2, 0.9)
dist = Erlang(5, 1)
dist = JohnsonSU(0, 1, 0, 1)
# dist = BetaPrime(1, 2)
# dist = Exponential(1)
# dist = Laplace(0.5, 0.1)
# dist = LogitNormal(-5, 7)
# dist = LogUniform(0.1, 0.9)
# dist = Uniform(-5, 7)
# dist = MixtureModel([Normal(0.2, 0.1), Normal(0.5, 0.1), Normal(0.9, 0.1)], [0.3, 0.4, 0.3])
# dist = MixtureModel([Cauchy(0.25, 0.2), Laplace(0.5, 0.1)], [0.6, 0.4])
# dist = MixtureModel(
#     [Uniform(0, 0.2), Uniform(0.2, 0.5), Uniform(0.5, 0.7), Uniform(0.7, 1)],
#     [0.1, 0.2, 0.2, 0.5],
# )
Λ = rand(EigvalsSampler(dist), matsize)
V = rand(EigvecsSampler(dist), matsize, matsize)
H = Hamiltonian(Eigen(Λ, V))
# H = diagonalhamil(matsize, 100)
emin, emax = eigvals_extrema(H)
lower_bound, upper_bound = 0, 1
𝐱 = sample_by_pdf(bell_distribution(μ, β, 10), μ, (lower_bound, upper_bound))
H_scaled = rescale_one_zero(emin, emax)(H)
dm_exact = fermi_dirac(H_scaled, μ, β)
N_exact = tr(dm_exact)

nbins = 40
layers = 20:3:50
ys = []
fit_errors = []
diff_norms = []
Noccs = []
derivative_norms = []
estimated_mu = []
for nlayers in layers
    𝛉, σ, v = fit_fermi_dirac(𝐱, μ, β, nlayers)
    𝐲 = fermi_dirac_model(𝐱, 𝛉)
    residuals = 𝐲 - fermi_dirac.(𝐱, μ, β)
    fit_err = mean(abs2, residuals)
    push!(fit_errors, fit_err)
    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱, 𝛉)
    dm = fermi_dirac_model(H_scaled, 𝛉)
    Nocc = tr(dm)
    push!(estimated_mu, estimate_mu(H_scaled, Nocc))
    push!(diff_norms, norm(dm_exact - dm))
    push!(Noccs, Nocc)
    push!(derivative_norms, norm(𝝝̄))
    push!(ys, 𝐲)
end

plot(; layout=(2, 3), PLOT_DEFAULTS...)

scatter!(layers, diff_norms; subplot=1, xticks=layers, label="")
xlims!(extrema(layers); subplot=1)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| D - P |$"; subplot=1)

hline!([N_exact]; subplot=2, xticks=layers, label="exact Nocc")
scatter!(layers, Noccs; subplot=2, xticks=layers, label="Nocc")
xlims!(extrema(layers); subplot=2)
xlabel!(raw"number of layers $L$"; subplot=2)
ylabel!(raw"$N$"; subplot=2)

scatter!(layers, derivative_norms; subplot=3, xticks=layers, label="")
xlims!(extrema(layers); subplot=3)
xlabel!(raw"number of layers $L$"; subplot=3)
ylabel!(raw"$| \dot{\theta} |$"; subplot=3)

hline!([μ]; subplot=4, xticks=layers, label="original μ")
hline!(
    [compute_mu(H_scaled, N_exact)]; subplot=4, xticks=layers, label="reversed solving μ"
)
scatter!(
    layers,
    estimated_mu;
    subplot=4,
    markershape=:diamond,
    xticks=layers,
    legend_position=:left,
    label="estimatd μ",
)
xlims!(extrema(layers); subplot=4)
xlabel!(raw"number of layers $L$"; subplot=4)
ylabel!(raw"MSE of fitting"; subplot=4)

plot!(
    eigvals(H),
    fermi_dirac.(eigvals(H), μ, β);
    subplot=5,
    linestyle=:dot,
    label="exact FD on eigenvalues of H",
)
plot!(
    𝐱, fermi_dirac.(𝐱, μ, β); subplot=5, linestyle=:solid, label=raw"exact FD in $[0, 1]$"
)
for (𝐲, nlayer) in zip(ys, layers)
    plot!(
        𝐱,
        𝐲;
        subplot=5,
        linestyle=:dash,
        legend_position=:left,
        label="N=$nlayer in " * raw"$[0, 1]$",
    )
end
xlabel!(raw"eigenvalues distribution"; subplot=5)
ylabel!("Fermi–Dirac function"; subplot=5)

histogram!(
    eigvals(H);
    subplot=6,
    nbins=nbins,
    normalize=true,
    legend_position=:top,
    label="diagonalized eigenvalues distribution",
)
histogram!(Λ; subplot=6, nbins=nbins, normalize=true, label="original random eigvals")
plot!(dist; subplot=6, label="original distribution")
xlabel!("eigenvalues distribution"; subplot=6)
ylabel!("density"; subplot=6)
