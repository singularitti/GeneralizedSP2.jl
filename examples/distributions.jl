using Distributions
using GershgorinDiscs
using GeneralizedSP2
using GeneralizedSP2: transform_fermi_dirac_derivative
using LinearAlgebra
using Plots
using StatsPlots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :size => (800, 600),
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
    :left_margin => (1, :mm),
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
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

function compute_mu(𝐇, Nocc)
    Nocc = floor(Int, Nocc)
    evals = eigvals(𝐇)
    HOMO, LUMO = evals[Nocc], evals[Nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = Nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=true)
end

set_isapprox_rtol(1e-13)
matsize = 1000
# dist = Cauchy(0.35, 0.2)
# dist = Arcsine(0.2, 0.9)
# dist = Beta(2, 2)
# dist = Exponential(1)
# dist = Laplace(0.5, 0.1)
# dist = LogitNormal(0, 1)
# dist = LogUniform(0.1, 0.9)
# dist = Uniform(0, 0.8)
# dist = MixtureModel([Normal(0.2, 0.1), Normal(0.5, 0.1), Normal(0.9, 0.1)], [0.3, 0.4, 0.3])
dist = MixtureModel([Cauchy(0.25, 0.2), Laplace(0.5, 0.1)], [0.6, 0.4])
# dist = MixtureModel(
#     [Uniform(0, 0.2), Uniform(0.2, 0.5), Uniform(0.5, 0.7), Uniform(0.7, 1)],
#     [0.1, 0.2, 0.2, 0.5],
# )
# sampler = EigvalsSampler(dist)
# Λ = rand(sampler, matsize)
# V = eigvecs(hamiltonian1(matsize))
# H = Hamiltonian(Eigen(Λ, V))

H = diagonalhamil(matsize, 100)

β = 4
μ = 0.5
emin, emax = eigvals_extrema(H)
𝐱 = rescale_zero_one(emin, emax).(sort(eigvals(H)))  # Cannot do `sort(eigvals(Hinput))` because it is reversed!
𝐲̂ = fermi_dirac.(𝐱, μ, β)

dm_exact = fermi_dirac(H, μ, β)
N_exact = tr(dm_exact)

H_scaled = rescale_zero_one(emin, emax)(H)

nbins = 40
layers = 10:20
Nocc_record = []
diff_norms = []
diff_Nocc = []
derivative_norms = []
estimated_mu = []
computed_mu = []
plot(; layout=(2, 2), PLOT_DEFAULTS...)
for nlayers in layers
    𝛉 = fit_fermi_dirac(𝐱, μ, β, nlayers)
    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱, 𝛉)
    dm = fermi_dirac_model(H_scaled, 𝛉)
    Nocc = tr(dm)
    push!(estimated_mu, estimate_mu(H_scaled, Nocc))
    push!(computed_mu, compute_mu(H_scaled, Nocc))
    push!(Nocc_record, Nocc)
    push!(diff_norms, norm(dm_exact - dm))
    push!(diff_Nocc, N_exact - Nocc)
    push!(derivative_norms, norm(𝝝̄))
end
scatter!(layers, diff_norms; subplot=1, xticks=layers, label="")
scatter!(layers, diff_Nocc; subplot=3, xticks=layers, label="")
# scatter!(layers, derivative_norms; subplot=4, xticks=layers, label="")
hline!([μ]; subplot=4, xticks=layers, label="original μ")
scatter!(
    layers, estimated_mu; subplot=4, markershape=:diamond, xticks=layers, label="estimatd μ"
)
scatter!(
    layers, computed_mu; subplot=4, markershape=:circle, xticks=layers, label="computed μ"
)
xlabel!(raw"number of layers $L$"; subplot=1)
xlabel!(raw"number of layers $L$"; subplot=3)
xlabel!(raw"number of layers $L$"; subplot=4)
ylabel!(raw"$| D - P |^2$"; subplot=1)
ylabel!(raw"$N_0 - N$"; subplot=3)
ylabel!(raw"$\mu$"; subplot=4)

histogram!(
    eigvals(H); subplot=2, nbins=nbins, normalize=true, label="solve the Hamiltonian"
)
histogram!(Λ; subplot=2, nbins=nbins, normalize=true, label="original random eigvals")
plot!(truncated(dist; lower=0, upper=1); subplot=2, label="original distribution")
xlims!(0, 1; subplot=2)
xlabel!("eigenvalues distribution"; subplot=2)
ylabel!("density"; subplot=2)
