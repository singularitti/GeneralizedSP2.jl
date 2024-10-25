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
    :size => (1400, 700),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 3,
    :markerstrokealpha => 0,
    :markerstrokewidth => 0,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 7,
    :tickfontsize => 6,
    :legendfontsize => 6,
    :margin => (4, :mm),
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
matsize = 2048
# dist = Cauchy(0.35, 0.2)
# dist = Arcsine(0.2, 0.9)
# dist = Erlang(5, 1)
# dist = JohnsonSU(0, 1, 0, 1)
# dist = BetaPrime(1, 2)
# dist = Exponential(1)
# dist = Laplace(0.5, 0.1)
# dist = LogitNormal(-5, 7)
dist = LogUniform(12, 20)
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
emin, emax = eigvals_extrema(H)
lower_bound, upper_bound = 0, 1
𝐱 = sample_by_pdf(bell_distribution(μ, β, 10), μ, (lower_bound, upper_bound))
H_scaled = rescale_one_zero(emin, emax)(H)
dm_exact = rescaled_fermi_dirac(H, μ, β)
dm_exact ≈ fermi_dirac(H_scaled, μ, β)
N_exact = tr(dm_exact)

nbins = 40
layers = 15:3:40
ys = []
fit_errors = []
diff_norms = []
Noccs = []
derivative_norms = []
estimated_mu = []
dms = []
for nlayers in layers
    𝛉, σ, v = fit_fermi_dirac(𝐱, μ, β, nlayers)
    𝐲 = fermi_dirac_model(𝐱, 𝛉)
    residuals = 𝐲 - fermi_dirac.(𝐱, μ, β)
    fit_err = mean(abs2, residuals)
    push!(fit_errors, fit_err)
    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱, 𝛉)
    dm = fermi_dirac_model(H_scaled, 𝛉)
    push!(dms, dm)
    Nocc = tr(dm)
    push!(estimated_mu, estimate_mu(H_scaled, Nocc))
    push!(diff_norms, norm(dm_exact - dm))
    push!(Noccs, Nocc)
    push!(derivative_norms, norm(𝝝̄))
    push!(ys, 𝐲)
end

layout = @layout [
    grid(1, 4){0.5h}
    [
        grid(1, 2) grid(1, 1){0.5w}
    ]
]
plot(; layout=layout, PLOT_DEFAULTS...)

scatter!(layers, diff_norms; subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlims!(extrema(layers); subplot=1)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| D - P |$"; subplot=1)

hline!([N_exact]; subplot=2, xticks=layers, label="exact Nocc")
scatter!(layers, Noccs; subplot=2, xticks=layers, label="Nocc", PLOT_DEFAULTS...)
xlims!(extrema(layers); subplot=2)
xlabel!(raw"number of layers $L$"; subplot=2)
ylabel!(raw"$N$"; subplot=2)

scatter!(layers, derivative_norms; subplot=3, xticks=layers, label="", PLOT_DEFAULTS...)
xlims!(extrema(layers); subplot=3)
xlabel!(raw"number of layers $L$"; subplot=3)
ylabel!(raw"$| \dot{\theta} |$"; subplot=3)

scatter!(
    layers, fit_errors; yscale=:log10, subplot=4, xticks=layers, label="", PLOT_DEFAULTS...
)
xlims!(extrema(layers); subplot=4)
xlabel!(raw"number of layers $L$"; subplot=4)
ylabel!(raw"MSE of fitting"; subplot=4)

𝛌 = eigvals(H)
plot!(𝛌, eigvals(dm_exact); subplot=5, linestyle=:dot, label="exact FD on eigenvalues of H")
for (dm, nlayer) in zip(dms, layers)
    plot!(
        𝛌, eigvals(dm); subplot=5, linestyle=:dash, legend_position=:left, label="N=$nlayer"
    )
end
xlabel!(raw"eigenvalues distribution"; subplot=5)
ylabel!("Fermi–Dirac function"; subplot=5)

histogram!(
    𝛌;
    subplot=6,
    nbins=nbins,
    normalize=true,
    legend_position=:top,
    label="diagonalized eigenvalues distribution",
    PLOT_DEFAULTS...,
)
histogram!(
    Λ;
    subplot=6,
    nbins=nbins,
    normalize=true,
    label="original random eigvals",
    PLOT_DEFAULTS...,
)
plot!(dist; subplot=6, label="original distribution")
xlabel!("eigenvalues distribution"; subplot=6)
ylabel!("density"; subplot=6)

hline!([μ]; subplot=7, xticks=layers, label="original μ")
hline!(
    [compute_mu(H_scaled, N_exact)]; subplot=7, xticks=layers, label="reversed solving μ"
)
scatter!(
    layers,
    estimated_mu;
    subplot=7,
    markershape=:circle,
    xticks=layers,
    legend_position=:left,
    label="estimatd μ",
    PLOT_DEFAULTS...,
)
xlims!(extrema(layers); subplot=7)
xlabel!(raw"number of layers $L$"; subplot=7)
ylabel!(raw"$\mu$"; subplot=7)
