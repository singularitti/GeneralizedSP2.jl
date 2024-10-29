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
    :size => (1800, 700),
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

function hamiltonian(dist, size=2048, rtol=1e-13)
    set_isapprox_rtol(rtol)
    Λ = rand(EigvalsSampler(dist), size)
    V = rand(EigvecsSampler(dist), size, size)
    return Float32.(Hamiltonian(Eigen(Λ, V)))
end

function rescaled_hamiltonian(H::AbstractMatrix)
    # emin, emax = eigvals_extrema(H)
    emin, emax = minimum(eigvals(H)) - 10, maximum(eigvals(H)) + 10
    return rescale_one_zero(emin, emax)(H), emin, emax
end

function samplex(μ, β, npoints_scale=100)
    lower_bound, upper_bound = zero(μ), oneunit(μ)
    return sample_by_pdf(
        bell_distribution(μ, β, npoints_scale), μ, (lower_bound, upper_bound)
    )
end

β = 10.0f0
μ = 0.5f0

# dist = Cauchy(0.35, 0.2)
# dist = Chisq(5)
# dist = Erlang(100, 10)
# dist = JohnsonSU(0, 1, 0, 1)
# dist = BetaPrime(1, 2)
# dist = Semicircle(50)
# dist = Laplace(0.5, 0.1)
# dist = LogitNormal(-5, 7)
dist = LogUniform(100.0f0, 200.0f0)
# dist = Uniform(-5, 7)
# dist = MixtureModel([Normal(-40, 10), Normal(0, 10), Normal(40, 10)], [0.25, 0.5, 0.25])
# dist = MixtureModel([Cauchy(0.25, 0.2), Laplace(0.5, 0.1)], [0.6, 0.4])
# dist = MixtureModel([Uniform(-10, 50), Uniform(50, 90)], [0.4, 0.6])

H = hamiltonian(dist, 2048)
H_scaled, emin, emax = rescaled_hamiltonian(H)
exact_densitymatrix = rescaled_fermi_dirac(H, μ, β, (emin, emax))
@assert exact_densitymatrix ≈ fermi_dirac(H_scaled, μ, β)
exact_occupation = tr(exact_densitymatrix)

layers = 10:2:30
ys = []
fit_errors = []
diff_norms = []
occupations = []
derivative_norms = []
estimated_mu = []
densitymatrices = []
for nlayers in layers
    𝛉, _, _ = fit_fermi_dirac(𝐱, μ, β, nlayers)

    𝐲 = fermi_dirac_model(𝐱, 𝛉)
    push!(ys, 𝐲)

    residuals = 𝐲 - fermi_dirac.(𝐱, μ, β)
    push!(fit_errors, mean(abs2, residuals))

    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱, 𝛉)
    push!(derivative_norms, norm(𝝝̄))

    densitymatrix = fermi_dirac_model(H_scaled, 𝛉)
    push!(densitymatrices, densitymatrix)
    push!(diff_norms, norm(exact_densitymatrix - densitymatrix))

    occupation = tr(densitymatrix)
    push!(occupations, occupation)

    push!(estimated_mu, estimate_mu(H_scaled, occupation))
end

layout = (2, 4)
plot(; layout=layout, PLOT_DEFAULTS...)

scatter!(layers, diff_norms; subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlims!(extrema(layers); subplot=1)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| D - P |$"; subplot=1)

hline!([exact_occupation]; subplot=2, xticks=layers, label="exact Nocc")
scatter!(layers, occupations; subplot=2, xticks=layers, label="Nocc", PLOT_DEFAULTS...)
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

hline!([μ]; subplot=5, xticks=layers, label="preset μ")
hline!(
    [compute_mu(H_scaled, exact_occupation)];
    subplot=5,
    xticks=layers,
    label="reversed solving μ",
)
scatter!(
    layers,
    estimated_mu;
    subplot=5,
    markershape=:circle,
    xticks=layers,
    legend_position=:left,
    label="estimatd μ",
    PLOT_DEFAULTS...,
)
xlims!(extrema(layers); subplot=5)
xlabel!(raw"number of layers $L$"; subplot=5)
ylabel!(raw"$\mu$"; subplot=5)

𝛌 = eigvals(H)
𝐎 = eigvals(exact_densitymatrix)
plot!(
    𝛌, 𝐎; subplot=6, linestyle=:dot, label="exact FD on eigenvalues of H", PLOT_DEFAULTS...
)
for (densitymatrix, nlayer) in zip(densitymatrices, layers)
    plot!(
        𝛌,
        eigvals(densitymatrix);
        subplot=6,
        linestyle=:dash,
        legend_position=:left,
        label="N=$nlayer",
    )
end
xlims!(extrema(𝛌); subplot=6)
xlabel!(raw"eigenvalues distribution"; subplot=6)
ylabel!("Fermi–Dirac function"; subplot=6)

hline!([zero(𝐎)]; subplot=7, seriescolor=:black, primary=false, PLOT_DEFAULTS...)
for (densitymatrix, nlayer) in zip(densitymatrices, layers)
    plot!(
        𝛌,
        eigvals(densitymatrix) .- 𝐎;
        subplot=7,
        linestyle=:dash,
        legend_position=:topleft,
        label="N=$nlayer",
    )
end
xlims!(extrema(𝛌); subplot=7)
xlabel!(raw"eigenvalues distribution"; subplot=7)
ylabel!("Fermi–Dirac function difference"; subplot=7)

histogram!(
    𝛌;
    subplot=8,
    nbins=40,
    normalize=true,
    legend_position=:top,
    label="diagonalized",
    PLOT_DEFAULTS...,
)
density!(Λ; subplot=8, bandwidth=8, trim=true, label="preset")
xlabel!("eigenvalues distribution"; subplot=8)
ylabel!("density"; subplot=8)
