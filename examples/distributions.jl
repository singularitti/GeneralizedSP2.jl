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
    :size => (1900, 900),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 3,
    :markerstrokealpha => 0,
    :markerstrokewidth => 0,
    :yformatter => :plain,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 7,
    :tickfontsize => 6,
    :legendfontsize => 6,
    :left_margin => (8, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function estimate_mu(𝐇, β, Nocc)
    Nocc = floor(Int, Nocc)
    diagonal = sort(diag(𝐇))
    HOMO, LUMO = diagonal[Nocc], diagonal[Nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = Nocc - sum(fermi_dirac.(diagonal, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(diagonal, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

function compute_mu(𝐇, β, Nocc)
    Nocc = floor(Int, Nocc)
    evals = eigvals(𝐇)
    HOMO, LUMO = evals[Nocc], evals[Nocc + 1]
    μ₀ = (HOMO + LUMO) / 2
    g(μ) = Nocc - sum(fermi_dirac.(evals, μ, β))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

function hamiltonian(dist, syssize=2048; rtol=1e-13)
    set_isapprox_rtol(rtol)
    Λ = rand(EigvalsSampler(dist), syssize)
    V = rand(EigvecsSampler(dist), syssize, syssize)
    return Hamiltonian(Eigen(Λ, V))
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

# dist = Cauchy(0.35, 0.2)
# dist = Chisq(5)
# dist = Erlang(100, 10)
# dist = JohnsonSU(0, 1, 0, 1)
# dist = BetaPrime(1, 2)
# dist = Semicircle(50)
# dist = Laplace(0.5, 0.1)
# dist = LogitNormal(-5, 7)
dist = LogUniform(100, 200)
# dist = Uniform(-5, 7)
# dist = MixtureModel([Normal(-40, 10), Normal(0, 10), Normal(40, 10)], [0.25, 0.5, 0.25])
# dist = MixtureModel([Cauchy(0.25, 0.2), Laplace(0.5, 0.1)], [0.6, 0.4])
# dist = MixtureModel([Uniform(-10, 50), Uniform(50, 90)], [0.4, 0.6])

layout = (3, 3)
plot(; layout=layout, PLOT_DEFAULTS...)

T = Float64
H = hamiltonian(dist, 2048)
H = T.(H)
β = convert(T, 100)
μ = convert(T, 0.4)
H_scaled, εₘᵢₙ, εₘₐₓ = rescaled_hamiltonian(H)
exact_densitymatrix = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
exact_occupation = tr(exact_densitymatrix)
𝛌 = eigvals(H)
𝐎 = fermi_dirac.(rescale_one_zero(εₘᵢₙ, εₘₐₓ).(𝛌), μ, β)

𝐱 = samplex(μ, β, 100)
𝐱_inv = εₘₐₓ .- (εₘₐₓ - εₘᵢₙ) * 𝐱

layers = 15:2:30
𝚯 = map(layers) do nlayers
    𝛉, _, _ = fit_fermi_dirac(𝐱, μ, β, nlayers)
    𝛉
end
ys = map(𝚯) do 𝛉
    fermi_dirac_model(𝐱, 𝛉)
end
fit_errors = map(𝚯, ys) do 𝛉, 𝐲
    residuals = 𝐲 - fermi_dirac.(𝐱, μ, β)
    mean(abs2, residuals)
end
derivative_norms = map(𝚯) do 𝛉
    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱, 𝛉)
    norm(𝝝̄)
end
densitymatrices = map(𝚯) do 𝛉
    fermi_dirac_model(H_scaled, 𝛉)
end
diff_norms = map(densitymatrices) do densitymatrix
    norm(exact_densitymatrix - densitymatrix)
end
occupations = map(densitymatrices) do densitymatrix
    tr(densitymatrix)
end
estimated_mu = map(occupations) do occupation
    estimate_mu(H_scaled, β, occupation)
end
band_energies = map(densitymatrices) do densitymatrix
    tr(densitymatrix * H)
end
exact_band_energies = tr(exact_densitymatrix * H)

scatter!(
    layers,
    fit_errors;
    yscale=:log10,
    subplot=1,
    xticks=layers,
    label=string(eltype(fit_errors)),
    PLOT_DEFAULTS...,
)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"MSE of fitting"; subplot=1)

plot!(
    𝛌,
    𝐎;
    subplot=2,
    linestyle=:dash,
    label="exact FD on eigenvalues of H: " * string(eltype(𝐎)),
    PLOT_DEFAULTS...,
)
plot!(
    𝐱_inv,
    ys[end];
    subplot=2,
    linestyle=:solid,
    legend_position=:left,
    label="fitting with N=$(layers[end])",
    PLOT_DEFAULTS...,
)
for (densitymatrix, nlayer) in zip(densitymatrices, layers)
    plot!(
        𝛌,
        eigvals(densitymatrix);
        subplot=2,
        linestyle=:dot,
        legend_position=:left,
        label="N=$nlayer: " * string(eltype(densitymatrix)),
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=2)
xlabel!(raw"eigenvalues distribution"; subplot=2)
ylabel!("Fermi–Dirac function"; subplot=2)

hline!([zero(𝐎)]; subplot=3, seriescolor=:black, primary=false, PLOT_DEFAULTS...)
for (densitymatrix, nlayer) in zip(densitymatrices, layers)
    plot!(
        𝛌,
        eigvals(densitymatrix) .- 𝐎;
        subplot=3,
        linestyle=:dot,
        legend_position=:topleft,
        label="N=$nlayer: " * string(eltype(densitymatrix)),
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=3)
xlabel!(raw"eigenvalues distribution"; subplot=3)
ylabel!("Fermi–Dirac function difference"; subplot=3)

scatter!(
    layers,
    diff_norms;
    subplot=4,
    xticks=layers,
    label=string(eltype(diff_norms)),
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
)
xlabel!(raw"number of layers $L$"; subplot=4)
ylabel!(raw"$| D - P | / | P |$"; subplot=4)

hline!(
    [exact_occupation];
    subplot=5,
    xticks=layers,
    label="exact Nocc: " * string(eltype(exact_occupation)),
    PLOT_DEFAULTS...,
)
scatter!(
    layers,
    occupations;
    subplot=5,
    xticks=layers,
    label="Nocc: " * string(eltype(occupations)),
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
    PLOT_DEFAULTS...,
)
xlims!(extrema(layers); subplot=5)
xlabel!(raw"number of layers $L$"; subplot=5)
ylabel!(raw"$N$"; subplot=5)

scatter!(
    layers,
    band_energies .- exact_band_energies;
    subplot=6,
    xticks=layers,
    label=string(eltype(band_energies)),
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
    PLOT_DEFAULTS...,
)
xlabel!(raw"number of layers $L$"; subplot=6)
ylabel!(raw"$\mathrm{tr}(\rho H) - \mathrm{tr}(\rho_{\textrm{exact}} H)$"; subplot=6)

scatter!(layers, derivative_norms; subplot=7, xticks=layers, label="", PLOT_DEFAULTS...)
xlabel!(raw"number of layers $L$"; subplot=7)
ylabel!(raw"$| \dot{\theta} |$"; subplot=7)

hline!([μ]; subplot=8, xticks=layers, label="preset μ", PLOT_DEFAULTS...)
# hline!(
#     [compute_mu(H_scaled, β, exact_occupation)];
#     subplot=8,
#     xticks=layers,
#     label="reversed solving μ: " * string(eltype(exact_occupation)),
#     PLOT_DEFAULTS...,
# )
scatter!(
    layers,
    estimated_mu;
    subplot=8,
    markershape=:circle,
    xticks=layers,
    legend_position=:left,
    label="estimatd μ: " * string(eltype(estimated_mu)),
    PLOT_DEFAULTS...,
)
xlims!(extrema(layers); subplot=8)
xlabel!(raw"number of layers $L$"; subplot=8)
ylabel!(raw"$\mu$"; subplot=8)

histogram!(
    𝛌;
    subplot=9,
    nbins=45,
    normalize=true,
    legend_position=:top,
    label=string(eltype(𝛌)),
    PLOT_DEFAULTS...,
)
xlims!(extrema(𝛌); subplot=9)
xlabel!("eigenvalues distribution"; subplot=9)
ylabel!("density"; subplot=9)

plot!()
