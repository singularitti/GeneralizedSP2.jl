using Distributions
using GershgorinDiscs
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_prime, transform_fermi_dirac_derivative
using LinearAlgebra
using Roots: Newton, find_zero
using Plots
using ProgressMeter
using Statistics: mean
using StatsPlots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
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
    :left_margin => (8, :mm),
    :bottom_margin => (6, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function estimate_mu(H_scaled, β′, Nocc)
    Nocc = floor(Int, Nocc)
    diagonal = sort(diag(H_scaled))
    homo, lumo = diagonal[Nocc], diagonal[Nocc + 1]
    μ₀ = (homo + lumo) / 2
    g(μ) = Nocc - sum(fermi_dirac.(diagonal, μ, β′))
    g′(μ) = sum(fermi_dirac_prime.(diagonal, μ, β′))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

function compute_mu(H_scaled, β′, Nocc)
    Nocc = floor(Int, Nocc)
    evals = eigvals(H_scaled)
    homo, lumo = evals[Nocc], evals[Nocc + 1]
    μ₀ = (homo + lumo) / 2
    g(μ) = Nocc - sum(fermi_dirac.(evals, μ, β′))
    g′(μ) = sum(fermi_dirac_prime.(evals, μ, β′))
    return find_zero((g, g′), μ₀, Newton(); atol=1e-8, maxiters=50, verbose=false)
end

function hamiltonian(dist, sys_size=2048; rtol=1e-13)
    set_isapprox_rtol(rtol)
    Λ = rand(EigvalsSampler(dist), sys_size)
    V = rand(EigvecsSampler(dist), sys_size, sys_size)
    return Hamiltonian(Eigen(Λ, V))
end

function rescale_hamiltonian(H::AbstractMatrix)
    # εₘᵢₙ, εₘₐₓ = eigvals_extrema(H)
    𝚲 = eigvals(H)  # Must be all reals
    εₘᵢₙ, εₘₐₓ = floor(minimum(𝚲)), ceil(maximum(𝚲))
    return rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H), εₘᵢₙ, εₘₐₓ
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
dist_name = "loguniform"

H = hamiltonian(dist, 512)
# H = diagonalhamil(1024, 40)
β = 1.25  # Physical
μ = 150  # Physical
H_scaled, εₘᵢₙ, εₘₐₓ = rescale_hamiltonian(H)
β′ = rescale_beta(β, (εₘᵢₙ, εₘₐₓ))
μ′ = rescale_mu(μ, (εₘᵢₙ, εₘₐₓ))

exact_densitymatrix = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
exact_densitymatrix_norm = norm(exact_densitymatrix, Inf)
exact_occupation = tr(exact_densitymatrix)
E = eigen(H)
𝛌, V = E.values, E.vectors
𝐎 = diag(inv(V) * exact_densitymatrix * V)  # Cannot just use `eigvals` since it is not in corresponding order

𝐱′ = reverse(chebyshevnodes_1st(1000, (0, 1)))  # Have to reverse since β′ is negative
𝐲̂ = fermi_dirac.(𝐱′, μ′, β′)
𝐱′_inv = sort(inv(rescale_one_zero(εₘᵢₙ, εₘₐₓ)).(𝐱′))

layers = 10:21
max_iters = [1_000, 10_000, 100_000]

results = map(max_iters) do max_iter
    println("fitting for max_iter = $max_iter")
    timed_results = @showprogress map(layers) do nlayers
        @timed fit_fermi_dirac(𝐱′, μ′, β′, nlayers; max_iter=max_iter)
    end
    𝚯 = map(timed_results) do timed_result
        first(timed_result.value)
    end
    times = map(timed_results) do timed_result
        timed_result.time
    end
    𝐲_fitted = map(𝚯) do 𝛉
        fermi_dirac_model(𝐱′, 𝛉)
    end
    rmse = map(𝚯, 𝐲_fitted) do 𝛉, 𝐲
        residuals = 𝐲 - 𝐲̂
        sqrt(mean(abs2, residuals))
    end
    (rmse=rmse, times=times)
end

time_matrix = hcat([result.times for result in results]...)
rmse_matrix = hcat([result.rmse for result in results]...)

layout = (1, 2)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 400))
plot!(
    layers,
    rmse_matrix;
    subplot=1,
    label=hcat(("max iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    xlabel=raw"number of layers $L$",
    ylabel="RMSE of fitting",
    PLOT_DEFAULTS...,
    legend_position=:topright,
)
plot!(
    layers,
    time_matrix;
    subplot=2,
    label=hcat(("max iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    xlabel=raw"number of layers $L$",
    ylabel="time (s)",
    PLOT_DEFAULTS...,
    legend_position=:topleft,
)
savefig("$(dist_name)_$(β)_$(μ)_time.png")

max_iter = 1_000_000
layers = 18:21
println("fitting for max_iter = $max_iter")
𝚯 = @showprogress map(layers) do nlayers
    𝛉, _, _ = fit_fermi_dirac(𝐱′, μ′, β′, nlayers; max_iter=max_iter)
    𝛉
end
𝐲_fitted = map(𝚯) do 𝛉
    fermi_dirac_model(𝐱′, 𝛉)
end
rmse = map(𝚯, 𝐲_fitted) do 𝛉, 𝐲
    residuals = 𝐲 - 𝐲̂
    sqrt(mean(abs2, residuals))
end
derivative_norms = map(𝚯) do 𝛉
    𝝝̄ = manualdiff_model(transform_fermi_dirac_derivative, 𝐱′, 𝛉)
    norm(𝝝̄, Inf)
end
densitymatrices = map(𝚯) do 𝛉
    fermi_dirac_model(H_scaled, 𝛉)
end
diff_norms = map(densitymatrices) do densitymatrix
    norm(densitymatrix - exact_densitymatrix, Inf)
end
fd_distributions = map(densitymatrices) do densitymatrix
    diag(inv(V) * densitymatrix * V)  # Cannot just use `eigvals` since it is not in corresponding order
end
occupations = map(densitymatrices) do densitymatrix
    tr(densitymatrix)
end
estimated_mu = map(occupations) do occupation
    estimate_mu(H_scaled, β′, occupation)
end
band_energies = map(densitymatrices) do densitymatrix
    tr(densitymatrix * H)
end
exact_band_energies = tr(exact_densitymatrix * H)

layout = (1, 3)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600, 400))

scatter!(layers, rmse; yscale=:log10, subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"RMSE of fitting"; subplot=1)

plot!(
    𝐱′_inv,
    𝐲̂;
    subplot=2,
    linestyle=:solid,
    label=raw"$\hat{\mathbf{y}}$ for fitting",
    PLOT_DEFAULTS...,
)
plot!(
    𝐱′_inv,
    𝐲_fitted[end];
    subplot=2,
    linestyle=:dashdotdot,
    legend_position=:left,
    label="fitted with N=$(layers[end])",
    PLOT_DEFAULTS...,
)
plot!(𝛌, 𝐎; subplot=2, linestyle=:dash, label="exact DM eigvals", PLOT_DEFAULTS...)
for (fd_distribution, nlayer) in zip(fd_distributions, layers)
    plot!(
        𝛌,
        fd_distribution;
        subplot=2,
        linestyle=:dot,
        legend_position=:left,
        label="N=$nlayer",
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=2)
xlabel!(raw"eigenvalue distribution"; subplot=2)
ylabel!("Fermi–Dirac function"; subplot=2)

hline!([zero(𝐎)]; subplot=3, seriescolor=:black, primary=false, PLOT_DEFAULTS...)
plot!(
    𝐱′_inv,
    𝐲_fitted[end] - 𝐲̂;
    subplot=3,
    linestyle=:dashdotdot,
    legend_position=:left,
    label="fitting with N=$(layers[end])",
    PLOT_DEFAULTS...,
)
for (fd_distribution, nlayer) in zip(fd_distributions, layers)
    plot!(
        𝛌,
        fd_distribution .- 𝐎;
        subplot=3,
        linestyle=:dot,
        legend_position=:topleft,
        label="N=$nlayer",
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=3)
xlabel!(raw"eigenvalue distribution"; subplot=3)
ylabel!("occpuation difference"; subplot=3)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_fermi_dirac.png")

layout = (1, 3)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600, 400))

scatter!(
    layers,
    diff_norms;
    subplot=1,
    yscale=:log10,
    xticks=layers,
    label="",
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| \rho - \rho_{\textrm{exact}} |_\infty$"; subplot=1)

hline!([exact_occupation]; subplot=2, xticks=layers, label="exact Nocc", PLOT_DEFAULTS...)
scatter!(
    layers,
    occupations;
    subplot=2,
    xticks=layers,
    label="Nocc",
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
    yformatter=:plain,
)
xlims!(extrema(layers); subplot=2)
xlabel!(raw"number of layers $L$"; subplot=2)
ylabel!(raw"$\mathrm{tr}(\rho)$"; subplot=2)

scatter!(
    layers,
    (band_energies .- exact_band_energies) ./ exact_band_energies;
    subplot=3,
    xticks=layers,
    label="",
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
    PLOT_DEFAULTS...,
)
xlabel!(raw"number of layers $L$"; subplot=3)
ylabel!(
    raw"$\left(\mathrm{tr}(\rho H) - \mathrm{tr}(\rho_{\textrm{exact}} H)\right) / \mathrm{tr}(\rho_{\textrm{exact}} H)$";
    subplot=3,
)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_norm.png")

layout = (1, 2)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 400))

scatter!(layers, derivative_norms; subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| \dot{\theta} |_\infty$"; subplot=1)

hline!([μ′]; subplot=2, xticks=layers, label="preset", PLOT_DEFAULTS...)
hline!(
    [compute_mu(H_scaled, β′, exact_occupation)];
    subplot=2,
    linestyle=:dash,
    xticks=layers,
    label="reversed solved",
    PLOT_DEFAULTS...,
)
scatter!(
    layers,
    estimated_mu;
    subplot=2,
    markershape=:circle,
    xticks=layers,
    legend_position=:left,
    label="estimatd",
    PLOT_DEFAULTS...,
)
xlims!(extrema(layers); subplot=2)
xlabel!(raw"number of layers $L$"; subplot=2)
ylabel!(raw"rescaled $\mu$"; subplot=2)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_mu.png")

layout = (1, 1)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600 / 3, 400))

histogram!(
    𝛌; subplot=1, nbins=45, normalize=true, legend_position=:top, label="", PLOT_DEFAULTS...
)
xlims!(extrema(𝛌); subplot=1)
xlabel!("eigenvalue distribution"; subplot=1)
ylabel!("density"; subplot=1)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_hist.png")
