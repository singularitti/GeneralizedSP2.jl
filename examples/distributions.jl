using AffineScaler: rescale_one_zero
using Distributions
using GeneralizedSP2
using GeneralizedSP2: fermi_dirac_deriv, fermi_dirac_jac
using LinearAlgebra: Eigen, Hermitian, diag, eigen, eigvals, norm, tr
using Roots: Newton, find_zero
using Plots
using Statistics: mean
using StatsPlots
using ToyHamiltonians: Hamiltonian, EigvalsSampler, EigvecsSampler, set_isapprox_rtol

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 2,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 8,
    :tickfontsize => 6,
    :legendfontsize => 8,
    :left_margin => (6, :mm),
    :bottom_margin => (6, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function hamiltonian(dist, sys_size=2048; rtol=1e-13)
    set_isapprox_rtol(rtol)
    Λ = rand(EigvalsSampler(dist), sys_size)
    V = rand(EigvecsSampler(dist), sys_size, sys_size)
    return Hamiltonian(Eigen(Λ, V))
end

function rescale_hamiltonian(H::AbstractMatrix, μ, β)
    𝚲 = eigvals(H)  # Must be all reals
    εₘᵢₙ, εₘₐₓ = floor(minimum(𝚲)), ceil(maximum(𝚲))
    H′ = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)
    β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
    μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
    return H′, μ′, β′, εₘᵢₙ, εₘₐₓ
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
H_scaled, μ′, β′, εₘᵢₙ, εₘₐₓ = rescale_hamiltonian(H, μ, β)

exact_densitymatrix = fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
exact_occupation = tr(exact_densitymatrix)
E = eigen(H)
𝛌, V = E.values, E.vectors
𝐎 = diag(inv(V) * exact_densitymatrix * V)  # Cannot just use `eigvals` since it is not in corresponding order

𝛆′ = reverse(chebyshevnodes_1st(1000, (0, 1)))  # Have to reverse since β′ is negative
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)
𝛆′_inv = sort(inv(rescale_one_zero(εₘᵢₙ, εₘₐₓ)).(𝛆′))

max_iter = 10_000_000
layers = 18:21
models = map(layers) do nlayers
    fit_fermi_dirac(𝛆′, μ′, β′, init_model(μ′, nlayers); max_iter=max_iter).model
end
𝐲_fitted = map(models) do model
    fermi_dirac(model).(𝛆′)
end
rmse = map(𝐲_fitted) do 𝐲
    residuals = 𝐲 - 𝐲̂
    sqrt(mean(abs2, residuals))
end
derivative_norms = map(models) do model
    𝝝̄ = fermi_dirac_jac(𝛆′, model, Manual())
    norm(𝝝̄, Inf)
end
densitymatrices = map(models) do model
    fermi_dirac(model)(H_scaled)
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
exact_band_energies = tr(exact_densitymatrix * H)
band_energies_diff = map(densitymatrices) do densitymatrix
    tr(densitymatrix * H) - exact_band_energies
end

layout = (1, 3)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600, 400))

scatter!(layers, rmse; yscale=:log10, subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"RMSE of fitting"; subplot=1)

plot!(
    𝛆′_inv,
    𝐲̂;
    subplot=2,
    linestyle=:solid,
    label=raw"reference for fitting",
    PLOT_DEFAULTS...,
)
plot!(
    𝛆′_inv,
    𝐲_fitted[end];
    subplot=2,
    linestyle=:dashdotdot,
    legend_position=:left,
    label="fitted by $(layers[end]) layers",
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
        label="DM eigvals ($nlayer layers)",
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=2)
xlabel!(raw"eigenvalue distribution $\varepsilon$"; subplot=2)
ylabel!(raw"$n(\varepsilon)$"; subplot=2)

hline!([zero(𝐎)]; subplot=3, label="reference for fitting", z_order=:back, PLOT_DEFAULTS...)
plot!(
    𝛆′_inv,
    𝐲_fitted[end] - 𝐲̂;
    subplot=3,
    linestyle=:dashdotdot,
    legend_position=:left,
    label="fitted by $(layers[end]) layers",
    PLOT_DEFAULTS...,
)
hline!([zero(𝐎)]; subplot=3, linestyle=:dash, label="exact DM eigvals", PLOT_DEFAULTS...)
for (fd_distribution, nlayer) in zip(fd_distributions, layers)
    plot!(
        𝛌,
        fd_distribution .- 𝐎;
        subplot=3,
        linestyle=:dot,
        legend_position=:topleft,
        label="DM eigvals ($nlayer layers)",
        PLOT_DEFAULTS...,
    )
end
xlims!(extrema(𝛌); subplot=3)
xlabel!(raw"eigenvalue distribution $\varepsilon$"; subplot=3)
ylabel!(raw"$\Delta n(\varepsilon)$"; subplot=3)
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

hline!([exact_occupation]; subplot=2, xticks=layers, label="exact", PLOT_DEFAULTS...)
scatter!(
    layers,
    occupations;
    subplot=2,
    xticks=layers,
    label="model",
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
    yformatter=:plain,
)
xlims!(extrema(layers); subplot=2)
xlabel!(raw"number of layers $L$"; subplot=2)
ylabel!(raw"$N_\textrm{occ} = \mathrm{tr}(\rho)$"; subplot=2)

scatter!(
    layers,
    band_energies_diff ./ exact_band_energies;
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

layout = (1, 1)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 400))
scatter!(layers, derivative_norms; subplot=1, xticks=layers, label="", PLOT_DEFAULTS...)
xlabel!(raw"number of layers $L$"; subplot=1)
ylabel!(raw"$| \dot{\theta} |_\infty$"; subplot=1)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_derivnorm.png")

layout = (1, 1)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600 / 3, 400))

histogram!(
    𝛌;
    subplot=1,
    nbins=45,
    normalize=true,
    legend_position=:top,
    label="",
    linewidth=0,
    PLOT_DEFAULTS...,
)
xlims!(extrema(𝛌); subplot=1)
xlabel!(raw"eigenvalue distribution $\varepsilon$"; subplot=1)
ylabel!("density"; subplot=1)
savefig("$(dist_name)_$(β)_$(μ)_$(max_iter)_hist.png")
