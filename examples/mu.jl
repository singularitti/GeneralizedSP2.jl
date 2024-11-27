using GeneralizedSP2
using GershgorinDiscs
using LinearAlgebra
using Plots
using Printf
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
    :left_margin => (1, :mm),
    :bottom_margin => (1, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :topright,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

β = 1.25
μ = 100
H = diagonalhamil(1000, 235)
𝛜 = eigvals_extrema(H)
β′ = rescale_beta(𝛜)(β)
μ′ = rescale_mu(𝛜)(μ)
H_scaled = rescale_one_zero𝛜(H)
nlayers = 18

lower_bound, upper_bound = 0, 1
𝛆′ = chebyshevnodes_1st(500, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, nlayers; max_iter=100_000);
dm = fermi_dirac(fitted.model)(H_scaled)
N = tr(dm)
N_target = N + 50

plot(; PLOT_DEFAULTS..., size=(1600 / 3, 400))
ϵₘᵢₙ, ϵₘₐₓ = extrema(eigvals(H))
μ′_histories = []
for μ_init in (ϵₘᵢₙ + 10):50:(ϵₘₐₓ - 10)
    μ′_history = estimate_mu(
        N_target,
        H,
        β,
        𝛆′,
        (εₘᵢₙ, εₘₐₓ),
        μ_init,
        nlayers;
        occ_tol=1e-4,
        fitting_max_iter=10000,
    )
    μ′_final = μ′_history[end]
    fitted_final = fit_fermi_dirac(𝛆′, μ′_final, β′, nlayers; max_iter=1_000_000)
    dm_final = fermi_dirac(fitted_final.model)(H_scaled)
    N_final = tr(dm_final)
    @show diff = N_final - N_target

    push!(μ′_histories, μ′_history)
end
max_iter = maximum(map(length, μ′_histories))
for (μ′_history, μ_init) in zip(μ′_histories, (ϵₘᵢₙ + 10):50:(ϵₘₐₓ - 10))
    plot!(
        map(rescale_mu(𝛜), μ′_history);
        xticks=Base.OneTo(length(μ′_history)),
        label="μ₀=" * Printf.format(Printf.Format("%.4f"), μ_init),
        PLOT_DEFAULTS...,
    )
end
plot!(; xticks=Base.OneTo(max_iter))
xlabel!("iteration")
ylabel!(raw"$\mu$")
