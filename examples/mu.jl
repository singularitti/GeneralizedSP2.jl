using AffineScaler: rescale_one_zero
using GeneralizedSP2
using GershgorinDiscs
using LinearAlgebra
using Plots
using Printf
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 2,
    :markerstrokewidth => 0,
    :markerstrokealpha => 0,
    :minorticks => 5,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 8,
    :tickfontsize => 6,
    :legendfontsize => 8,
    :left_margin => (1, :mm),
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
spectral_bounds = eigvals_extrema(H)
β′ = rescale_beta(spectral_bounds)(β)
μ′ = rescale_mu(spectral_bounds)(μ)
H_scaled = rescale_one_zero(spectral_bounds)(H)
nlayers = 18

lower_bound, upper_bound = 0, 1
𝛆′ = chebyshevnodes_1st(500, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, init_model(μ′, nlayers); max_iter=100_000);
dm = fermi_dirac(fitted.model)(H_scaled)
N = tr(dm)
N_target = N + 75

plot(; PLOT_DEFAULTS..., size=(1600 / 3, 400))
ϵₘᵢₙ, ϵₘₐₓ = extrema(eigvals(H))
μ′_histories = []
spectral_bounds_histories = []
for μ_init in (ϵₘᵢₙ + 10):50:(ϵₘₐₓ - 10)
    μ′_history, spectral_bounds_history = estimate_mu(
        N_target,
        H,
        β,
        𝛆′,
        spectral_bounds,
        μ_init,
        nlayers;
        occ_tol=1e-4,
        fit_max_iter=10000,
    )
    μ′_final, spectral_bounds_final = μ′_history[end], spectral_bounds_history[end]
    fitted_final = fit_fermi_dirac(
        𝛆′, μ′_final, β′, init_model(μ′_final, nlayers); max_iter=100_000
    )
    H_scaled = rescale_one_zero(spectral_bounds_final)(H)  # Calculate the final H′
    dm_final = fermi_dirac(fitted_final.model)(H_scaled)
    N_final = tr(dm_final)
    @show diff = N_final - N_target
    push!(μ′_histories, μ′_history)
    push!(spectral_bounds_histories, spectral_bounds_history)

    plot!(
        recover_mu_history(μ′_history, spectral_bounds_history);
        label="μᵢₙᵢₜ=" * Printf.format(Printf.Format("%.4f"), μ_init),
        PLOT_DEFAULTS...,
    )
end
max_iter = maximum(map(length, μ′_histories))
xlabel!("Newton iteration")
ylabel!(raw"$\mu$")
title!(
    raw"Convergence of the estimated $\mu$ starting from different $\mu_\textnormal{init}$ given an $N_\textnormal{target}$",
)
xlims!(1, max_iter)
plot!(; xminorticks=0)
savefig("mu_convergence.pdf")
