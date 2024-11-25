using GeneralizedSP2
using GershgorinDiscs
using LinearAlgebra
using Plots
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

β = 1.25
μ = 100
H = tridiagonalhamil(1000, 235, 400)
εₘᵢₙ, εₘₐₓ = eigvals_extrema(H)
β′ = rescale_beta(β, (εₘᵢₙ, εₘₐₓ))
μ′ = rescale_mu(μ, (εₘᵢₙ, εₘₐₓ))
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)
nlayers = 18

lower_bound, upper_bound = 0, 1
𝛆′ = chebyshevnodes_1st(500, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, nlayers; max_iter=100_000);
dm = fermi_dirac(fitted.model)(H_scaled)
N = tr(dm)
N_target = N + 50

plot(; PLOT_DEFAULTS..., size=(1600 / 3, 400))
emin, emax = extrema(eigvals(H))
for μ_init in (emin + 10):100:(emax - 10)
    μ′_history = estimate_mu(
        N_target,
        H,
        β,
        𝛆′,
        (εₘᵢₙ, εₘₐₓ),
        μ_init,
        nlayers;
        # occ_tol=1e-4,
        fitting_max_iter=10000,
    )
    μ′_final = μ′_history[end]
    fitted_final = fit_fermi_dirac(𝛆′, μ′_final, β′, nlayers; max_iter=1_000_000)
    dm_final = fermi_dirac(fitted_final.model)(H_scaled)
    N_final = tr(dm_final)
    @show diff = N_final - N_target

    hline!([μ]; seriescolor=:black, primary=false, PLOT_DEFAULTS...)
    plot!(μ′_history; label="μ₀=$μ_init", PLOT_DEFAULTS...)
end
