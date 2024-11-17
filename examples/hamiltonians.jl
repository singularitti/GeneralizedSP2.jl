using GeneralizedSP2
using LinearAlgebra
using Plots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :size => (400, 300),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 1,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 9,
    :plot_titlefontsize => 9,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :left_margin => (0, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

β = 1.25
μ = 100
H = diagonalhamil(1000, 235)
𝚲 = eigvals(H)  # Must be all reals
εₘᵢₙ, εₘₐₓ = floor(minimum(𝚲)), ceil(maximum(𝚲))
β′ = rescale_beta(β, (εₘᵢₙ, εₘₐₓ))
μ′ = rescale_mu(μ, (εₘᵢₙ, εₘₐₓ))
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

lower_bound, upper_bound = 0, 1
𝐱′ = chebyshevnodes_1st(1000, (lower_bound, upper_bound))
𝛉 = fit_fermi_dirac(𝐱′, μ′, β′, 18; max_iter=1_000_000).model

dm = fermi_dirac(𝛉)(H_scaled)
N = tr(dm)

@assert rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ)) ≈ fermi_dirac(H_scaled, μ′, β′)
dm_exact = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
N_exact = tr(dm_exact)

scatter(eigvals(H), eigvals(dm_exact); label="target Fermi–Dirac", PLOT_DEFAULTS...)
scatter!(eigvals(H), eigvals(dm); label="MLSP2 model", PLOT_DEFAULTS...)
xlabel!("eigenvalues of H")
ylabel!("Fermi–Dirac distribution")
