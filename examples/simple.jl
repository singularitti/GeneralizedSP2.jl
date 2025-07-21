using AffineScaler: rescale_one_zero
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
E = eigen(H)
𝛌, V = E.values, E.vectors
εₘᵢₙ, εₘₐₓ = floor(minimum(𝛌)), ceil(maximum(𝛌))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

lower_bound, upper_bound = 0, 1
𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝛆′, μ′, β′, init_model(μ′, 18); maxiters=10_000_000);
M = fitted.model
M̄ = fitted.jac

@show norm(M̄)

dm = fermi_dirac(M)(H_scaled)
N = tr(dm)

dm_exact = fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
@assert dm_exact ≈ fermi_dirac(H_scaled, μ′, β′)
N_exact = tr(dm_exact)

scatter(
    eigvals(H), diag(inv(V) * dm_exact * V); label="target Fermi–Dirac", PLOT_DEFAULTS...
)
scatter!(eigvals(H), diag(inv(V) * dm * V); label="model", PLOT_DEFAULTS...)
xlabel!("eigenvalues of H")
ylabel!("Fermi–Dirac distribution")
savefig("fd.pdf")

manifolds = eachcol(transpose(hcat(basis(M).(𝛆′)...))[:, (end - 5):end])
plot(𝛆′, manifolds[1]; linestyle=:dot, label="basis", PLOT_DEFAULTS...)
plot!(𝛆′, manifolds[1]; linestyle=:solid, label="accumulated curve", PLOT_DEFAULTS...)
animation = @animate for (manifold, summed) in zip(manifolds, cumsum(manifolds))
    plot!(𝛆′, manifold; linestyle=:dot, label="", PLOT_DEFAULTS...)
    plot!(𝛆′, summed; linestyle=:solid, label="", PLOT_DEFAULTS...)
end
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
gif(animation, "animation.gif"; fps=0.8)
