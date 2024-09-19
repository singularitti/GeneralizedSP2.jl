using GeneralizedSP2
using Plots

PLOT_DEFAULTS = Dict(
    # :size => (450, 600),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 2,
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
    :legend_position => :left,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

target_fermi_dirac(ε) = 1 / (1 + exp(β * (ε - μ)))

β = 9.423
μ = 0.568
minlayers = 2
maxlayers = 4
lower_bound, upper_bound = 0, 1

branches = determine_branches(μ, maxlayers)
𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))
𝐲̂ = target_fermi_dirac.(𝐱)

𝝷, 𝐲, predictions = fit0(𝐱, 𝐲̂, 4)

plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
hline!([0]; label="", seriescolor=:black, primary=false)
plot!(
    𝐱,
    target_fermi_dirac.(𝐱);
    z_order=:back,
    seriescolor=:maroon,
    label="Reference Fermi function",
    PLOT_DEFAULTS...,
)
for (i, 𝐲) in enumerate(predictions)
    plot!(𝐱, 𝐲; label="Fitted: \$x^{$(2^(i - 1))}\$", linestyle=:dot, PLOT_DEFAULTS...)
end
title!("Fitting without regularization")
savefig("fitting_no_regularization.png")

𝝷, 𝐲, predictions = fit(𝐱, 𝐲̂, 4; λ₁=2.5, λ₂=2.5)

plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
hline!([0]; label="", seriescolor=:black, primary=false)
plot!(
    𝐱,
    target_fermi_dirac.(𝐱);
    z_order=:back,
    seriescolor=:maroon,
    label="Reference Fermi function",
    PLOT_DEFAULTS...,
)
for (i, 𝐲) in enumerate(predictions)
    plot!(𝐱, 𝐲; label="Fitted: \$x^{$(2^(i - 1))}\$", linestyle=:dot, PLOT_DEFAULTS...)
end
title!("Fitting with regularization")
savefig("fitting_regularization.png")

𝝷, 𝐲, predictions, residuals = fit_residuals0(𝐱, 𝐲̂, 4)

plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
hline!([0]; label="", seriescolor=:black, primary=false)
plot!(
    𝐱,
    target_fermi_dirac.(𝐱);
    z_order=:back,
    seriescolor=:maroon,
    label="Reference Fermi function",
    PLOT_DEFAULTS...,
)
plot!(𝐱, 𝐲; label="Total fitted result", linestyle=:dash, PLOT_DEFAULTS...)
for (i, 𝚫𝐲) in enumerate(predictions)
    plot!(𝐱, 𝚫𝐲; label="Correction: \$x^{$(2^(i - 1))}\$", linestyle=:dot, PLOT_DEFAULTS...)
end
title!("Fitting residuals without regularization")
savefig("residual_fitting_no_regularization.png")

𝝷, 𝐲, predictions, residuals = fit_residuals(𝐱, 𝐲̂, 4; λ₁=2.5, λ₂=2.5)

plt = plot()
xlims!(lower_bound, upper_bound)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
hline!([0]; label="", seriescolor=:black, primary=false)
plot!(
    𝐱,
    target_fermi_dirac.(𝐱);
    z_order=:back,
    seriescolor=:maroon,
    label="Reference Fermi function",
    PLOT_DEFAULTS...,
)
plot!(𝐱, 𝐲; label="Total fitted result", linestyle=:dash, PLOT_DEFAULTS...)
for (i, 𝚫𝐲) in enumerate(predictions)
    plot!(𝐱, 𝚫𝐲; label="Correction: \$x^{$(2^(i - 1))}\$", linestyle=:dot, PLOT_DEFAULTS...)
end
title!("Fitting residuals with regularization")
savefig("residual_fitting_regularization.png")
