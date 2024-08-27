using Plots

PLOT_DEFAULTS = Dict(
    :size => (450, 450),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 10,
    :guidefontsize => 10,
    :tickfontsize => 8,
    :legendfontsize => 7,
    :left_margin => (0, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :topleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

projection1(𝐱) = 𝐱 .^ 2

projection2(𝐱) = 2𝐱 .- 𝐱 .^ 2

𝐱 = 0:0.01:1

plot()
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝐱, projection2(𝐱); label=raw"$g(x) = 2x - x^2$", PLOT_DEFAULTS...)
plot!(𝐱, (projection1 ∘ projection2)(𝐱); label=raw"$f \circ g$", PLOT_DEFAULTS...)
plot!(𝐱, (projection2 ∘ projection1)(𝐱); label=raw"$g \circ f$", PLOT_DEFAULTS...)
plot!(𝐱, projection1(𝐱); label=raw"$f(x) = x^2$", PLOT_DEFAULTS...)
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
savefig("projections.pdf")

plot()
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝐱, projection2(𝐱); label=raw"$g(x) = 2x - x^2$", PLOT_DEFAULTS...)
plot!(𝐱, (projection1 ∘ projection2)(𝐱); label=raw"$f \circ g$", PLOT_DEFAULTS...)
plot!(
    𝐱,
    (projection2 ∘ projection1 ∘ projection2)(𝐱);
    label=raw"$g \circ f \circ g$",
    PLOT_DEFAULTS...,
)
plot!(
    𝐱,
    (projection1 ∘ projection2 ∘ projection1 ∘ projection2)(𝐱);
    label=raw"$f \circ g \circ f \circ g$",
    PLOT_DEFAULTS...,
)
plot!(
    𝐱,
    (projection2 ∘ projection1 ∘ projection2 ∘ projection1 ∘ projection2)(𝐱);
    label=raw"$g \circ f \circ g \circ f \circ g$",
    PLOT_DEFAULTS...,
)
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
savefig("projections_more.pdf")
