using GeneralizedSP2
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

f(𝐱) = 𝐱 .^ 2

g(𝐱) = 2𝐱 .- 𝐱 .^ 2

𝐱 = chebyshevnodes_1st(1000, (0, 1))

plot()
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝐱, g(𝐱); label=raw"$g(x) = 2x - x^2$", PLOT_DEFAULTS...)
plot!(𝐱, (f ∘ g)(𝐱); label=raw"$f \circ g$", PLOT_DEFAULTS...)
plot!(𝐱, (g ∘ f)(𝐱); label=raw"$g \circ f$", PLOT_DEFAULTS...)
plot!(𝐱, f(𝐱); label=raw"$f(x) = x^2$", PLOT_DEFAULTS...)
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
savefig("projections.pdf")

plot()
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
plot!(𝐱, g(𝐱); label=raw"$g(x) = 2x - x^2$", PLOT_DEFAULTS...)
plot!(𝐱, (f ∘ g)(𝐱); label=raw"$f \circ g$", PLOT_DEFAULTS...)
plot!(𝐱, (g ∘ f ∘ g)(𝐱); label=raw"$g \circ f \circ g$", PLOT_DEFAULTS...)
plot!(𝐱, (f ∘ g ∘ f ∘ g)(𝐱); label=raw"$f \circ g \circ f \circ g$", PLOT_DEFAULTS...)
plot!(
    𝐱,
    (g ∘ f ∘ g ∘ f ∘ g)(𝐱);
    label=raw"$g \circ f \circ g \circ f \circ g$",
    PLOT_DEFAULTS...,
)
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
savefig("projections_more.pdf")

plot()
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
projections = Iterators.accumulate(∘, map(i -> iseven(i) ? f : g, 1:6))
animation = @animate for projection in projections
    plot!(
        𝐱,
        projection.(𝐱);
        label=string(repr(projection; context=:module => Main)),  # See https://discourse.julialang.org/t/122702/2
        PLOT_DEFAULTS...,
        legend_position=:bottomright,
    )
end
gif(animation, "animation.gif"; fps=2)
