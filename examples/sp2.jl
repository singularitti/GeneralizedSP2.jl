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
    :fontfamily => "Palatino Roman",
)

𝐱 = chebyshevnodes_1st(1000, (0, 1))

plot()
hline!([1 / 2]; label="", seriescolor=:black, primary=false)
for nlayers in 4:1:13
    branches = determine_branches(1 / 2, nlayers)
    𝐲 = forward_pass(branches, 𝐱)
    plot!(𝐱, 𝐲; label="L=" * string(nlayers), PLOT_DEFAULTS...)
end
xlims!(0, 1)
xlabel!(raw"$x$")
ylabel!(raw"$y$")
savefig("sp2.pdf")
