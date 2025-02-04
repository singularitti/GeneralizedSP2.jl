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
    :legend_position => :bottomright,
    :background_color_inside => nothing,
    :color_palette => :tab10,
    :legendfontfamily => "Palatino Italic",
    :guidefontfamily => "Palatino Italic",
    :tickfontfamily => "Palatino Roman",
)

function heaviside(x, μ)
    if x < μ
        return 0
    elseif x == μ
        return 1 / 2
    else
        return 1
    end
end

μ = 0.4
𝐱 = 0:0.001:1
𝐲₀ = Base.Fix2(heaviside, μ).(𝐱)
plot(; layout=grid(2, 1; heights=(0.7, 0.3)))
for nlayers in 7:15
    branches = determine_branches(μ, nlayers)
    𝐲 = forward_pass(branches, 𝐱)
    plot!(𝐱, 𝐲; subplot=1, linestyle=:dash, label="I=" * string(nlayers), PLOT_DEFAULTS...)
    plot!(𝐱, 𝐲 - 𝐲₀; subplot=2, label="", linestyle=:dash, PLOT_DEFAULTS..., minorticks=2)
end
plot!(𝐱, 𝐲₀; subplot=1, linetype=:steppre, label="H(x - 0.4)", PLOT_DEFAULTS...)
hline!([0]; subplot=2, label="", seriescolor=:black, primary=false)
xlims!(0, 1)
xlabel!("x"; subplot=2)
ylabel!("y"; subplot=1)
ylabel!("y - H(x - 0.4)"; subplot=2)
savefig("sp2.png")
