using GeneralizedSP2
using Plots

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 15,
    :guidefontsize => 15,
    :tickfontsize => 12,
    :legendfontsize => 12,
    :left_margin => (2.5, :mm),
    :bottom_margin => (2.5, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomright,
    :background_color_inside => nothing,
    :color_palette => :seaborn_bright,
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
layers = 7:15
layout = @layout [a{0.6h}; b{0.4h}]
plot(; size=(800, 600), layout=layout)
hline!([1 / 2]; subplot=1, linewidth=1, label="", seriescolor=:black, primary=false)
for nlayers in layers
    branches = determine_branches(μ, nlayers)
    𝐲 = forward_pass(branches, 𝐱)
    μᵢ = backward_pass(branches)[1]
    plot!(𝐱, 𝐲; subplot=1, linestyle=:dash, label="I=" * string(nlayers), PLOT_DEFAULTS...)
    plot!(𝐱, 𝐲 - 𝐲₀; subplot=2, label="", linestyle=:dash, PLOT_DEFAULTS..., yminorticks=2)
end
plot!(𝐱, 𝐲₀; subplot=1, linetype=:steppre, label="H(x - $μ)", PLOT_DEFAULTS...)
hline!([0]; subplot=2, linewidth=1, label="", seriescolor=:black, primary=false)
xlims!(0, 1; subplot=1)
xlims!(0, 1; subplot=2)
xlabel!("x"; subplot=2)
ylabel!("y"; subplot=1)
ylabel!("y - H(x - $μ)"; subplot=2)
savefig("sp2.png")
