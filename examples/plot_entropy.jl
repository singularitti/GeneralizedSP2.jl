using GeneralizedSP2
using Plots

PLOT_DEFAULTS = Dict(
    :size => (450, 600),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 2,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 8,
    :tickfontsize => 6,
    :legendfontsize => 6,
    :left_margin => (1, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function plot_entropy(β, μ=0.568)
    minlayers = 12
    maxlayers = 14
    lower_bound, upper_bound = 0, 1

    𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))

    plt = plot(; layout=grid(2, 1; heights=(0.5, 0.5)))
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$\varepsilon\prime$")
    ylabel!(raw"$S(\varepsilon\prime)$"; subplot=1)
    ylabel!(raw"$\Delta S(\varepsilon\prime)$"; subplot=2)
    hline!([0]; subplot=2, label="", seriescolor=:black, primary=false)
    plot!(
        𝐱,
        electronic_entropy.(𝐱, μ, β);
        primary=false,
        z_order=:back,
        seriescolor=:maroon,
        subplot=1,
        label="Reference entropy",
        PLOT_DEFAULTS...,
    )
    for nlayers in minlayers:maxlayers
        𝛉 = fit_electronic_entropy(𝐱, μ, β, init_model(μ, nlayers); max_iter=100000).model
        plot!(
            𝐱,
            electronic_entropy(𝛉).(𝐱);
            subplot=1,
            label="MLSP2 with $nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱,
            electronic_entropy.(𝐱, μ, β) - electronic_entropy(𝛉).(𝐱);
            subplot=2,
            label="MLSP2 with $nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
    end
    savefig("fits_beta=$β,nlayers=$maxlayers.png")
    return plt
end

plot_entropy(9.423)
plot_entropy(20)
