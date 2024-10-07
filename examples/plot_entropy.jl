using GeneralizedSP2
using GeneralizedSP2: transform_entropy
using Plots

PLOT_DEFAULTS = Dict(
    :size => (450, 600),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
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
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function plot_entropy(β, μ=0.568)
    calculate_entropy(𝐱, 𝝷ₛ) = transform_entropy.(iterate_heaviside(𝐱, 𝝷ₛ))

    minlayers = 2
    maxlayers = 4
    lower_bound, upper_bound = 0, 1

    𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))

    plt = plot(; layout=grid(2, 1; heights=(0.6, 0.4)))
    plot!(; subplot=1, title="My fitted results μ=$μ, β=$β")
    plot!(; subplot=2, title="Error of the approximation")
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$x$")
    ylabel!(raw"$S$")
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
        _, 𝝷ₛ = fit_model(𝐱, μ, β, nlayers)
        plot!(
            𝐱,
            calculate_entropy(𝐱, 𝝷ₛ);
            subplot=1,
            label="MLSP2 with $nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱,
            electronic_entropy.(𝐱, μ, β) - calculate_entropy(𝐱, 𝝷ₛ);
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
