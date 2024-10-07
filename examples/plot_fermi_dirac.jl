using GeneralizedSP2
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

function plot_fermi_dirac(β, μ=0.568)
    minlayers = 2
    maxlayers = 4
    lower_bound, upper_bound = 0, 1

    branches = determine_branches(μ, maxlayers)
    𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))
    𝐲 = forward_pass(branches, 𝐱)

    plt = plot(; layout=grid(2, 1; heights=(0.6, 0.4)))
    plot!(; subplot=1, title="My fitted results μ=$μ, β=$β")
    plot!(; subplot=2, title="Error of the approximation")
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$x$")
    ylabel!(raw"$y$")
    hline!([1 / 2]; subplot=1, label="", seriescolor=:black, primary=false)
    hline!([0]; subplot=2, label="", seriescolor=:black, primary=false)
    plot!(
        𝐱,
        fermi_dirac.(𝐱, μ, β);
        primary=false,
        z_order=:back,
        seriescolor=:maroon,
        subplot=1,
        label="Reference Fermi function",
        PLOT_DEFAULTS...,
    )
    plot!(
        𝐱,
        oneunit.(𝐲) - 𝐲;
        subplot=1,
        label="SP2 with $maxlayers layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    plot!(
        𝐱,
        fermi_dirac.(𝐱, μ, β) - oneunit.(𝐲) + 𝐲;
        subplot=2,
        label="SP2 with $maxlayers layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    for nlayers in minlayers:maxlayers
        𝝷FD, _ = fit_model(𝐱, μ, β, nlayers)
        plot!(
            𝐱,
            iterate_fermi_dirac(𝐱, 𝝷FD);
            subplot=1,
            label="MLSP2 with $nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱,
            fermi_dirac.(𝐱, μ, β) - iterate_fermi_dirac(𝐱, 𝝷FD);
            subplot=2,
            label="MLSP2 with $nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
    end
    for nlayers in minlayers:maxlayers
        𝐱′ = chebyshevnodes_1st(length(𝐱), (lower_bound, upper_bound))
        𝝷FD, 𝝷ₛ = fit_model(𝐱′, μ, β, nlayers)
        𝐲′ = iterate_fermi_dirac(𝐱′, 𝝷FD)
        plot!(
            𝐱′,
            𝐲′;
            subplot=1,
            label="MLSP2 with $nlayers layers by Chebyshev nodes",
            linestyle=:dashdot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱′,
            fermi_dirac.(𝐱′, μ, β) - 𝐲′;
            subplot=2,
            label="$nlayers layers by Chebyshev nodes",
            linestyle=:dashdot,
            PLOT_DEFAULTS...,
        )
    end
    return savefig("fits_beta=$β,nlayers=$maxlayers.png")
end

plot_fermi_dirac(9.423)
plot_fermi_dirac(20)
