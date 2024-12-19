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
    :legendfontsize => 8,
    :left_margin => (1, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

function plot_entropy(μ′, β′)
    minlayers = 20
    maxlayers = 22
    lower_bound, upper_bound = 0, 1

    𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (lower_bound, upper_bound))

    plt = plot(; layout=grid(2, 1; heights=(0.5, 0.5)))
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$\varepsilon\prime$")
    ylabel!(raw"$S(\varepsilon\prime)$"; subplot=1)
    ylabel!(raw"$\Delta S(\varepsilon\prime)$"; subplot=2)
    hline!([0]; subplot=2, label="Reference", z_order=:back, PLOT_DEFAULTS...)
    plot!(
        𝛆′,
        electronic_entropy.(𝛆′, μ′, β′);
        subplot=1,
        z_order=:back,
        label="Reference",
        PLOT_DEFAULTS...,
    )
    for nlayers in minlayers:maxlayers
        model =
            fit_electronic_entropy(
                𝛆′, μ′, β′, init_model(μ′, nlayers); max_iter=10_000_000
            ).model
        plot!(
            𝛆′,
            electronic_entropy(model).(𝛆′);
            subplot=1,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝛆′,
            symlog.(electronic_entropy.(𝛆′, μ′, β′) - electronic_entropy(model).(𝛆′));
            subplot=2,
            yformatter=symlogformatter,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
    end
    for nlayers in minlayers:maxlayers
        𝐱′ = chebyshevnodes_1st(length(𝛆′), (lower_bound, upper_bound))
        model =
            fit_electronic_entropy(
                𝐱′, μ′, β′, init_model(μ′, nlayers); max_iter=10_000_000
            ).model
        𝐲′ = electronic_entropy(model).(𝐱′)
        plot!(
            𝐱′,
            𝐲′;
            subplot=1,
            label="$nlayers layers (Chebyshev)",
            linestyle=:dashdot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱′,
            symlog.(electronic_entropy.(𝐱′, μ′, β′) - 𝐲′);
            subplot=2,
            yformatter=symlogformatter,
            label="$nlayers layers (Chebyshev)",
            linestyle=:dashdot,
            PLOT_DEFAULTS...,
        )
    end
    return plt
end

# See https://discourse.julialang.org/t/26455 & https://discourse.julialang.org/t/45709/3
symlog(y, n=-5) = sign(y) * (log10(1 + abs(y) / (10.0^n)))

function symlogformatter(z, n=-5)
    if z == 0  # Handle the case when the transformed value is 0
        return "0"
    else
        s = z > 0 ? "" : "-"
        # Reverse the symlog transformation to find the original y
        abs_y = (10.0^abs(z) - 1) * 10.0^n
        return s * string(round(abs_y; digits=5))  # Format as a rounded number
    end
end

μ′ = 0.568
β′ = 60
plt = plot_entropy(μ′, β′)
savefig(plt, "S μ=$μ′ β=$β′.pdf")
