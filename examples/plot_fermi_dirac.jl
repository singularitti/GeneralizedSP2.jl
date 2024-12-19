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

function plot_fermi_dirac(μ′, β′)
    minlayers = 20
    maxlayers = 22
    lower_bound, upper_bound = 0, 1

    branches = determine_branches(μ′, 12)
    𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (lower_bound, upper_bound))
    𝐲 = forward_pass(branches, 𝛆′)

    plt = plot(; layout=grid(2, 1; heights=(0.5, 0.5)))
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$\varepsilon\prime$")
    ylabel!(raw"$n(\varepsilon\prime)$"; subplot=1)
    ylabel!(raw"$\Delta n(\varepsilon\prime)$"; subplot=2)
    hline!([0]; subplot=2, label="Reference", z_order=:back, PLOT_DEFAULTS...)
    plot!(
        𝛆′,
        fermi_dirac.(𝛆′, μ′, β′);
        subplot=1,
        z_order=:back,
        label="Reference",
        PLOT_DEFAULTS...,
    )
    plot!(
        𝛆′,
        oneunit.(𝐲) - 𝐲;
        subplot=1,
        label="SP2 with 12 layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    plot!(
        𝛆′,
        symlog.(fermi_dirac.(𝛆′, μ′, β′) - oneunit.(𝐲) + 𝐲);
        subplot=2,
        yformatter=symlogformatter,
        label="SP2 with 12 layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    for nlayers in minlayers:maxlayers
        model = fit_fermi_dirac(𝛆′, μ′, β′, init_model(μ′, nlayers); max_iter=10000).model
        plot!(
            𝛆′,
            fermi_dirac(model).(𝛆′);
            subplot=1,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝛆′,
            symlog.(fermi_dirac.(𝛆′, μ′, β′) - fermi_dirac(model).(𝛆′));
            subplot=2,
            yformatter=symlogformatter,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
    end
    for nlayers in minlayers:maxlayers
        𝐱′ = chebyshevnodes_1st(length(𝛆′), (lower_bound, upper_bound))
        model = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, nlayers); max_iter=10000).model
        𝐲′ = fermi_dirac(model).(𝐱′)
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
            symlog.(fermi_dirac.(𝐱′, μ′, β′) - 𝐲′);
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
symlog(y, n=-7) = sign(y) * (log10(1 + abs(y) / (10.0^n)))

function symlogformatter(z, n=-7)
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
plt = plot_fermi_dirac(μ′, β′)
savefig(plt, "fd μ=$μ′ β=$β′.pdf")
