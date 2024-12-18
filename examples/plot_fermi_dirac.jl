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

function plot_fermi_dirac(μ, β)
    minlayers = 14
    maxlayers = 16
    lower_bound, upper_bound = 0, 1

    branches = determine_branches(μ, minlayers)
    𝐱 = sample_by_pdf(bell_distribution(μ, β), μ, (lower_bound, upper_bound))
    𝐲 = forward_pass(branches, 𝐱)

    plt = plot(; layout=grid(2, 1; heights=(0.6, 0.4)))
    xlims!(lower_bound, upper_bound)
    xlabel!(raw"$\varepsilon\prime$")
    ylabel!(raw"$n(\varepsilon\prime)$"; subplot=1)
    ylabel!(raw"$\Delta n(\varepsilon\prime)$"; subplot=2)
    hline!([0]; subplot=2, label="Reference", z_order=:back, PLOT_DEFAULTS...)
    plot!(
        𝐱,
        fermi_dirac.(𝐱, μ, β);
        subplot=1,
        z_order=:back,
        label="Reference",
        PLOT_DEFAULTS...,
    )
    plot!(
        𝐱,
        oneunit.(𝐲) - 𝐲;
        subplot=1,
        label="SP2 with $minlayers layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    plot!(
        𝐱,
        symlog.(fermi_dirac.(𝐱, μ, β) - oneunit.(𝐲) + 𝐲);
        subplot=2,
        yformatter=symlogformatter,
        label="SP2 with $minlayers layers",
        linestyle=:dash,
        PLOT_DEFAULTS...,
    )
    for nlayers in minlayers:maxlayers
        𝛉 = fit_fermi_dirac(𝐱, μ, β, init_model(μ, nlayers); max_iter=10000).model
        plot!(
            𝐱,
            fermi_dirac(𝛉).(𝐱);
            subplot=1,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
        plot!(
            𝐱,
            symlog.(fermi_dirac.(𝐱, μ, β) - fermi_dirac(𝛉).(𝐱));
            subplot=2,
            yformatter=symlogformatter,
            label="$nlayers layers",
            linestyle=:dot,
            PLOT_DEFAULTS...,
        )
    end
    for nlayers in minlayers:maxlayers
        𝐱′ = chebyshevnodes_1st(length(𝐱), (lower_bound, upper_bound))
        𝛉 = fit_fermi_dirac(𝐱′, μ, β, init_model(μ, nlayers); max_iter=10000).model
        𝐲′ = fermi_dirac(𝛉).(𝐱′)
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
            symlog.(fermi_dirac.(𝐱′, μ, β) - 𝐲′);
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

function symlogformatter(y, n=-5)
    if sign(y) == 0
        raw"$0$"
    else
        s = sign(y) == 1 ? "" : "-"
        nexp = sign(y) * (abs(y) + n)
        if sign(y) == -1
            nexp = -nexp
        end
        '$' * s * "10^{$nexp}" * '$'
    end
end

μ = 0.568
β = 50
plt = plot_fermi_dirac(μ, β)
savefig(plt, "fd μ=$μ β=$β.png")
