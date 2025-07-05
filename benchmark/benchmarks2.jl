using ChairmarksExtras: @btimed
using GeneralizedSP2
using OrderedCollections: OrderedDict
using Plots: plot, plot!, palette, xlims!, savefig
using Printf: @sprintf
using Statistics: mean
using Serialization: serialize

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :grid => nothing,
    :left_margin => (8, :mm),
    :bottom_margin => (8, :mm),
    :linewidth => 2,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :xformatter => x -> x < 0.1 ? string(x) : @sprintf("%0.0f", x),
    :guidefontsize => 15,
    :tickfontsize => 12,
    :legendfontsize => 12,
    :legend_position => :bottomright,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :background_color_inside => nothing,
    :legendfontfamily => "Palatino Italic",
    :xguidefontfamily => "Palatino Italic",
    :yguidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
)

μ′ = 0.6
β′_vals = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 50, 60, 80, 100, 200, 400, 800, 1600]
layers = 18:20
max_iters = [1_000_000, 10_000_000]
strategy = Manual()
𝛆′ = chebyshevnodes_1st(350, (0, 1))

all_results = OrderedDict{}()
all_models = OrderedDict{}()
for β′ in β′_vals
    𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)
    for max_iter in max_iters
        for nlayers in layers
            println(
                "fitting for β′ = $β′, max_iter = $max_iter, nlayers = $nlayers, strategy = $strategy",
            )
            model_init = init_model(μ′, nlayers)
            result = @btimed fit_fermi_dirac(
                𝛆′, μ′, β′, model_init; max_iter=max_iter, diff=strategy
            ) samples = 3 evals = 1
            model = result.value.model
            time = result.time
            bytes = result.bytes
            𝐲_fitted = fermi_dirac(model).(𝛆′)
            residuals = 𝐲_fitted - 𝐲̂
            rmse = sqrt(mean(abs2, residuals))
            all_results[(β′, max_iter, nlayers)] = (rmse=rmse, time=time, bytes=bytes)
            all_models[(β′, max_iter, nlayers)] = model
            serialize("table.jls", all_results)
            serialize("all_models.jls", all_models)
        end
    end
end

colors = palette(:seaborn_bright)
linestyles = Dict(1_000_000 => :solid, 10_000_000 => :dashdot, 100_000_000 => :dot)
markers = Dict(1_000_000 => :circle, 10_000_000 => :star5, 100_000_000 => :cross)
plot(; layout=(1, 2), PLOT_DEFAULTS..., size=(1300, 500))
for (i, nlayers) in enumerate(layers)
    for (j, max_iter) in enumerate(max_iters)
        # Gather RMSE and time for each β′
        rmse = [all_results[(β′, max_iter, nlayers)].rmse for β′ in β′_vals]
        time = [all_results[(β′, max_iter, nlayers)].time for β′ in β′_vals]
        label = "L=$(nlayers), I=$(max_iter)"
        linestyle = get(linestyles, max_iter, :solid)
        markershape = get(markers, max_iter, :circle)
        seriescolor = colors[i]
        plot!(
            β′_vals,
            rmse;
            subplot=1,
            seriestype=:path,
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label=label,
            scale=:log10,
            xticks=β′_vals,
            xlabel=raw"β′",
            ylabel=raw"RMSE of fitting",
            PLOT_DEFAULTS...,
        )
        plot!(
            β′_vals,
            time;
            subplot=2,
            seriestype=:path,
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label=label,
            scale=:log10,
            xticks=β′_vals,
            xlabel=raw"β′",
            ylabel=raw"time (s)",
            PLOT_DEFAULTS...,
        )
    end
end
savefig("benchmarks_by_beta.png")
