using ChairmarksExtras: @btimed
using DifferentiationInterface
using Enzyme
using GeneralizedSP2
using Mooncake
using OrderedCollections: OrderedDict
using Plots: plot, plot!, palette, xlims!, savefig
using Statistics: mean
using Zygote
using FiniteDiff

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :guidefontsize => 15,
    :tickfontsize => 12,
    :legendfontsize => 12,
    :left_margin => (8, :mm),
    :bottom_margin => (8, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :background_color_inside => nothing,
    :legendfontfamily => "Palatino Italic",
    :guidefontfamily => "Palatino Italic",
    :tickfontfamily => "Palatino Roman",
)

β′ = 60
μ′ = 0.568

𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (0, 1))
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)

layers = 12:20
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
strategy = Manual()
strategy = Auto(AutoEnzyme(; mode=Reverse, function_annotation=Const))
strategy = Auto(AutoFiniteDiff())
strategy = Default()

all_results = OrderedDict()

results = map(
    Iterators.takewhile(
        max_iter ->
            (isa(strategy, Auto) || isa(strategy, Default) && max_iter < 1e7) ||
                isa(strategy, Manual),
        max_iters,
    ),
) do max_iter
    timed_results = map(layers) do nlayers
        println("fitting for max_iter = $max_iter, nlayers = $nlayers, strategy = $strategy")
        model_init = init_model(μ′, nlayers)
        result = @btimed fit_fermi_dirac(
            𝛆′, μ′, β′, model_init; maxiters=max_iter, diff=strategy
        ) samples = 1 evals = 1
        result.value, result.time, result.bytes
    end
    models = map(timed_results) do timed_result
        timed_result[1].model
    end
    times = map(timed_results) do timed_result
        timed_result[2]  # In seconds
    end
    bytes = map(timed_results) do timed_result
        timed_result[3]
    end
    rmse = map(models) do model
        𝐲_fitted = fermi_dirac(model).(𝛆′)
        residuals = 𝐲_fitted - 𝐲̂
        sqrt(mean(abs2, residuals))
    end
    (rmse=rmse, times=times, bytes=bytes)
end
all_results[string(strategy)] = results

benchtime = [result.times for result in all_results[string(Manual())]]

plot(; layout=(1, 2), PLOT_DEFAULTS..., size=(1300, 500))
for (strategy, strategy_str, linestyle, markershape) in zip(
    keys(all_results),
    ("opt", "Ez", "FD", "none"),
    (:solid, :dash, :dot, :dashdot, :dashdotdot),
    (:circle, :diamond, :cross, :star4, :star5),
)
    results = all_results[strategy]
    time = [result.times for result in results]  # In seconds
    memory = [result.bytes for result in results] / 1024^2  # In MB
    rmse = [result.rmse for result in results]
    for (iterindex, seriescolor) in zip(1:length(results), palette(:seaborn_bright))
        plot!(
            layers,
            rmse[iterindex];
            subplot=1,
            seriestype=:path,
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label="I=$(max_iters[iterindex]), $strategy_str",
            yscale=:log10,
            xticks=layers,
            yticks=exp10.((-9):(-3)),
            xminorticks=0,
            yminorticks=5,
            legend_position=:bottomleft,
            xlabel="L",
            ylabel="RMSE of fitting",
            yguidefontfamily="Palatino Roman",
            PLOT_DEFAULTS...,
        )
        plot!(
            layers,
            time[iterindex];
            subplot=2,
            seriestype=:path,
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label="I=$(max_iters[iterindex]), $strategy_str",
            yscale=:log10,
            xticks=layers,
            yticks=exp10.(-2:3),
            xminorticks=0,
            yminorticks=5,
            legend_position=:topleft,
            xlabel="L",
            ylabel="time (s)",
            yguidefontfamily="Palatino Roman",
            PLOT_DEFAULTS...,
        )
        # plot!(
        #     layers,
        #     time[iterindex] ./ benchtime[iterindex];
        #     subplot=3,
        #     seriestype=:path,
        #     linestyle=linestyle,
        #     markershape=markershape,
        #     seriescolor=seriescolor,
        #     label="I=$(max_iters[iterindex]), $strategy_str",
        #     yscale=:log10,
        #     xticks=layers,
        #     yticks=exp10.(-1:10),
        #     xminorticks=0,
        #     yminorticks=5,
        #     xlabel="number of layers",
        #     ylabel="memory (MB)",
        #     PLOT_DEFAULTS...,
        # )
    end
end
xlims!(extrema(layers))
savefig("benchmarks.png")
