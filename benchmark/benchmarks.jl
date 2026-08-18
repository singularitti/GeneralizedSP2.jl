using ChairmarksExtras: @btimed
using CSV
using DataFrames: DataFrame
using DifferentiationInterface
using Enzyme
using FiniteDiff
using GeneralizedSP2
using Mooncake
using OrderedCollections: OrderedDict
using Plots: plot, plot!, palette, xlims!, savefig
using Statistics: mean

β′ = 60
μ′ = 0.568

𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (0, 1))
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)

layers = 12:20
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

backend_specs = (
    (name = "opt", diff = Manual()),
    (
        name = "Enzyme",
        diff = Auto(AutoEnzyme(; mode = Reverse, function_annotation = Const)),
    ),
    (name = "FiniteDiff", diff = Auto(AutoFiniteDiff())),
    (name = "Mooncake", diff = Auto(AutoMooncake(; config = nothing))),
)

all_results = OrderedDict{String, Any}()

for spec in backend_specs
    results = map(max_iters) do max_iter
        timed_results = map(layers) do nlayers
            println(
                "fitting for backend = $(spec.name), max_iter = $max_iter, nlayers = $nlayers",
            )
            model_init = init_model(μ′, nlayers)
            result = @btimed fit_fermi_dirac(
                𝛆′,
                μ′,
                β′,
                model_init;
                maxiters = max_iter,
                diff = spec.diff,
            ) samples = 1 evals = 1
            model = result.value.model
            𝐲_fitted = fermi_dirac(model).(𝛆′)
            residuals = 𝐲_fitted - 𝐲̂
            rmse = sqrt(mean(abs2, residuals))
            return (
                model = model,
                rmse = rmse,
                time = result.time,
                bytes = result.bytes,
                trace_entries = length(result.value.trace),
            )
        end
        return (
            rmse = [result.rmse for result in timed_results],
            times = [result.time for result in timed_results],
            bytes = [result.bytes for result in timed_results],
            trace_entries = [result.trace_entries for result in timed_results],
        )
    end
    all_results[spec.name] = results
end

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
plot_styles = Dict(
    "opt" => (linestyle = :solid, markershape = :circle),
    "Enzyme" => (linestyle = :dash, markershape = :diamond),
    "FiniteDiff" => (linestyle = :dot, markershape = :cross),
    "Mooncake" => (linestyle = :dashdot, markershape = :star4),
)

figure = plot(; layout = (1, 2), PLOT_DEFAULTS..., size = (1300, 500))
for spec in backend_specs
    results = all_results[spec.name]
    style = plot_styles[spec.name]
    time = [result.times for result in results] # In seconds
    rmse = [result.rmse for result in results]
    for (iterindex, seriescolor) in zip(1:length(results), palette(:seaborn_bright))
        plot!(
            figure,
            layers,
            rmse[iterindex];
            subplot = 1,
            seriestype = :path,
            linestyle = style.linestyle,
            markershape = style.markershape,
            seriescolor = seriescolor,
            label = "I=$(max_iters[iterindex]), $(spec.name)",
            yscale = :log10,
            xticks = layers,
            yticks = exp10.((-9):(-3)),
            xminorticks = 0,
            yminorticks = 5,
            legend_position = :bottomleft,
            xlabel = "L",
            ylabel = "RMSE of fitting",
            yguidefontfamily = "Palatino Roman",
            PLOT_DEFAULTS...,
        )
        plot!(
            figure,
            layers,
            time[iterindex];
            subplot = 2,
            seriestype = :path,
            linestyle = style.linestyle,
            markershape = style.markershape,
            seriescolor = seriescolor,
            label = "I=$(max_iters[iterindex]), $(spec.name)",
            yscale = :log10,
            xticks = layers,
            yticks = exp10.(-2:3),
            xminorticks = 0,
            yminorticks = 5,
            legend_position = :topleft,
            xlabel = "L",
            ylabel = "time (s)",
            yguidefontfamily = "Palatino Roman",
            PLOT_DEFAULTS...,
        )
    end
end
xlims!(figure, extrema(layers))

savefig(figure, "benchmarks.pdf")
savefig(figure, "benchmarks.png")

results_table = DataFrame(
    backend = String[],
    max_iter = Int[],
    layers = Int[],
    rmse = Float64[],
    benchmark_seconds = Float64[],
    bytes = Float64[],
    trace_entries = Int[],
)
for spec in backend_specs
    for (iterindex, max_iter) in enumerate(max_iters)
        results = all_results[spec.name][iterindex]
        for (layerindex, nlayers) in enumerate(layers)
            push!(
                results_table,
                (
                    spec.name,
                    max_iter,
                    nlayers,
                    results.rmse[layerindex],
                    results.times[layerindex],
                    results.bytes[layerindex],
                    results.trace_entries[layerindex],
                ),
            )
        end
    end
end
CSV.write("benchmarks.csv", results_table)
