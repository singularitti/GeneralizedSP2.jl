using Chairmarks
using BenchmarkTools
using DifferentiationInterface
using Enzyme
using GeneralizedSP2
using Mooncake
using Plots: plot, plot!, xlims!, savefig
using Statistics: mean

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 8,
    :left_margin => (8, :mm),
    :bottom_margin => (8, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomright,
    :background_color_inside => nothing,
    :color_palette => :tab10,
    :legendfontfamily => "Palatino Italic",
    :guidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
)

β′ = 60
μ′ = 0.568

𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (0, 1))
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)

layers = 12:20
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
strategy = Auto(AutoEnzyme(; mode=Reverse, function_annotation=Const))
strategy = Manual()

all_results = Dict()

results = map(max_iters) do max_iter
    timed_results = map(layers) do nlayers
        println("fitting for max_iter = $max_iter, nlayers = $nlayers, strategy = $strategy")
        model_init = init_model(μ′, nlayers)
        result = Ref{Any}()
        benchmark = @b _ fit_fermi_dirac(
            $𝛆′, $μ′, $β′, $model_init; max_iter=$max_iter, diff=strategy
        ) result[] = _ samples = 1 evals = 1
        value = result[]
        value, benchmark.time, benchmark.bytes
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
all_results[strategy] = results

time_matrix = hcat([result.times for result in results]...)  # In seconds
mem_matrix = hcat([result.bytes for result in results]...) / 1024^2  # In MB
rmse_matrix = hcat([result.rmse for result in results]...)

layout = (1, 3)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1800, 480))
plot!(
    layers,
    rmse_matrix;
    subplot=1,
    label=hcat(("I=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    yticks=exp10.((-9):(-3)),
    xminorticks=0,
    yminorticks=5,
    xlabel="number of layers",
    ylabel="RMSE of fitting",
    PLOT_DEFAULTS...,
    legend_position=:bottomleft,
)
plot!(
    layers,
    time_matrix;
    subplot=2,
    label=hcat(("I=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    yticks=exp10.(-2:3),
    xminorticks=0,
    yminorticks=5,
    xlabel="number of layers",
    ylabel="time (s)",
    PLOT_DEFAULTS...,
    legend_position=:topleft,
)
plot!(
    layers,
    time_matrix;
    subplot=3,
    label=hcat(("I=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    yticks=exp10.(-1:10),
    xminorticks=0,
    yminorticks=5,
    xlabel="number of layers",
    ylabel="memory (MB)",
    PLOT_DEFAULTS...,
    legend_position=:topleft,
)
xlims!(extrema(layers))
savefig("benchmarks.png")
