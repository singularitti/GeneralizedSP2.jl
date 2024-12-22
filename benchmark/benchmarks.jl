using GeneralizedSP2
using Plots
using ProgressMeter: @showprogress
using Statistics: mean
using BenchmarkTools: BenchmarkTools, BenchmarkGroup, prunekwargs, hasevals, @btimed

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.
PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 2,
    :markersize => 2,
    :markerstrokewidth => 0,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 8,
    :tickfontsize => 6,
    :legendfontsize => 8,
    :bottom_margin => (2, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

β′ = 60
μ′ = 0.568

𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (0, 1))
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)

layers = 12:20
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

results = map(max_iters) do max_iter
    println("fitting for max_iter = $max_iter")
    timed_results = @showprogress map(layers) do nlayers
        model_init = init_model(μ′, nlayers)
        value, time = @btimed fit_fermi_dirac(
            $𝛆′, $μ′, $β′, $model_init; max_iter=$max_iter
        ) samples = 1
        (value=value, time=time)
    end
    models = map(timed_results) do timed_result
        timed_result.value.model
    end
    times = map(timed_results) do timed_result
        timed_result.time
    end
    rmse = map(models) do model
        𝐲_fitted = fermi_dirac(model).(𝛆′)
        residuals = 𝐲_fitted - 𝐲̂
        sqrt(mean(abs2, residuals))
    end
    (rmse=rmse, times=times)
end

time_matrix = hcat([result.times for result in results]...) / 1e6  # Default units are in nanoseconds
rmse_matrix = hcat([result.rmse for result in results]...)

layout = (1, 2)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 450))
plot!(
    layers,
    rmse_matrix;
    subplot=1,
    label=hcat(("# fitting iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    yticks=exp10.((-9):(-3)),
    xminorticks=0,
    yminorticks=5,
    xlabel=raw"number of layers $L$",
    ylabel="RMSE of fitting",
    PLOT_DEFAULTS...,
    left_margin=(3, :mm),
    legend_position=:bottomleft,
)
plot!(
    layers,
    time_matrix;
    subplot=2,
    label=hcat(("# fitting iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    yticks=exp10.(0:6),
    xminorticks=0,
    yminorticks=5,
    xlabel=raw"number of layers $L$",
    ylabel="time (ms)",
    PLOT_DEFAULTS...,
    left_margin=(1, :mm),
    legend_position=:topleft,
)
xlims!(12, 19)
savefig("benchmarks.pdf")
