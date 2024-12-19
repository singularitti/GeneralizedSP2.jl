using GeneralizedSP2
using Plots
using ProgressMeter: @showprogress
using Statistics: mean
using BenchmarkTools: BenchmarkTools, BenchmarkGroup, prunekwargs, hasevals, @benchmarkable

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# See https://discourse.julialang.org/t/62644/9 & https://github.com/JuliaCI/BenchmarkTools.jl/blob/v1.5.0/src/execution.jl#L658-L686
@eval BenchmarkTools macro btimed(args...)
    _, params = prunekwargs(args...)
    bench, trial, result = gensym(), gensym(), gensym()
    trialmin = gensym()
    tune_phase = hasevals(params) ? :() : :($BenchmarkTools.tune!($bench))
    return esc(
        quote
            local $bench = $BenchmarkTools.@benchmarkable $(args...)
            $tune_phase
            local $trial, $result = $BenchmarkTools.run_result(
                $bench; warmup=$(hasevals(params))
            )
            local $trialmin = $BenchmarkTools.minimum($trial)
            $result, $BenchmarkTools.time($trialmin)
        end,
    )
end

# Write your benchmarks here.
PLOT_DEFAULTS = Dict(
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

β′ = 60
μ′ = 0.568

𝛆′ = sample_by_pdf(bell_distribution(μ′, β′), μ′, (0, 1))
𝐲̂ = fermi_dirac.(𝛆′, μ′, β′)

layers = 10:21
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

results = map(max_iters) do max_iter
    println("fitting for max_iter = $max_iter")
    timed_results = @showprogress map(layers) do nlayers
        model_init = init_model(μ′, nlayers)
        value, time = BenchmarkTools.@btimed fit_fermi_dirac(
            $𝛆′, $μ′, $β′, $model_init; max_iter=$max_iter
        )
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

time_matrix = hcat([result.times for result in results]...)
rmse_matrix = hcat([result.rmse for result in results]...)

layout = (1, 2)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 450))
plot!(
    layers,
    rmse_matrix;
    subplot=1,
    label=hcat(("max iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    xlabel=raw"number of layers $L$",
    ylabel="RMSE of fitting",
    PLOT_DEFAULTS...,
    legend_position=:topright,
)
plot!(
    layers,
    time_matrix;
    subplot=2,
    label=hcat(("max iter=$max_iter" for max_iter in max_iters)...),
    yscale=:log10,
    xticks=layers,
    xlabel=raw"number of layers $L$",
    ylabel="time (s)",
    PLOT_DEFAULTS...,
    legend_position=:topleft,
)
