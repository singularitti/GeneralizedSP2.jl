using GeneralizedSP2
using Plots
using ProgressMeter: @showprogress
using Statistics: mean
using BenchmarkTools

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.
PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 3,
    :markerstrokealpha => 0,
    :markerstrokewidth => 0,
    :titlefontsize => 8,
    :plot_titlefontsize => 8,
    :guidefontsize => 7,
    :tickfontsize => 6,
    :legendfontsize => 6,
    :left_margin => (8, :mm),
    :bottom_margin => (6, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

β′ = 100
μ′ = 0.4

𝐱′ = reverse(chebyshevnodes_1st(400, (0, 1)))  # Have to reverse since β′ is negative
𝐲̂ = fermi_dirac.(𝐱′, μ′, β′)

layers = 10:21
max_iters = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

results = map(max_iters) do max_iter
    println("fitting for max_iter = $max_iter")
    timed_results = @showprogress map(layers) do nlayers
        @timed fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, nlayers); max_iter=max_iter)
    end
    𝚯 = map(timed_results) do timed_result
        first(timed_result.value)
    end
    times = map(timed_results) do timed_result
        timed_result.time
    end
    𝐲_fitted = map(𝚯) do 𝛉
        fermi_dirac_model(𝐱′, 𝛉)
    end
    rmse = map(𝚯, 𝐲_fitted) do 𝛉, 𝐲
        residuals = 𝐲 - 𝐲̂
        sqrt(mean(abs2, residuals))
    end
    (rmse=rmse, times=times)
end

time_matrix = hcat([result.times for result in results]...)
rmse_matrix = hcat([result.rmse for result in results]...)

layout = (1, 2)
plot(; layout=layout, PLOT_DEFAULTS..., size=(3200 / 3, 400))
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
