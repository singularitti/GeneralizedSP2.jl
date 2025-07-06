using ChairmarksExtras: @btimed
using GeneralizedSP2
using OrderedCollections: OrderedDict
using Plots: plot, plot!, palette, xticks!, xlabel!, yticks!, ylabel!, savefig
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
    :xformatter => x -> if x < 0.1
        @sprintf("%0.2f", x)
    else
        x > 1 ? @sprintf("%0.0f", x) : @sprintf("%0.1f", x)
    end,
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
linestyles = Dict(1_000_000 => :solid, 10_000_000 => :dot, 100_000_000 => :dash)
markers = Dict(1_000_000 => :circle, 10_000_000 => :diamond, 100_000_000 => :cross)
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
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label=label,
            scale=:log10,
            xrotation=90,
            PLOT_DEFAULTS...,
            xtickfontsize=7,
        )
        plot!(
            β′_vals,
            time;
            subplot=2,
            linestyle=linestyle,
            markershape=markershape,
            seriescolor=seriescolor,
            label=label,
            scale=:log10,
            xrotation=90,
            PLOT_DEFAULTS...,
            xtickfontsize=7,
        )
    end
end
xticks!(β′_vals)
yticks!(exp10.(-11:-7); subplot=1)
yticks!(exp10.(0:3); subplot=2)
xlabel!(raw"β′")
ylabel!(raw"RMSE of fitting"; subplot=1)
ylabel!(raw"time (s)"; subplot=2)
savefig("errors_time.png")

𝛆′ = chebyshevnodes_1st(1000, (0, 1))
plot(; size=(650, 500), PLOT_DEFAULTS...)
for (i, β′) in enumerate(β′_vals)
    for (j, nlayers) in enumerate(layers)
        for max_iter in max_iters
            key = (β′, max_iter, nlayers)
            haskey(all_models, key) || continue
            model = all_models[key]
            𝐲_fitted = fermi_dirac(model).(𝛆′)
            seriescolor = colors[mod1(i, length(colors))]
            linestyle = get(linestyles, max_iter, :solid)
            plot!(
                𝛆′,
                𝐲_fitted;
                label="β′=$(β′), I=$(max_iter), L=$(nlayers)",
                seriescolor=seriescolor,
                linestyle=linestyle,
                PLOT_DEFAULTS...,
                legend=nothing,
            )
        end
    end
end
xlabel!("𝛆′")
ylabel!("Fermi–Dirac function")
savefig("fitted.png")
