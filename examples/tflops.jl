using CSV
using DataFrames
using NsightCompute: compute_flops
using Plots

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 12,
    :guidefontsize => 10,
    :tickfontsize => 8,
    :legendfontsize => 8,
    :margin => (6, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_columns => 2,
    :background_color_inside => nothing,
    :color_palette => :tab10,
    :legendfontfamily => "Palatino Italic",
    :guidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
    :titlefontfamily => "Palatino Roman",
)

# Prepare parameters
function compute_peak_tflops(
    types::Vector{String}, precisions::Vector{String}, sizes::Vector{Int}
)
    results = DataFrame(;
        Type=String[], Precision=String[], Size=Int[], Peak_TFLOPS=Float64[]
    )
    # Loop through files, compute TFLOPS and record peak values
    for t in types, p in precisions, s in sizes
        filename = "$(t)_$(p)_$s.csv"
        _, tflops_df = compute_flops(filename)
        if isnothing(tflops_df)
            continue
        end
        # Record peak TFLOPS (max across all rows)
        peak_tflops = maximum(tflops_df.TFLOPS_Total) ./ 10^12
        push!(results, (t, p, s, peak_tflops))
    end
    return results
end

types = ["exactgpu", "modelgpu"]
precisions = ["f32", "f64", "mixed"]
sizes = [512, 1024, 2048, 4096, 8192, 16384]
results = compute_peak_tflops(types, precisions, sizes)

# Define display names for types and precisions
type_display = Dict("exactgpu" => "exact diagonalization", "modelgpu" => "model")

precision_display = Dict(
    "f32" => "single precision", "f64" => "double precision", "mixed" => "mixed precision"
)

# Define the custom layout
l = @layout [
    [a{0.25h}; b{0.25h}; c{0.25h}; d{0.25h}] e{0.5w};
]

plt = plot(; size=(1800, 1200), layout=l, PLOT_DEFAULTS..., right_margin=(20, :mm))

for t in types
    for p in precisions
        df_plot = filter(row -> row.Type == t && row.Precision == p, results)
        if !isempty(df_plot)  # Only plot if we have data for this combination
            linestyle = t == "exactgpu" ? :solid : :dash

            plot!(
                plt,
                df_plot.Size,
                df_plot.Peak_TFLOPS;
                subplot=5,
                label="$(type_display[t]), $(precision_display[p])",
                linestyle=linestyle,
                xlabel=raw"system size $N$",
                ylabel="peak teraFLOPS",
                xscale=:log2,
                yscale=:log10,
                xticks=sizes,
                legend_position=:bottomright,
                PLOT_DEFAULTS...,
            )
        end
    end
end
title!(plt, "peak teraFLOPS versus system size"; subplot=5)

# Function to plot a specific type+precision combination
function plot_type_precision!(plt, type, precision, subplot_idx, custom_title=nothing)
    linestyle = type == "exactgpu" ? :solid : :dash

    # Use the display names for the title if no custom title is provided
    if isnothing(custom_title)
        display_title = "$(type_display[type]) with $(precision_display[precision])"
    else
        display_title = custom_title
    end

    for s in sizes
        filename = "$(type)_$(precision)_$s.csv"
        if !isfile(filename)
            continue
        end

        try
            df, tflops_df = compute_tflops(filename)
            if isnothing(df) || isnothing(tflops_df)
                continue
            end

            endindex = min(200, length(df.ID))

            plot!(
                plt,
                df.ID[1:endindex],
                tflops_df.TFLOPS_Total[1:endindex];
                subplot=subplot_idx,
                label="N=$s",
                linestyle=linestyle,
                xlabel="profiling trace identifier (ID)",
                ylabel="instant teraFLOPS",
                PLOT_DEFAULTS...,
            )
        catch e
            println("Error processing $filename: $e")
        end
    end
    title!(plt, display_title; subplot=subplot_idx)
    return plt
end

# Plot each type+precision combination in its own subplot
plot_type_precision!(plt, "exactgpu", "f32", 1)
plot_type_precision!(plt, "exactgpu", "f64", 2)
plot_type_precision!(plt, "modelgpu", "f32", 3)
plot_type_precision!(plt, "modelgpu", "f64", 4)

savefig("tflops.png")
# plot!(plt)
