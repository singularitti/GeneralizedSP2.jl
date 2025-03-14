using CSV
using DataFrames
using Plots

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 10,
    :guidefontsize => 10,
    :tickfontsize => 8,
    :legendfontsize => 8,
    :margin => (4, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :outerright,
    :legend_columns => 2,
    :background_color_inside => nothing,
    :color_palette => :tab10,
    :legendfontfamily => "Palatino Italic",
    :guidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
)

# Function to compute TFLOPS from a CSV file
function compute_tflops(filepath::String)
    if !isfile(filepath)
        return nothing, nothing
    end
    # Read CSV skipping the 2nd row (units)
    df = CSV.read(filepath, DataFrame; header=1, skipto=3)
    # GPU frequency (Hz) from SM average frequency (GHz)
    gpu_freq_hz = df[!, "sm__cycles_elapsed.avg.per_second"] .* 1e9
    # Instruction counts per cycle (inst/cycle)
    inst_dadd = df[
        !, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"
    ]
    inst_dmul = df[
        !, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"
    ]
    inst_dfma = df[
        !, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed"
    ]

    inst_fadd = df[
        !, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
    ]
    inst_fmul = df[
        !, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
    ]
    inst_ffma = df[
        !, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
    ]

    inst_hfma = df[
        !, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed"
    ]
    # Compute FLOPS in teraFLOPS (1 TFLOPS = 1e12 FLOPS)
    TFLOPS_DP = (inst_dfma .* 2 .+ inst_dadd .+ inst_dmul) .* gpu_freq_hz ./ 1e12
    TFLOPS_SP = (inst_ffma .* 2 .+ inst_fadd .+ inst_fmul) .* gpu_freq_hz ./ 1e12
    TFLOPS_HP = (inst_hfma .* 2) .* gpu_freq_hz ./ 1e12
    # Total TFLOPS
    TFLOPS_Total = TFLOPS_DP .+ TFLOPS_SP .+ TFLOPS_HP
    # Create a new DataFrame with just the TFLOPS values
    tflops_df = DataFrame(;
        TFLOPS_DP=TFLOPS_DP,
        TFLOPS_SP=TFLOPS_SP,
        TFLOPS_HP=TFLOPS_HP,
        TFLOPS_Total=TFLOPS_Total,
    )
    return df, tflops_df
end

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
        _, tflops_df = compute_tflops(filename)
        if isnothing(tflops_df)
            continue
        end
        # Record peak TFLOPS (max across all rows)
        peak_tflops = maximum(tflops_df.TFLOPS_Total)
        push!(results, (t, p, s, peak_tflops))
    end
    return results
end

types = ["exactgpu", "modelgpu"]
precisions = ["f32", "f64", "mixed"]
sizes = [512, 1024, 2048, 4096, 8192, 16384]
results = compute_peak_tflops(types, precisions, sizes)

plt = plot(;
    size=(1200, 1200),
    layout=grid(3, 1; heights=[0.34, 0.33, 0.33]),
    PLOT_DEFAULTS...,
    right_margin=(20, :mm),
)

# Define marker shapes for each precision
marker_shapes = Dict("f32" => :circle, "f64" => :square, "mixed" => :diamond)

# First subplot - peak TFLOPS vs system size for all combinations
for t in types
    for p in precisions
        df_plot = filter(row -> row.Type == t && row.Precision == p, results)
        if !isempty(df_plot)  # Only plot if we have data for this combination
            linestyle = t == "exactgpu" ? :solid : :dash
            marker = marker_shapes[p]

            plot!(
                plt,
                df_plot.Size,
                df_plot.Peak_TFLOPS;
                subplot=1,
                label="$t, $p",
                linestyle=linestyle,
                markershape=marker,
                xlabel=raw"system size $N$",
                ylabel="peak teraFLOPS",
                xscale=:log2,
                yscale=:log10,
                xticks=sizes,
                PLOT_DEFAULTS...,
            )
        end
    end
end
title!(plt, "peak teraFLOPS versus system size"; subplot=1)

# Second subplot - exact diagonalization
for s in sizes
    for p in precisions
        filename = "exactgpu_$(p)_$s.csv"
        # Skip if file doesn't exist
        if !isfile(filename)
            continue
        end

        try
            df, tflops_df = compute_tflops(filename)
            marker = marker_shapes[p]
            endindex = length(df.ID) >= 200 ? 200 : length(df.ID)

            plot!(
                plt,
                df.ID[1:endindex],
                tflops_df.TFLOPS_Total[1:endindex];
                subplot=2,
                label="exact, $p, N=$s",
                linestyle=:solid,
                markershape=marker,
                xlabel="profiling trace identifier (ID)",
                ylabel="instant teraFLOPS",
                PLOT_DEFAULTS...,
            )
        catch e
            println("Error processing $filename: $e")
        end
    end
end
title!(plt, "teraFLOPS versus each timestep for exact diagonalization"; subplot=2)

# Third subplot - model
for s in sizes
    for p in precisions
        filename = "modelgpu_$(p)_$s.csv"
        # Skip if file doesn't exist
        if !isfile(filename)
            continue
        end

        try
            df, tflops_df = compute_tflops(filename)
            marker = marker_shapes[p]
            endindex = length(df.ID) >= 200 ? 200 : length(df.ID)

            plot!(
                plt,
                df.ID[1:endindex],
                tflops_df.TFLOPS_Total[1:endindex];
                subplot=3,
                label="model, $p, N=$s",
                linestyle=:dash,
                markershape=marker,
                xlabel="profiling trace identifier (ID)",
                ylabel="instant teraFLOPS",
                PLOT_DEFAULTS...,
            )
        catch e
            println("Error processing $filename: $e")
        end
    end
end
title!(plt, "teraFLOPS versus each timestep for model"; subplot=3)
# savefig("tflops.png")
# plot!(plt)
