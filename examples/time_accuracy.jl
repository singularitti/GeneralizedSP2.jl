using DataFrames: DataFrame, sort!, unstack, nrow
using NsightSystems.Recipes:
    CUDA_API, CUDA_KERNEL, MEMORY_OPER, Microsecond, load_summary_csv, get_total_time
using Plots: plot, plot!, xticks!, yticks!, xlims!, xlabel!, ylabel!, savefig

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1.5,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :left_margin => (1, :mm),
    :bottom_margin => (1, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :bottomright,
    :legend_columns => 2,
    :background_color_inside => nothing,
    :legendfontfamily => "Palatino Roman",
    :guidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
)

function process_csv_files(types, precisions, sizes, basedir)
    # Path to the CSV files
    basedir = abspath(expanduser(basedir))
    # Create a DataFrame to store the results
    results = DataFrame(; type=String[], precision=Int[], size=Int[], time=Microsecond[])
    # Process each file and add to the DataFrame
    for type in types
        for precision in precisions
            for size in sizes
                filename = "$(type)_fp$(precision)_$(size)_cuda_api_gpu_sum.csv"
                filepath = joinpath(basedir, filename)
                if !isfile(filepath)
                    continue  # Skip if file doesn't exist
                end
                try
                    summary = load_summary_csv(filepath)
                    time = get_total_time(summary, [CUDA_API, CUDA_KERNEL, MEMORY_OPER])
                    push!(results, (type, precision, size, time))
                catch e
                    println("Error processing $filename: $e")
                end
            end
        end
    end
    sort!(results, [:type, :precision, :size])
    # Create a wide-format DataFrame for easier visualization
    unstacked = unstack(results, [:type, :precision], :size, :time)
    return results, unstacked
end

function benchmark_errors(matrices)
    results = DataFrame(;
        size=Int[], type=String[], precision=Int[], relative_error=Float64[]
    )
    # For each size, compare all matrices to the exactgpu_64 benchmark
    for size in unique(matrices.size)
        benchmark_row = filter(
            row -> row.type == "exactgpu" && row.precision == 64 && row.size == size,
            matrices,
        )
        if nrow(benchmark_row) == 1
            benchmark_matrix = benchmark_row[1, :matrix]
            # Get all matrices with the same size (excluding the benchmark)
            test_matrices = filter(
                row -> row.size == size && !(row.type == "exactgpu" && row.precision == 64),
                matrices,
            )
            # Calculate relative error for each test matrix
            for row in eachrow(test_matrices)
                test_matrix = row.matrix
                # Compute infinity norm error
                relative_error =
                    norm(test_matrix - benchmark_matrix, Inf) / norm(benchmark_matrix, Inf)
                push!(results, (size, row.type, row.precision, relative_error))
            end
        end
    end
    return results
end

types = ["modelgpu", "exactgpu"]
precisions = [16, 32, 64]
sizes = [512, 1024, 2048, 4096, 8192, 16384]

results, unstacked = process_csv_files(types, precisions, sizes, basedir)

colors = Dict(16 => "#1f77b4", 32 => "#ff7f0e", 64 => "#2ca02c")
line_styles = Dict("exactgpu" => :solid, "modelgpu" => :dash)
plt = plot(; size=(450, 625), layout=(2, 1), xscale=:log2, yscale=:log10)
for type in unique(results.type)
    for precision in unique(results.precision)
        filtered = filter(row -> row.type == type && row.precision == precision, results)
        if nrow(filtered) > 0
            label = "$(type[begin:(end-3)]) FP$precision"
            plot!(
                plt,
                filtered.size,
                filtered.time;
                subplot=1,
                color=colors[precision],
                linestyle=line_styles[type],
                label=label,
                PLOT_DEFAULTS...,
            )
        end
    end
end
yticks!(plt, exp10.(0:8); subplot=1)
ylabel!("total time (μs)"; subplot=1)

# errors = benchmark_errors(matrices_df)
sort!(errors, [:type, :precision, :size])
for type in ["modelgpu", "exactgpu"]
    for precision in [16, 32, 64]
        # Skip exactgpu_64 as it's the benchmark
        if type == "exactgpu" && precision == 64
            continue
        end
        # Skip exactgpu_16 as it doesn't exist
        if type == "exactgpu" && precision == 16
            continue
        end
        # Filter data for this combination
        filtered = filter(row -> row.type == type && row.precision == precision, errors)
        if nrow(filtered) > 0
            # Sort by size to ensure proper line connection
            sort!(filtered, :size)
            label = "$(type[begin:(end-3)]) FP$precision"
            plot!(
                plt,
                filtered.size,
                filtered.relative_error;
                subplot=2,
                color=colors[precision],
                linestyle=line_styles[type],
                label=label,
                PLOT_DEFAULTS...,
            )
        end
    end
end
xticks!(plt, sizes)
yticks!(plt, exp10.((-10):2:(-1)); subplot=2)
xlims!(plt, extrema(sizes))
xlabel!(plt, "system size (N)"; subplot=2)
ylabel!("relative errors (infinity norm)"; subplot=2)
savefig("relative_errors.png")
