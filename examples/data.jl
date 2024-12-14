using CSV
using DataFrames
using Plots
using Unitful

PLOT_DEFAULTS = Dict(
    :size => (600, 400),
    :dpi => 400,
    :framestyle => :box,
    :linewidth => 1,
    :markersize => 1,
    :markerstrokewidth => 0,
    :minorticks => 5,
    :titlefontsize => 9,
    :plot_titlefontsize => 9,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :left_margin => (0, :mm),
    :grid => nothing,
    :legend_foreground_color => nothing,
    :legend_background_color => nothing,
    :legend_position => :outertop,
    :legend_columns => 3,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

# Function to process a file with shared logic
function process_file(file_path, rows_of_interest)
    df = CSV.read(file_path, DataFrame)

    # Define column names explicitly
    proj_avg_col = Symbol("Proj Avg")
    proj_stddev_col = Symbol("Proj StdDev")
    range_instances_col = Symbol("Range Instances")

    # Filter rows of interest
    filtered = filter(row -> row[:Range] in rows_of_interest, df)

    # Parse "Proj Avg" and "Proj StdDev", removing spaces
    filtered[!, proj_avg_col] = map(
        x -> u"μs"(Unitful.uparse(nospace(x))), filtered[!, proj_avg_col]
    )
    filtered[!, proj_stddev_col] = map(
        x -> u"μs"(Unitful.uparse(nospace(x))), filtered[!, proj_stddev_col]
    )

    # Calculate total projection time (T_i) and variance (Var(T_i))
    total_time_col = Symbol("Total Time")
    var_total_time_col = Symbol("Var Total Time")
    filtered[!, total_time_col] =
        filtered[!, range_instances_col] .* filtered[!, proj_avg_col]
    filtered[!, var_total_time_col] =
        filtered[!, range_instances_col] .* (filtered[!, proj_stddev_col] .^ 2)

    # Calculate sum of total projection times (S) and its variance
    S = sum(filtered[!, total_time_col])
    Var_S = sum(filtered[!, var_total_time_col])
    Std_S = sqrt(Var_S)

    return (S, Std_S)
end

function get_rows_of_interest(filename)
    return Dict(
        "exactcuda.csv" => [
            ":Exact diagonalization",
            ":Apply the Fermi–Dirac function to eigenvalues",
            ":Fill Fermi–Dirac diagonal elements to a square matrix",
            ":Compute V * f(Λ) * V⁻¹",
        ],
        "exactgpu.csv" =>
            ["CUDAExt:fermi_dirac!", "CUDAExt:density_matrix", "CUDAExt:diagonalize!"],
        "modelcu.csv" => ["CUDAExt:iterate", "CUDAExt:I - accumulator"],
        "modelgpu_double.csv" =>
            [":Main loop", ":Fermi–Dirac = I - GPU_accumulationMatrix"],
        "modelgpu_single.csv" =>
            [":Main loop", ":Fermi–Dirac = I - GPU_accumulationMatrix"],
        "modelgpu_mixed.csv" => [":Main loop", ":Fermi–Dirac = I - GPU_accumulationMatrix"],
    )[filename]
end

function process_file_with_mapping(folder, filename)
    file_path = joinpath(folder, filename)
    rows_of_interest = get_rows_of_interest(filename)
    return process_file(file_path, rows_of_interest)
end

nospace(s::AbstractString) = replace(s, r"\s+" => "")

function process_all_folders(folders)
    results = Dict()
    for folder in folders
        S, Std_S = process_file_with_mapping(folder, "modelgpu_double.csv")
        results[folder] = (S, Std_S)
    end
    return results
end

function plot_benchmark(folders, filenames, labels)
    mat_sizes = parse.(Int, folders)  # Convert folder names to integers for the x-axis
    plt = plot(; PLOT_DEFAULTS...)
    for filename in filenames
        times = typeof(1.0u"μs")[]
        errors = typeof(1.0u"μs")[]
        for folder in folders
            time, error = process_file_with_mapping(folder, filename)
            push!(times, time)
            push!(errors, error)
        end
        plot!(
            mat_sizes,
            times;
            xticks=(mat_sizes, string.(mat_sizes)),
            yticks=[10, 100, 1000, 10^4, 10^5, 10^6, 10^7],
            xaxis=:log2,
            yaxis=:log10,
            yerror=errors,
            label=labels[filename],
        )
    end
    xlabel!(plt, "Hamiltonian size (N)")
    ylabel!(plt, "Total kernel time (μs)")
    return plt
end

folders = ["512", "1024", "2048", "3072", "4096", "8192", "16384"]
filenames = [
    "exactcuda.csv",
    "exactgpu.csv",
    "modelcu.csv",
    "modelgpu_double.csv",
    "modelgpu_single.csv",
    "modelgpu_mixed.csv",
]
# Custom labels
labels = Dict(
    "exactcuda.csv" => "CUDA-C++ exact",
    "exactgpu.csv" => "Julia GPU exact",
    "modelcu.csv" => "Julia GPU model",
    "modelgpu_double.csv" => "CUDA-C++ model double",
    "modelgpu_single.csv" => "CUDA-C++ model single",
    "modelgpu_mixed.csv" => "CUDA-C++ model mixed",
)

# Plot results with custom labels
plot_benchmark(folders, filenames, labels)