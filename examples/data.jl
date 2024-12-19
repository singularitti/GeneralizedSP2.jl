using CSV
using DataFrames: DataFrame
using Plots
using Plots: bbox
using Unitful: uparse, ustrip, @u_str

PLOT_DEFAULTS = Dict(
    :size => (900, 600),
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
    :left_margin => (2, :mm),
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
        x -> ustrip(u"μs", uparse(nospace(x))), filtered[!, proj_avg_col]
    )
    filtered[!, proj_stddev_col] = map(
        x -> ustrip(u"μs", uparse(nospace(x))), filtered[!, proj_stddev_col]
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

function plot_benchmark(folders, filenames, labels)
    mat_sizes = parse.(Int, folders)  # Convert folder names to integers for the x-axis
    plt = plot(; PLOT_DEFAULTS...)
    plot!(
        plt;
        subplot=2,
        inset=(1, bbox(0.65, 0.5, 0.3, 0.4, :bottom, :right)),
        PLOT_DEFAULTS...,
    )
    times_dict = Dict()
    errors_dict = Dict()
    for filename in filenames
        times = Float64[]
        errors = Float64[]
        for folder in folders
            time, error = process_file_with_mapping(folder, filename)
            push!(times, time)
            push!(errors, error)
        end
        times_dict[filename] = times
        errors_dict[filename] = errors
        # Main plot
        plot!(
            mat_sizes,
            times;
            xticks=(mat_sizes, string.(mat_sizes)),
            yticks=exp10.(0:9),
            xaxis=:log2,
            yaxis=:log10,
            yerror=errors,
            label=labels[filename],
            PLOT_DEFAULTS...,
        )
        inset_range = mat_sizes .>= 8192
        inset_times = times_dict[filename][inset_range]
        inset_errors = errors_dict[filename][inset_range]
        inset_sizes = mat_sizes[inset_range]
        plot!(  # See https://discourse.julialang.org/t/102936/12
            plt,
            inset_sizes,
            inset_times;
            subplot=2,
            xticks=(inset_sizes, string.(inset_sizes)),
            yticks=exp10.(5:9),
            yminorticks=100,
            xaxis=:log2,
            yaxis=:log10,
            yerror=inset_errors,
            label="",
            PLOT_DEFAULTS...,
        )
    end
    xlabel!(plt, raw"Hamiltonian size $N$"; subplot=1)
    ylabel!(plt, "total (kernel) time (μs)"; subplot=1)
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
labels = Dict(
    "exactcuda.csv" => "CUDA-C++ exact",
    "exactgpu.csv" => "Julia GPU exact",
    "modelcu.csv" => "Julia GPU model",
    "modelgpu_double.csv" => "CUDA-C++ model double",
    "modelgpu_single.csv" => "CUDA-C++ model single",
    "modelgpu_mixed.csv" => "CUDA-C++ model mixed",
)
plot_benchmark(folders, filenames, labels)
