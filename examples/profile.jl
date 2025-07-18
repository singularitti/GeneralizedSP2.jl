using AffineScaler: rescale_one_zero
using ChairmarksExtras: @btimed
using CUDA: DEFAULT_MATH, FAST_MATH, PEDANTIC_MATH, MathMode, CuMatrix, math_mode!
using DataFrames: DataFrame, dropmissing, nrow
using Distributions: LogUniform
using GeneralizedSP2
using LinearAlgebra
using OrderedCollections: OrderedDict
using Plots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :dpi => 400,
    :background_color_inside => nothing,
    :framestyle => :box,
    :grid => nothing,
    :left_margin => (6, :mm),
    :bottom_margin => (7, :mm),
    :linewidth => 2,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 10,
    :guidefontsize => 12,
    :tickfontsize => 10,
    :legendfontsize => 10,
    :legend_background_color => nothing,
    :legend_foreground_color => nothing,
    :legendfontfamily => "Palatino Roman",
    :xguidefontfamily => "Palatino Roman",
    :yguidefontfamily => "Palatino Roman",
    :tickfontfamily => "Palatino Roman",
)

const ELTYPE_PAIRS = (
    (Float16, Float16), (Float16, Float32), (Float32, Float32), (Float64, Float64)
)
const MATH_MODES = (
    (DEFAULT_MATH, nothing),
    (PEDANTIC_MATH, nothing),
    (FAST_MATH, :Float16),
    (FAST_MATH, :TensorFloat32),
)

lower_bound, upper_bound = 0, 1
𝐱′ = chebyshevnodes_1st(1000, (0, 1))

β = 1.25  # Physical
μ = 11.5  # Physical
sys_size = 2048
εₘᵢₙ, εₘₐₓ = 1, 20
dist = LogUniform(εₘᵢₙ, εₘₐₓ)
Λ = rand(EigvalsSampler(dist), sys_size)
V = rand(EigvecsSampler(dist), sys_size, sys_size)
set_isapprox_rtol(1e-10)
H = Hamiltonian(Eigen(Λ, V))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)
fitted = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, 18); maxiters=1000)

function exactcpu(H′::Matrix, μ′, β′)
    return @btimed fermi_dirac(H′, μ′, β′)
end
# cpu_exact = exactcpu(H′)
# exact_N = tr(cpu_exact)
# exact_fd = diag(inv(V′) * cpu_exact * V′)

function modelcpu!(rho, H′::Matrix, model)
    return @btimed fermi_dirac!(rho, model, H′)
end
# cpu_model = modelcpu(H′)
# cpu_N = tr(cpu_model)
# cpu_fd = diag(inv(V) * cpu_model * V)

function modelgpu!(rho::CuMatrix, H′::CuMatrix, model; preheat=3)  # Julia model
    for _ in 1:preheat
        fermi_dirac!(rho, model, H′)  # Preheating GPU
    end
    return @btimed fermi_dirac!(rho, model, H′)  # Only profile the last run
end
modelgpu!(rho::Matrix, H′::Matrix, model; kwargs...) =
    modelgpu!(CuMatrix(rho), CuMatrix(H′), model; kwargs...)

function exactgpu!(rho::CuMatrix, H′::CuMatrix, μ′, β′; preheat=3)  # Julia
    for _ in 1:preheat
        fermi_dirac!(rho, H′, μ′, β′)  # Preheating GPU
    end
    return @btimed fermi_dirac!(rho, H′, μ′, β′)
end
exactgpu!(rho::Matrix, H′::Matrix, μ′, β′; kwargs...) =
    exactgpu!(CuMatrix(rho), CuMatrix(H′), μ′, β′; kwargs...)

results = OrderedDict{}()
# Iterate over element type pairs and math modes
for (INPUT_ELTYPE, OUTPUT_ELTYPE) in ELTYPE_PAIRS
    H″ = INPUT_ELTYPE.(H_scaled)
    μ″ = INPUT_ELTYPE(μ′)
    β″ = INPUT_ELTYPE(β′)
    model = convert(Model{INPUT_ELTYPE}, fitted.model)
    for (math_mode, precision) in MATH_MODES
        # Skip FAST_MATH for non-Float32 input-output pairs
        if math_mode == FAST_MATH && INPUT_ELTYPE != Float32
            continue
        end
        # Set the math mode
        if math_mode == FAST_MATH
            math_mode!(math_mode; precision=precision)
        else
            math_mode!(math_mode)
        end
        # Run the functions and store results
        key = (INPUT_ELTYPE, OUTPUT_ELTYPE, math_mode, precision, sys_size)
        result = (
            exactcpu=if sys_size < 8192
                exactcpu(H″, μ″, β″)
            else
                missing  # Skip large systems for exactcpu
            end,
            modelcpu=if sys_size < 8192
                modelcpu!(similar(H″, OUTPUT_ELTYPE), H″, model)
            else
                missing  # Skip large systems for modelcpu
            end,
            modelgpu=modelgpu!(similar(H″, OUTPUT_ELTYPE), H″, model),
            exactgpu=if (INPUT_ELTYPE, OUTPUT_ELTYPE) in
                ((Float32, Float32), (Float64, Float64))
                exactgpu!(similar(H″, OUTPUT_ELTYPE), H″, μ″, β″)
            else
                missing
            end,
        )
        results[key] = result
    end
end

df = DataFrame(;
    INPUT_ELTYPE=DataType[],
    OUTPUT_ELTYPE=DataType[],
    math_mode=MathMode[],
    precision=Union{Symbol,Nothing}[],
    sys_size=Int[],
    function_name=String[],
    time=Union{Missing,Float64}[],
    evals=Union{Missing,Int}[],
    samples=Union{Missing,Int}[],
    time_per_eval_sample=Union{Missing,Float64}[],
    value=Union{Missing,Matrix}[],
)
for (key, result) in results
    input_eltype, output_eltype, math_mode, precision, sys_size = key
    # Process each function's benchmark
    for (func_name, bench) in [
        ("exactcpu", result.exactcpu),
        ("modelcpu", result.modelcpu),
        ("modelgpu", result.modelgpu),
        ("exactgpu", result.exactgpu),
    ]
        # Handle missing exactgpu results
        if ismissing(bench)
            push!(
                df,
                (
                    input_eltype,
                    output_eltype,
                    math_mode,
                    precision,
                    sys_size,
                    func_name,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                ),
            )
        else
            time = bench.time
            evals = bench.evals
            samples = bench.samples
            time_per_eval_sample =
                (evals * samples == 0) ? missing : time / (evals * samples)
            push!(
                df,
                (
                    input_eltype,
                    output_eltype,
                    math_mode,
                    precision,
                    sys_size,
                    func_name,
                    time,
                    evals,
                    samples,
                    time_per_eval_sample,
                    bench.value,
                ),
            )
        end
    end
end
df = dropmissing(df, :time_per_eval_sample)

# Add a new column for error_norm, initialized to zeros for all rows
df[!, :error_norm] = zeros(nrow(df))
# Loop over each unique system size to compute errors independently
for sys in unique(df.sys_size)
    # Filter the DataFrame to rows matching the current system size
    df_sys = filter(row -> row.sys_size == sys, df)
    # Initialize benchmark value as nothing; it will be set based on priority
    bench_value = nothing
    # First preference: exactcpu with PEDANTIC_MATH in Float64 (strict precision on CPU)
    ped_cpu = filter(
        row ->
            row.function_name == "exactcpu" &&
                row.math_mode == PEDANTIC_MATH &&
                isnothing(row.precision) &&
                row.INPUT_ELTYPE == Float64 &&
                row.OUTPUT_ELTYPE == Float64,
        df_sys,
    )
    if !isempty(ped_cpu)
        bench_value = ped_cpu.value[1]
    else
        # Second: exactcpu with DEFAULT_MATH in Float64 (optimized but precise on CPU)
        def_cpu = filter(
            row ->
                row.function_name == "exactcpu" &&
                    row.math_mode == DEFAULT_MATH &&
                    isnothing(row.precision) &&
                    row.INPUT_ELTYPE == Float64 &&
                    row.OUTPUT_ELTYPE == Float64,
            df_sys,
        )
        if !isempty(def_cpu)
            bench_value = def_cpu.value[1]
        else
            # Third: exactgpu with PEDANTIC_MATH in Float64 (strict precision on GPU)
            ped_gpu = filter(
                row ->
                    row.function_name == "exactgpu" &&
                        row.math_mode == PEDANTIC_MATH &&
                        isnothing(row.precision) &&
                        row.INPUT_ELTYPE == Float64 &&
                        row.OUTPUT_ELTYPE == Float64,
                df_sys,
            )
            if !isempty(ped_gpu)
                bench_value = ped_gpu.value[1]
            else
                # Fourth: exactgpu with DEFAULT_MATH in Float64 (optimized on GPU)
                def_gpu = filter(
                    row ->
                        row.function_name == "exactgpu" &&
                            row.math_mode == DEFAULT_MATH &&
                            isnothing(row.precision) &&
                            row.INPUT_ELTYPE == Float64 &&
                            row.OUTPUT_ELTYPE == Float64,
                    df_sys,
                )
                if !isempty(def_gpu)
                    bench_value = def_gpu.value[1]
                end
            end
        end
    end
    # Compute relative norm differences for all rows matching the current system size
    sys_indices = findall(row -> row.sys_size == sys, eachrow(df))
    for idx in sys_indices
        # Calculate relative error: norm(computed - benchmark) / norm(benchmark)
        # Convert computed value to Matrix{Float64} for consistent comparison
        computed_value = Matrix{Float64}(df.value[idx])
        abs_diff_norm = norm(computed_value - bench_value)
        bench_norm = norm(bench_value)
        df.error_norm[idx] = (bench_norm > 0) ? abs_diff_norm / bench_norm : abs_diff_norm  # Avoid division by zero
    end
end

labels = OrderedDict(
    (Float64, Float64) => "FP64 → FP64",
    (Float32, Float32) => "FP32 → FP32",
    (Float16, Float32) => "FP16 → FP32",
    (Float16, Float16) => "FP16 → FP16",
)
colors = palette(:tab10)
markers = Dict(
    (DEFAULT_MATH, nothing) => :utriangle, (FAST_MATH, :TensorFloat32) => :circle
)
plt = plot(; layout=(1, 2), PLOT_DEFAULTS..., size=(1300, 500))

# Plot error_norm in subplot 1
for ((input_eltype, output_eltype), color) in zip(reverse(collect(keys(labels))), colors)
    label = labels[(input_eltype, output_eltype)]
    df_pair = filter(
        row -> row.INPUT_ELTYPE == input_eltype && row.OUTPUT_ELTYPE == output_eltype, df
    )
    for func_name in ("modelgpu", "exactgpu")
        df_func = filter(row -> row.function_name == func_name, df_pair)
        for math_mode in (FAST_MATH, DEFAULT_MATH)
            df_mode = filter(row -> row.math_mode == math_mode, df_func)
            for precision in (:TensorFloat32, nothing)
                df_subset = filter(row -> row.precision == precision, df_mode)
                # Sort by sys_size to ensure correct line plotting
                sort!(df_subset, :sys_size)
                if !isempty(df_subset)
                    # Skip plotting if all error_norms in this subset are zero (i.e., this is the benchmark configuration)
                    if all(df_subset.error_norm .== 0)
                        continue
                    end
                    # Use solid line for model functions, dot for exact functions
                    linestyle = occursin("exact", func_name) ? :dot : :solid
                    # Get marker for this math_mode/precision combination
                    marker = markers[(math_mode, precision)]
                    combined_label = "$func_name, $label"
                    if !isnothing(precision)
                        combined_label *= " ($precision)"
                    end
                    # Replace zero error_norms with eps(Float64) for log-scale compatibility
                    plot_error = copy(df_subset.error_norm)
                    plot_error[plot_error .== 0] .= 10^(-13)
                    plot!(
                        plt,
                        df_subset.sys_size,
                        plot_error;
                        xscale=:log2,
                        yscale=:log10,
                        subplot=1,
                        label=combined_label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        legend_position=(0.1, 0.4),
                        PLOT_DEFAULTS...,
                    )
                end
            end
        end
    end
end
xticks!(exp2.(9:15); subplot=1)
yticks!(exp10.(-15:1); subplot=1)
xlabel!(raw"system size"; subplot=1)
ylabel!(raw"norm difference ratio"; subplot=1)

# Plot time_per_eval_sample in subplot 2
for func_name in ("exactgpu", "modelgpu")
    df_func = filter(row -> row.function_name == func_name, df)
    for ((input_eltype, output_eltype), color) in zip(keys(labels), colors)
        label = labels[(input_eltype, output_eltype)]
        df_pair = filter(
            row -> row.INPUT_ELTYPE == input_eltype && row.OUTPUT_ELTYPE == output_eltype,
            df_func,
        )
        for math_mode in (DEFAULT_MATH, FAST_MATH)
            df_mode = filter(row -> row.math_mode == math_mode, df_pair)
            for precision in (nothing, :TensorFloat32)
                df_subset = filter(row -> row.precision == precision, df_mode)
                # Sort by sys_size to ensure correct line plotting
                sort!(df_subset, :sys_size)
                if !isempty(df_subset)
                    # Use solid line for model functions, dot for exact functions
                    linestyle = occursin("exact", func_name) ? :dot : :solid
                    # Get marker for this math_mode/precision combination
                    marker = markers[(math_mode, precision)]
                    combined_label = "$func_name, $label"
                    if !isnothing(precision)
                        combined_label *= " ($precision)"
                    end
                    plot!(
                        plt,
                        df_subset.sys_size,
                        df_subset.time_per_eval_sample;
                        xscale=:log2,
                        yscale=:log10,
                        subplot=2,
                        label=combined_label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        legend_position=(0.1, 0.96),
                        PLOT_DEFAULTS...,
                    )
                end
            end
        end
    end
end
xticks!(exp2.(9:15); subplot=2)
yticks!(exp10.(-6:6); subplot=2)
xlabel!(raw"system size"; subplot=2)
ylabel!(raw"time (s)"; subplot=2)
savefig("profile.png")
