using AffineScaler: rescale_one_zero
using ChairmarksExtras: @btimed
using CUDA: DEFAULT_MATH, FAST_MATH, PEDANTIC_MATH, MathMode, CuMatrix, math_mode!
using DataFrames: DataFrame, dropmissing
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
    :left_margin => (0, :mm),
    :linewidth => 2,
    :markersize => 4,
    :markerstrokewidth => 0,
    :minorticks => 10,
    :titlefontsize => 9,
    :plot_titlefontsize => 9,
    :guidefontsize => 9,
    :tickfontsize => 7,
    :legendfontsize => 7,
    :legend_background_color => nothing,
    :legend_foreground_color => nothing,
    :legend_position => :topleft,
    :legendfontfamily => "Palatino Roman",
    :xguidefontfamily => "Palatino Italic",
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
            exactcpu=exactcpu(H″, μ″, β″),
            modelcpu=modelcpu!(similar(H″, OUTPUT_ELTYPE), H″, model),
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
    value=Union{Missing,Any}[],
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

# Compute error_norm for each row
df[!, :error_norm] = zeros(nrow(df))
for sys in unique(df.sys_size)
    df_sys = filter(row -> row.sys_size == sys, df)

    # Find accurate benchmark value: prefer exactcpu (Float64, Float64) with PEDANTIC_MATH, then DEFAULT_MATH;
    # if not, exactgpu (Float64, Float64) with PEDANTIC_MATH, then DEFAULT_MATH
    bench_value = nothing
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

    # Compute norm differences
    sys_indices = findall(row -> row.sys_size == sys, eachrow(df))
    for idx in sys_indices
        df.error_norm[idx] = norm(Matrix{Float64}(df.value[idx]) - bench_value)
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
    (DEFAULT_MATH, nothing) => :utriangle,
    (PEDANTIC_MATH, nothing) => :+,
    (FAST_MATH, :Float16) => :star4,
    (FAST_MATH, :TensorFloat32) => :circle,
)
plt = plot(; layout=(1, 2), PLOT_DEFAULTS..., size=(1300, 500))

# Plot error_norm in subplot 1
for func_name in reverse(unique(df.function_name))
    df_func = filter(row -> row.function_name == func_name, df)
    for ((input_eltype, output_eltype), color) in zip(keys(labels), colors)
        label = labels[(input_eltype, output_eltype)]
        df_pair = filter(
            row -> row.INPUT_ELTYPE == input_eltype && row.OUTPUT_ELTYPE == output_eltype,
            df_func,
        )
        for math_mode in unique(df_pair.math_mode)
            df_mode = filter(row -> row.math_mode == math_mode, df_pair)
            for precision in unique(df_mode.precision)
                df_subset = filter(row -> row.precision == precision, df_mode)
                # Sort by sys_size to ensure correct line plotting
                sort!(df_subset, :sys_size)
                if !isempty(df_subset)
                    display(df_subset)
                    # Use solid line for model functions, dot for exact functions
                    linestyle = occursin("exact", func_name) ? :dot : :solid
                    # Get marker for this math_mode/precision combination
                    marker = markers[(math_mode, precision)]
                    combined_label = "$func_name, $label, $math_mode"
                    if !isnothing(precision)
                        combined_label *= " ($precision)"
                    end
                    plot!(
                        plt,
                        df_subset.sys_size,
                        df_subset.error_norm;
                        xscale=:log2,
                        yscale=:log10,
                        subplot=1,
                        label=combined_label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        PLOT_DEFAULTS...,
                    )
                end
            end
        end
    end
end

# Plot time_per_eval_sample in subplot 2
for func_name in reverse(unique(df.function_name))
    df_func = filter(row -> row.function_name == func_name, df)
    for ((input_eltype, output_eltype), color) in zip(keys(labels), colors)
        label = labels[(input_eltype, output_eltype)]
        df_pair = filter(
            row -> row.INPUT_ELTYPE == input_eltype && row.OUTPUT_ELTYPE == output_eltype,
            df_func,
        )
        for math_mode in unique(df_pair.math_mode)
            df_mode = filter(row -> row.math_mode == math_mode, df_pair)
            for precision in unique(df_mode.precision)
                df_subset = filter(row -> row.precision == precision, df_mode)
                # Sort by sys_size to ensure correct line plotting
                sort!(df_subset, :sys_size)
                if !isempty(df_subset)
                    display(df_subset)
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
                        PLOT_DEFAULTS...,
                    )
                end
            end
        end
    end
end
xticks!(exp2.(9:15); subplot=2)
yticks!(exp10.(-6:6); subplot=2)
xlabel!(raw"system size $N$"; subplot=2)
ylabel!(raw"RMSE of fitting"; subplot=1)
ylabel!(raw"time (s)"; subplot=2)
savefig("profile.png")
