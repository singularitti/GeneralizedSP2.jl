using AffineScaler: rescale_one_zero
using ChairmarksExtras: @btimed
using CUDA
using Distributions: LogUniform
using GeneralizedSP2
using LinearAlgebra
using OrderedCollections: OrderedDict
# using Plots
using ToyHamiltonians

PLOT_DEFAULTS = Dict(
    :size => (400, 300),
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
    :legend_position => :bottomleft,
    :background_color_inside => nothing,
    :color_palette => :tab10,
)

const ELTYPE_PAIRS = [
    (Float16, Float16), (Float16, Float32), (Float32, Float32), (Float64, Float64)
]
# Define math modes and precisions
const MATH_MODES = [
    (CUDA.DEFAULT_MATH, nothing),
    (CUDA.PEDANTIC_MATH, nothing),
    (CUDA.FAST_MATH, :Float16),
    (CUDA.FAST_MATH, :TensorFloat32),
]
INPUT_ELTYPE = Float32
OUTPUT_ELTYPE = Float32

β = 1.25  # Physical
μ = 11.5  # Physical
sys_size = 2048
dist = LogUniform(1, 20)
Λ = rand(EigvalsSampler(dist), sys_size)
V = rand(EigvecsSampler(dist), sys_size, sys_size)
set_isapprox_rtol(1e-10)
H = Hamiltonian(Eigen(Λ, V))
εₘᵢₙ, εₘₐₓ = 1, 20
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)
H′ = INPUT_ELTYPE.(H_scaled)

lower_bound, upper_bound = 0, 1
𝐱′ = chebyshevnodes_1st(1000, (0, 1))
fitted = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, 18); maxiters=1000)
model = convert(Model{INPUT_ELTYPE}, fitted.model)
μ′ = INPUT_ELTYPE(μ′)
β′ = INPUT_ELTYPE(β′)

function exactcpu(H′::Matrix, μ′, β′)
    return @btimed fermi_dirac(H′, μ′, β′)
end
# cpu_exact = exactcpu(H′)
# exact_N = tr(cpu_exact)
# exact_fd = diag(inv(V′) * cpu_exact * V′)

function modelcpu(H′::Matrix, model)
    𝞀 = similar(H′, OUTPUT_ELTYPE)
    return @btimed fermi_dirac!(𝞀, model, H′)
end
# cpu_model = modelcpu(H′)
# cpu_N = tr(cpu_model)
# cpu_fd = diag(inv(V) * cpu_model * V)

function modelgpu(H′::CuMatrix, model; preheat=3)  # Julia model
    𝞀 = similar(H′, OUTPUT_ELTYPE)
    for _ in 1:preheat
        fermi_dirac!(𝞀, model, H′)  # Preheating GPU
    end
    return @btimed fermi_dirac!(𝞀, model, H′)  # Only profile the last run
end
modelgpu(H′::Matrix, model; kwargs...) = modelgpu(CuMatrix(H′), model; kwargs...)

function exactgpu(H′::CuMatrix, μ′, β′; preheat=3)  # Julia
    𝞀 = similar(H′, OUTPUT_ELTYPE)
    for _ in 1:preheat
        fermi_dirac!(𝞀, H′, μ′, β′)  # Preheating GPU
    end
    return @btimed fermi_dirac!(𝞀, H′, μ′, β′)
end
exactgpu(H′::Matrix, μ′, β′; kwargs...) = exactgpu(CuMatrix(H′), μ′, β′; kwargs...)

results = OrderedDict{}()
# Iterate over element type pairs and math modes
for (INPUT_ELTYPE, OUTPUT_ELTYPE) in ELTYPE_PAIRS
    H′_typed = INPUT_ELTYPE.(H_scaled)
    μ′_typed = INPUT_ELTYPE(μ′)
    β′_typed = INPUT_ELTYPE(β′)
    model_typed = convert(Model{INPUT_ELTYPE}, model)
    for (math_mode, precision) in MATH_MODES
        # Skip FAST_MATH for non-Float32 input-output pairs
        if math_mode == CUDA.FAST_MATH && INPUT_ELTYPE != Float32
            continue
        end
        # Set the math mode
        if math_mode == CUDA.FAST_MATH
            CUDA.math_mode!(math_mode; precision=precision)
        else
            CUDA.math_mode!(math_mode)
        end
        # Run the functions and store results
        key = (INPUT_ELTYPE, OUTPUT_ELTYPE, math_mode, precision)
        result = (
            exactcpu=exactcpu(H′_typed, μ′_typed, β′_typed),
            exactgpu=exactgpu(H′_typed, μ′_typed, β′_typed),
            modelcpu=modelcpu(H′_typed, model_typed),
            modelgpu=modelgpu(H′_typed, model_typed),
        )
        results[key] = result
    end
end

# layout = (2, 1)
# plot(; layout=layout, PLOT_DEFAULTS..., size=(1600 / 3, 800))
# scatter!(𝛌, fd_benchmark; subplot=1, label="target Fermi–Dirac", PLOT_DEFAULTS...)
# scatter!(𝛌, fd_cpu; subplot=1, label="MLSP2 model", PLOT_DEFAULTS...)
# scatter!(𝛌, fd_gpu; subplot=1, label="MLSP2 model CUDA", PLOT_DEFAULTS...)
# xlabel!("eigenvalues of H"; subplot=1)
# ylabel!("Fermi–Dirac distribution"; subplot=1)

# hline!(
#     [zero(eltype(fd_benchmark))];
#     subplot=2,
#     seriescolor=:black,
#     primary=false,
#     PLOT_DEFAULTS...,
# )
# scatter!(𝛌, fd_benchmark - fd_cpu; subplot=2, label="MLSP2 model", PLOT_DEFAULTS...)
# scatter!(𝛌, fd_benchmark - fd_gpu; subplot=2, label="MLSP2 model CUDA", PLOT_DEFAULTS...)
# xlabel!("eigenvalues of H"; subplot=2)
# ylabel!("Fermi–Dirac distribution difference"; subplot=2)
