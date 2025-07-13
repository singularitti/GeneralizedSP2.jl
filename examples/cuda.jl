using AffineScaler: rescale_one_zero
using ChairmarksExtras: @btimed
using CUDA
using Distributions: LogUniform
using GeneralizedSP2
using LinearAlgebra
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

const INPUT_ELTYPE = Ref(Float32)
set_input_eltype(t::DataType) = INPUT_ELTYPE[] = t
get_input_eltype() = INPUT_ELTYPE[]

const OUTPUT_ELTYPE = Ref(Float32)
set_output_eltype(t::DataType) = OUTPUT_ELTYPE[] = t
get_output_eltype() = OUTPUT_ELTYPE[]

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
H′ = get_input_eltype().(H_scaled)

lower_bound, upper_bound = zero(get_input_eltype()), one(get_input_eltype())
𝐱′ = get_input_eltype().(chebyshevnodes_1st(1000, (0, 1)))
μ′ = get_input_eltype()(μ′)
β′ = get_input_eltype()(β′)
fitted = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, 18); max_iter=1000000)
model = convert(Model{get_input_eltype()}, fitted.model)

function exactcpu(H′::Matrix)
    return @btimed fermi_dirac(H′, μ′, β′)
end
# cpu_exact = exactcpu(H′)
# exact_N = tr(cpu_exact)
# exact_fd = diag(inv(V′) * cpu_exact * V′)

function modelcpu(H′::Matrix)
    𝞀 = similar(H′, get_output_eltype())
    return @btimed fermi_dirac!(𝞀, model, H′)
end
# cpu_model = modelcpu(H′)
# cpu_N = tr(cpu_model)
# cpu_fd = diag(inv(V) * cpu_model * V)

function modelgpu(H′::CuMatrix, model; preheat=3)  # Julia model
    𝞀 = similar(H′, get_output_eltype())
    for _ in 1:preheat
        fermi_dirac!(𝞀, model, H′)  # Preheating GPU
    end
    CUDA.@profile fermi_dirac!(𝞀, model, H′)  # Only profile the last run
    return 𝞀
end
modelgpu(H′::Matrix, model; kwargs...) = modelgpu(CuMatrix(H′), model; kwargs...)

function exactgpu(H′::CuMatrix; preheat=3)  # Julia
    𝞀 = similar(H′, get_output_eltype())
    for _ in 1:preheat
        fermi_dirac!(𝞀, H′, μ, β)  # Preheating GPU
    end
    CUDA.@profile fermi_dirac!(𝞀, H′, μ, β)
    return 𝞀
end
exactgpu(H′::Matrix; kwargs...) = exactgpu(CuMatrix(H′); kwargs...)

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
