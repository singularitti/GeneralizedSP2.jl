using AffineScaler: rescale_one_zero
using CUDA
# using Distributions: LogUniform
using GeneralizedSP2
using GeneralizedSP2: DOUBLE, SINGLE, MIXED
using LinearAlgebra
# using Plots
# using ToyHamiltonians
using NPZ

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

function direct_sum(matrices::Matrix...)
    # Calculate the total size of the resulting matrix
    total_rows = sum(size(mat, 1) for mat in matrices)
    total_cols = sum(size(mat, 2) for mat in matrices)
    # Initialize a zero matrix of the required size
    result = zeros(eltype(matrices[1]), total_rows, total_cols)
    # Place each matrix in its corresponding block
    row_offset = 0
    col_offset = 0
    for mat in matrices
        rows, cols = size(mat)
        result[(row_offset + 1):(row_offset + rows), (col_offset + 1):(col_offset + cols)] =
            mat
        row_offset += rows
        col_offset += cols
    end
    return result
end

β = 1.25  # Physical
μ = 11.5  # Physical
# sys_size = 1024
# dist = LogUniform(1, 20)
# Λ = rand(EigvalsSampler(dist), sys_size)
# V = rand(EigvecsSampler(dist), sys_size, sys_size)
# set_isapprox_rtol(1e-13)
# H = CuMatrix(Hamiltonian(Eigen(Λ, V)))
H = npzread("H.npy")
H = direct_sum(H, H, H, H)
𝛌 = eigvals(H)
εₘᵢₙ, εₘₐₓ = floor(minimum(𝛌)), ceil(maximum(𝛌))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

# lower_bound, upper_bound = 0, 1
# 𝐱′ = chebyshevnodes_1st(1000, (lower_bound, upper_bound))
# fitted = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, 18); max_iter=10000)
# M = fitted.model
M = loadmodel("18_1qw.npy")

N = 4096

function exactcpu(N)
    X = H_scaled[1:N, 1:N]
    return CUDA.@profile fermi_dirac(X, μ′, β′)
end
# cpu_exact = exactcpu(N)
# exact_N = tr(cpu_exact)
# exact_fd = diag(inv(V′) * cpu_exact * V′)

function modelcpu(N)
    X = H_scaled[1:N, 1:N]
    return CUDA.@profile fermi_dirac(M)(X)
end
# cpu_model = modelcpu(N)
# cpu_N = tr(cpu_model)
# cpu_fd = diag(inv(V) * cpu_model * V)

function modelcu(N; preheat=3)  # Julia model
    X = CuMatrix(H_scaled[1:N, 1:N])
    DM = zero(X)
    for _ in 1:preheat
        M(DM, X)  # Preheating GPU
    end
    CUDA.@profile M(DM, X)  # Only profile the last run
    return DM
end
function modelgpu(N, precision::Precision; preheat=3)  # CUDA
    X = CuMatrix(H[1:N, 1:N])
    DM = zero(X)
    for _ in 1:preheat
        M(DM, X, precision, (εₘᵢₙ, εₘₐₓ))  # Preheating GPU
    end
    CUDA.@profile M(DM, X, precision, (εₘᵢₙ, εₘₐₓ))
    return DM
end

function exactgpu(N, μ, β; preheat=3)  # Julia
    X = CuMatrix(H[1:N, 1:N])
    DM = zero(X)
    for _ in 1:preheat
        fermi_dirac!(DM, X, μ, β)  # Preheating GPU
    end
    CUDA.@profile fermi_dirac!(DM, X, μ, β)
    return DM
end

function exactcuda(N, μ, β; preheat=3)  # CUDA
    X = CuMatrix(H[1:N, 1:N])
    DM = zero(X)
    for _ in 1:preheat
        compute_exact_fermi_dirac!(DM, X, μ, β)  # Preheating GPU
    end
    CUDA.@profile compute_exact_fermi_dirac!(DM, X, μ, β)
    return DM
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
