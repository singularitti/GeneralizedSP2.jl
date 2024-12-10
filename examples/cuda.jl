using AffineScaler: rescale_one_zero
using CUDA
using Distributions: LogUniform
using GeneralizedSP2
using LinearAlgebra
using Plots
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

β = 1.25  # Physical
μ = 150  # Physical
sys_size = 2048
dist = LogUniform(100, 200)
Λ = rand(EigvalsSampler(dist), sys_size)
V = rand(EigvecsSampler(dist), sys_size, sys_size)
set_isapprox_rtol(1e-13)
H = Hamiltonian(Eigen(Λ, V))
E = eigen(H)
𝛌, V = E.values, E.vectors
εₘᵢₙ, εₘₐₓ = floor(minimum(𝛌)), ceil(maximum(𝛌))
β′ = rescale_beta((εₘᵢₙ, εₘₐₓ))(β)
μ′ = rescale_mu((εₘᵢₙ, εₘₐₓ))(μ)
H_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H)

lower_bound, upper_bound = 0, 1
𝐱′ = chebyshevnodes_1st(1000, (lower_bound, upper_bound))
fitted = fit_fermi_dirac(𝐱′, μ′, β′, init_model(μ′, 18); max_iter=1_000_000)
M = fitted.model

@assert rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ)) ≈ fermi_dirac(H_scaled, μ′, β′)
dm_exact = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
N_exact = tr(dm_exact)
fd_benchmark = fermi_dirac.(𝛌, μ, β)

dm = fermi_dirac(M)(H_scaled)
N = tr(dm)
fd_cpu = diag(inv(V) * dm * V)

X = CuMatrix(H_scaled)
DM = similar(X)
@assert Matrix(M(DM, X)) ≈ dm_exact
V′ = CuMatrix(V)
fd_gpu = Vector(diag(inv(V′) * DM * V′))

layout = (2, 1)
plot(; layout=layout, PLOT_DEFAULTS..., size=(1600 / 3, 800))
scatter!(𝛌, fd_benchmark; subplot=1, label="target Fermi–Dirac", PLOT_DEFAULTS...)
scatter!(𝛌, fd_cpu; subplot=1, label="MLSP2 model", PLOT_DEFAULTS...)
scatter!(𝛌, fd_gpu; subplot=1, label="MLSP2 model CUDA", PLOT_DEFAULTS...)
xlabel!("eigenvalues of H"; subplot=1)
ylabel!("Fermi–Dirac distribution"; subplot=1)

hline!(
    [zero(eltype(fd_benchmark))];
    subplot=2,
    seriescolor=:black,
    primary=false,
    PLOT_DEFAULTS...,
)
scatter!(𝛌, fd_benchmark - fd_cpu; subplot=2, label="MLSP2 model", PLOT_DEFAULTS...)
scatter!(𝛌, fd_benchmark - fd_gpu; subplot=2, label="MLSP2 model CUDA", PLOT_DEFAULTS...)
xlabel!("eigenvalues of H"; subplot=2)
ylabel!("Fermi–Dirac distribution difference"; subplot=2)
