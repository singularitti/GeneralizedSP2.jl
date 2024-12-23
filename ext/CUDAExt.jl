module CUDAExt

using CUDA:
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CUDA,
    CuMatrix,
    CuVector,
    CuDeviceMatrix,
    CuDeviceVector,
    CuPtr,
    DeviceMemory,
    blockIdx,
    blockDim,
    gridDim,
    threadIdx,
    launch_configuration,
    @cuda
using CUDA.CUSOLVER:
    CUSOLVER_EIG_MODE_VECTOR,
    cublasFillMode_t,
    cusolverDnCreate,
    cusolverDnDestroy,
    cusolverDnDsyevd,
    cusolverDnDsyevd_bufferSize,
    cusolverDnHandle_t,
    cusolverDnSsyevd,
    cusolverDnSsyevd_bufferSize
using CUDA.CUBLAS: axpy!, axpby!, gemm!, mul!
using LinearAlgebra: Diagonal, checksquare
using NVTX: @range

using GeneralizedSP2: AbstractModel, CUDAError, Precision, eachlayer, numlayers

import GeneralizedSP2:
    diagonalize,
    diagonalize!,
    fill_diagonal!,
    fermi_dirac,
    fermi_dirac!,
    compute_exact_fermi_dirac!

function diagonalize!(
    evals::CuVector{Cdouble,DeviceMemory},
    evecs::CuMatrix{Cdouble,DeviceMemory},
    H::CuMatrix{Cdouble},
)
    M, N = size(H)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    H′ = similar(H)  # Allocate a new `CuMatrix` on the GPU
    copyto!(H′, H)  # Efficiently copy data from `H` to `H′` on the GPU
    # Create cuSOLVER handle
    cusolver_handle = Ref{cusolverDnHandle_t}(C_NULL)
    cusolverDnCreate(cusolver_handle)
    # Specify cuSOLVER diag flags
    jobz = CUSOLVER_EIG_MODE_VECTOR  # Compute both singular values and singular vectors
    uplo = convert(cublasFillMode_t, 'L')  # `CUBLAS_FILL_MODE_LOWER`, see https://github.com/JuliaGPU/CUDA.jl/blob/45571e9/lib/cublas/util.jl#L49-L57
    # Determine the buffer size required
    lwork = Ref{Cint}(0)
    cusolverDnDsyevd_bufferSize(cusolver_handle[], jobz, uplo, N, H′, N, evals, lwork)
    # Allocate temporary workspace and device info array
    work = CuVector{Cdouble}(undef, lwork[])
    devInfo = CuVector{Cint}(undef, 1)
    # Diagonalize the matrix
    cusolverDnDsyevd(cusolver_handle[], jobz, uplo, N, H′, N, evals, work, lwork[], devInfo)
    # Handle errors
    retcode = only(Vector(devInfo))  # Copy memory from the GPU
    if retcode < 0
        throw(CUDAError(:cusolverDnDsyevd, "$(-retcode)th parameter is invalid!"))
    elseif retcode > 0
        throw(
            CUDAError(
                :cusolverDnDsyevd, "$(retcode)th off-diagonal elements did not converge!"
            ),
        )
    end
    copyto!(evecs, H′)  # Copy the eigenvectors to `evecs`
    cusolverDnDestroy(cusolver_handle[])  # Clean up resources
    return evals, evecs
end
function diagonalize!(
    evals::CuVector{Cfloat,DeviceMemory},
    evecs::CuMatrix{Cfloat,DeviceMemory},
    H::CuMatrix{Cfloat},
)
    M, N = size(H)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    H′ = similar(H)  # Allocate a new `CuMatrix` on the GPU
    copyto!(H′, H)  # Efficiently copy data from `H` to `H′` on the GPU
    # Create cuSOLVER handle
    cusolver_handle = Ref{cusolverDnHandle_t}(C_NULL)
    cusolverDnCreate(cusolver_handle)
    # Specify cuSOLVER diag flags
    jobz = CUSOLVER_EIG_MODE_VECTOR  # Compute both singular values and singular vectors
    uplo = convert(cublasFillMode_t, 'L')  # `CUBLAS_FILL_MODE_LOWER`, see https://github.com/JuliaGPU/CUDA.jl/blob/45571e9/lib/cublas/util.jl#L49-L57
    # Determine the buffer size required
    lwork = Ref{Cint}(0)
    cusolverDnSsyevd_bufferSize(cusolver_handle[], jobz, uplo, N, H′, N, evals, lwork)
    # Allocate temporary workspace and device info array
    work = CuVector{Cfloat}(undef, lwork[])
    devInfo = CuVector{Cint}(undef, 1)
    # Diagonalize the matrix
    cusolverDnSsyevd(cusolver_handle[], jobz, uplo, N, H′, N, evals, work, lwork[], devInfo)
    # Handle errors
    retcode = only(Vector(devInfo))  # Copy memory from the GPU
    if retcode < 0
        throw(CUDAError(:cusolverDnSsyevd, "$(-retcode)th parameter is invalid!"))
    elseif retcode > 0
        throw(
            CUDAError(
                :cusolverDnSsyevd, "$(retcode)th off-diagonal elements did not converge!"
            ),
        )
    end
    copyto!(evecs, H′)  # Copy the eigenvectors to `evecs`
    cusolverDnDestroy(cusolver_handle[])  # Clean up resources
    return evals, evecs
end
function diagonalize(H::CuMatrix)
    N = size(H, 1)
    evals = CuVector{eltype(H)}(undef, N)
    evecs = CuMatrix{eltype(H)}(undef, N, N)
    return diagonalize!(evals, evecs, H)
end

# Kernel to fill diagonal elements of a square matrix
function _fill_diagonal!(A::CuDeviceMatrix{T}, D::CuDeviceVector{T}, N) where {T}
    row = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    col = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if row <= N && col <= N
        if row == col
            @inbounds A[row, col] = D[row]
        else
            @inbounds A[row, col] = zero(T)
        end
    end
    return nothing
end
function fill_diagonal!(A::CuMatrix{T}, D::CuVector{T}) where {T}
    N = size(A, 1)
    kernel = @cuda launch = false _fill_diagonal!(A, D, N)  # Prepare the kernel without launching it
    config = launch_configuration(kernel.fun)  # Get optimal launch configuration
    max_threads_per_block = config.threads  # Maximum number of threads per block
    # Determine threads per block in x and y dimensions
    # Aim for square blocks, so take the square root
    threads_per_block_dim = min(N, floor(Int, sqrt(max_threads_per_block)))
    blocks_dim = cld(N, threads_per_block_dim)
    # Launch the kernel with the calculated threads and blocks
    CUDA.@sync begin
        kernel(
            A,
            D,
            N;
            threads=(threads_per_block_dim, threads_per_block_dim),  # Threads per block in x and y dimensions
            blocks=(blocks_dim, blocks_dim),  # The number of blocks needed in each dimension
        )
    end
    return A
end

function _fermi_dirac!(result, 𝛆, μ, β)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x  # Linear thread index
    stride = gridDim().x * blockDim().x
    i = index  # Stride-based loop with a while loop, see https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
    while i <= length(result)
        @inbounds begin
            η = exp((𝛆[i] - μ) * β)
            result[i] = inv(oneunit(η) + η)
        end
        i += stride
    end
    return nothing
end
function fermi_dirac!(result::CuVector{T}, 𝛆::CuVector{T}, μ::T, β::T) where {T}
    if size(result) != size(𝛆)
        throw(DimensionMismatch("result and 𝛆 must have the same size!"))
    end
    N = length(result)
    kernel = @cuda launch = false _fermi_dirac!(result, 𝛆, μ, β)  # Compile kernel without launching
    config = launch_configuration(kernel.fun)  # Get optimal launch configuration
    threads = min(N, config.threads)  # Use the maximum allowed threads or size of array
    blocks = cld(N, threads)  # Compute required blocks to cover all elements
    # Launch the kernel with dynamic configuration
    CUDA.@sync begin
        kernel(result, 𝛆, μ, β; threads=threads, blocks=blocks)
    end
    return result
end

function fermi_dirac!(DM::CuMatrix{T}, H::CuMatrix{T}, β::T, μ::T) where {T}
    M, N = size(H)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    # Allocate eigenvalues and eigenvectors
    evals = CuVector{T}(undef, N)
    evecs = CuMatrix{T}(undef, N, N)
    # Step 1: Diagonalize the Hamiltonian
    @range "diagonalize!" begin
        diagonalize!(evals, evecs, H)
    end
    # Step 2: Apply the Fermi–Dirac function to eigenvalues
    fermi_vals = CuVector{T}(undef, N)
    @range "fermi_dirac!" begin
        fermi_dirac!(fermi_vals, evals, μ, β)
    end
    # Step 3: Compute the density matrix
    # Compute V * Diagonal(f(Λ)) * Vᵀ efficiently
    @range "density_matrix" begin
        DM .= evecs * Diagonal(fermi_vals) * evecs'
    end
    return DM
end
function fermi_dirac(H::CuMatrix, β, μ)
    DM = similar(H)
    return fermi_dirac!(DM, H, β, μ)
end

function (model::AbstractModel)(
    DM::CuMatrix, H::CuMatrix, precision, spectral_bounds=extrema(H)
)
    M, N = size(H)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    nlayers = numlayers(model)
    model = parent(model)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(spectral_bounds)
    @ccall libpath.dm_mlsp2(
        model::Ptr{Cdouble},
        H::CuPtr{Cdouble},
        DM::CuPtr{Cdouble},
        nlayers::Cint,
        N::Cint,
        precision::Cint,
        ϵₘᵢₙ::Cdouble,
        ϵₘₐₓ::Cdouble,
    )::Cvoid
    return DM
end

function compute_exact_fermi_dirac!(DM::CuMatrix, H::CuMatrix, μ, β)
    if size(DM) != size(H)
        throw(DimensionMismatch("DM and H must have the same size!"))
    end
    M, N = size(H)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    @ccall libpath.compute_exact_fermi_dirac(
        H::CuPtr{Cdouble}, N::Cint, β::Cdouble, μ::Cdouble, DM::CuPtr{Cdouble}
    )::Cvoid
    return DM
end

end
