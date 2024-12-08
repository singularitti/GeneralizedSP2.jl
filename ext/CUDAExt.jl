module CUDAExt

using CUDA:
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CUDA,
    CuMatrix,
    CuVector,
    CuDeviceMatrix,
    CuDeviceVector,
    DeviceMemory,
    blockIdx,
    blockDim,
    threadIdx,
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

using GeneralizedSP2: CUDAError

import GeneralizedSP2: diagonalize, diagonalize!, fill_diagonal!, fermi_dirac

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
function _fill_diagonal!(A::CuDeviceVector{T}, D::CuDeviceVector{T}, N) where {T}
    # Get thread index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # See https://cuda.juliagpu.org/stable/tutorials/introduction/#Writing-a-parallel-GPU-kernel
    if i <= N^2
        if mod(i - 1, N + 1) == 0
            # Along the diagonal
            A[i] = D[div(i - 1, N + 1) + 1]
        else
            # Off-diagonal entries
            A[i] = zero(T)
        end
    end
    return nothing
end
function fill_diagonal!(A::CuMatrix{T}, D::CuVector{T}) where {T}
    N = size(A, 1)
    props = CUDA.device()  # Get the device properties
    threads_per_block = CUDA.attribute(props, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    num_blocks = cld(N^2, threads_per_block)  # Set grid and block dimensions dynamically
    flat_matrix = reshape(A, :)  # Flatten the matrix
    # Launch the kernel
    CUDA.@sync begin  # See https://cuda.juliagpu.org/stable/tutorials/introduction/#Writing-a-parallel-GPU-kernel
        @cuda threads = threads_per_block blocks = num_blocks always_inline = true _fill_diagonal!(
            flat_matrix, D, N
        )
    end
    return A
end

function fermi_dirac(H::CuMatrix, μ, β) end

end
