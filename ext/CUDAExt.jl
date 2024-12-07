module CUDAExt

using CUDA: CuMatrix, CuVector
using CUDA.CUSOLVER:
    cusolverDnCreate,
    cusolverDnDsyevd_bufferSize,
    cusolverDnDsyevd,
    cusolverDnDestroy,
    CUSOLVER_EIG_MODE_VECTOR,
    cublasFillMode_t,
    cusolverDnHandle_t
using LinearAlgebra: Eigen, checksquare

import GeneralizedSP2: fermi_dirac

function diagonalize!(E::Eigen, H::CuMatrix)
    checksquare(H)
    dim = size(H, 1)
    evals, evecs = E  # Destructuring via iteration
    # Create cuSOLVER handle
    cusolver_handle = Ref{cusolverDnHandle_t}(C_NULL)
    cusolverDnCreate(cusolver_handle)
    # Specify cuSOLVER diag flags
    jobz = CUSOLVER_EIG_MODE_VECTOR  # Compute eigenvectors
    uplo = cublasFillMode_t(0)      # CUBLAS_FILL_MODE_LOWER
    # Determine the buffer size required
    lwork = Ref{Cint}(0)
    cusolverDnDsyevd_bufferSize(cusolver_handle[], jobz, uplo, dim, H, dim, evals, lwork)
    # Allocate temporary workspace and device info array
    d_work = CuArray{Float64}(undef, lwork[])
    devInfo = CuArray{Cint}(undef, 1)
    # Diagonalize the matrix
    cusolverDnDsyevd(
        cusolver_handle[], jobz, uplo, dim, H, dim, evals, d_work, lwork[], devInfo
    )
    # Copy the eigenvectors to GPU_eigvecs
    copyto!(evecs, H)
    # Clean up resources
    cusolverDnDestroy(cusolver_handle[])
    return Eigen(evals, evecs)
end
function diagonalize(H::CuMatrix)
    dim = size(H, 1)
    evals = CuVector{Float64}(undef, dim)
    evecs = CuMatrix{Float64}(undef, dim, dim)
    return diagonalize!(Eigen(evals, evecs), H)
end

function fermi_dirac(H::CuMatrix, μ, β)
    
end

end
