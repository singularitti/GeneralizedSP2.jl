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

using GeneralizedSP2: CUDAError

import GeneralizedSP2: fermi_dirac

function diagonalize!(E::Eigen, H::CuMatrix{Cdouble})
    checksquare(H)
    N = size(H, 1)
    evals, evecs = E  # Destructuring via iteration
    # Create cuSOLVER handle
    cusolver_handle = Ref{cusolverDnHandle_t}(C_NULL)
    cusolverDnCreate(cusolver_handle)
    # Specify cuSOLVER diag flags
    jobz = CUSOLVER_EIG_MODE_VECTOR  # Compute both singular values and singular vectors
    uplo = cublasFillMode_t(0)      # CUBLAS_FILL_MODE_LOWER
    # Determine the buffer size required
    lwork = Ref{Cint}(0)
    cusolverDnDsyevd_bufferSize(cusolver_handle[], jobz, uplo, N, H, N, evals, lwork)
    # Allocate temporary workspace and device info array
    work = CuMatrix{Cdouble}(undef, lwork[])
    devInfo = CuVector{Cint}(undef, 1)
    # Diagonalize the matrix
    cusolverDnDsyevd(cusolver_handle[], jobz, uplo, N, H, N, evals, work, lwork[], devInfo)
    # Handle errors
    retcode = only(devInfo)
    if devInfo[1] < 0
        throw(CUDAError(:cusolverDnDsyevd, "$(-retcode)th parameter is invalid!"))
    elseif devInfo[1] > 0
        throw(
            CUDAError(
                :cusolverDnDsyevd, "($retcode)th off-diagonal elements did not converge!"
            ),
        )
    end
    # Copy the eigenvectors to GPU_eigvecs
    copyto!(evecs, H)
    # Clean up resources
    cusolverDnDestroy(cusolver_handle[])
    return Eigen(evals, evecs)
end
function diagonalize(H::CuMatrix)
    dim = size(H, 1)
    evals = CuVector{Cdouble}(undef, dim)
    evecs = CuMatrix{Cdouble}(undef, dim, dim)
    return diagonalize!(Eigen(evals, evecs), H)
end

function fermi_dirac(H::CuMatrix, μ, β)
    
end

end
