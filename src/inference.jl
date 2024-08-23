using LinearAlgebra: BLAS, Diagonal, eigen, I, inv
using SparseArrays

"""
    matrix_function(f, A)

Compute the matrix function `f(A)` for a square matrix `A` using the eigenvalue decomposition method.

The function computes `f(A)` by diagonalizing `A`, applying the scalar function `f` to the eigenvalues,
and then reconstructing the matrix. Specifically, it performs the following steps:
1. Compute the eigenvalue decomposition of `A` as `A = V * D * V⁻¹`, where `D` is a diagonal matrix of
   eigenvalues and `V` is the matrix of eigenvectors.
2. Apply the function `f` element-wise to the eigenvalues in `D`.
3. Reconstruct the matrix as `f(A) = V * Diagonal(f(D)) * V⁻¹`.
"""
function matrix_function(f, A)
    F = eigen(A)
    return F.vectors * Diagonal(f.(F.values)) * inv(F.vectors)  # `Diagonal` is faster than `diagm`
end

function heaviside_matrix(x, θ::AbstractMatrix)
    if size(θ, 1) != 4
        throw(ArgumentError("input coefficients matrix must have 4 rows!"))
    end
    y = x
    Y = zero(x)
    for θᵢ in eachcol(θ)
        Y += θᵢ[4] * y
        y = θᵢ[1] * y^2 + θᵢ[2] * y + θᵢ[3] * I
    end
    Y += y
    return Y
end
heaviside_matrix(x, θ::AbstractVector) = heaviside_matrix(x, reshape(θ, 4, :))

function fermi_matrix(x, θ)
    Y = heaviside_matrix(x, θ)
    return I - Y
end

function heaviside_matrix!(res, temp1, temp2, x, θ)
    npts = length(x)
    n = size(x, 2)
    typ = eltype(x)
    θ = reshape(θ, LAYER_WIDTH, :)
    nlayers = size(θ, 2)

    fill!(res, 0.0)
    y = temp1
    y² = temp2
    copy!(y, x)

    for i in 1:nlayers
        BLAS.gemm!('N', 'N', one(typ), y, y, zero(typ), y²) # y² = y*y

        BLAS.axpy!(θ[4, i], y, res)
        # @. res += θ[4, i] * y

        BLAS.axpby!(θ[1, i], y², θ[2, i], y) # y = θ₁ y² + θ₂ y
        # @. y = θ[1, i] * y² + θ[2, i] * y

        y[1:(n + 1):(n * n)] .+= θ[3, i] # y += θ₃ I
    end

    BLAS.axpy!(1.0, y, res)
    # @. res += y

    return res
end

function fermi_matrix!(res, temp1, temp2, x, θ)
    heaviside_matrix!(res, temp1, temp2, x, θ)

    n = size(x, 2)
    BLAS.scal!(n * n, -1.0, res, 1) # res *= -1
    res[1:(n + 1):(n * n)] .+= 1.0       # res += I

    return res
end
