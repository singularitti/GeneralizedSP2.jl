using LinearAlgebra: BLAS, Diagonal, eigen, inv
using SparseArrays

export iterate_heaviside, iterate_fermi_dirac, iterate_heaviside!, iterate_fermi_dirac!

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

# function iterate_heaviside(𝐱, θ::AbstractMatrix)
#     if size(θ, 1) != LAYER_WIDTH
#         throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
#     end
#     return map(𝐱) do x
#         v = vcat(x, collect(foldlist(sp2model, x, eachcol(θ))))
#         c = vcat(last(eachrow(θ)), oneunit(eltype(θ)))
#         dot(v, c)
#     end
# end
function iterate_heaviside(x, θ::AbstractMatrix)
    if size(θ, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    y = x
    Y = zero(x)
    for θᵢ in eachcol(θ)
        Y += θᵢ[4] * y
        y = θᵢ[1] * y .^ 2 + θᵢ[2] * y + θᵢ[3] * oneunit.(y)
    end
    Y += y
    return Y
end
iterate_heaviside(x, θ::AbstractVector) = iterate_heaviside(x, reshape(θ, LAYER_WIDTH, :))
function iterate_heaviside(X::AbstractMatrix, θ::AbstractMatrix)
    if size(θ, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    y = X
    Y = zero(X)
    for θᵢ in eachcol(θ)
        Y += θᵢ[4] * y
        y = θᵢ[1] * y^2 + θᵢ[2] * y + θᵢ[3] * oneunit(y)  # Note this is not element-wise!
    end
    Y += y
    return Y
end

function iterate_fermi_dirac(x, θ)
    Y = iterate_heaviside(x, θ)
    return oneunit.(Y) - Y
end
function iterate_fermi_dirac(x::AbstractMatrix, θ)
    Y = iterate_heaviside(x, θ)
    return oneunit(Y) - Y  # Note this is not element-wise!
end

function iterate_heaviside!(res, temp1, temp2, x, θ)
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

function iterate_fermi_dirac!(res, temp1, temp2, x, θ)
    iterate_heaviside!(res, temp1, temp2, x, θ)

    n = size(x, 2)
    BLAS.scal!(n * n, -1.0, res, 1) # res *= -1
    res[1:(n + 1):(n * n)] .+= 1.0       # res += I

    return res
end
