using GershgorinDiscs: eigvals_extrema
using LinearAlgebra: I, Diagonal, eigen, eigvals
using IsApprox: isunitary

export fermi_dirac,
    rescaled_fermi_dirac,
    electronic_energy,
    electronic_entropy,
    occupations,
    rescale_mu,
    rescale_beta,
    rescale_zero_one,
    rescale_one_zero

function fermi_dirac(ε, μ, β)
    η = exp((ε - μ) * β)
    return inv(oneunit(η) + η)
end
fermi_dirac(H::AbstractMatrix, μ, β) = matrix_function(ε -> fermi_dirac(ε, μ, β), H)

# function rescaled_fermi_dirac(H::AbstractMatrix, μ, β, 𝛆=eigvals_extrema(H))
#     μ′ = rescale_mu(μ, 𝛆)
#     β′ = rescale_beta(β, 𝛆)
#     H′ = H - μ′ * I
#     η = exp(H′ * β′)
#     return inv(oneunit(η) + η)
# end
function rescaled_fermi_dirac(H::AbstractMatrix, μ, β, 𝛆=eigvals_extrema(H))
    μ′ = rescale_mu(μ, 𝛆)
    β′ = rescale_beta(β, 𝛆)
    f = rescale_one_zero(𝛆)
    return matrix_function(H) do ε
        ε′ = f(ε)
        fermi_dirac(ε′, μ′, β′)
    end
end

function fermi_dirac_prime(ε, μ, β)
    fd = fermi_dirac(ε, μ, β)
    return -β * fd * (oneunit(fd) - fd)
end

function electronic_energy(ε, μ, β)
    η = (ε - μ) * β
    if η > -20oneunit(η)
        return -inv(β) * log1p(exp(-η))  # `log1p(x)` is accurate for `x` near zero
    else
        return -inv(β) * (log1p(exp(η)) - η)  # Avoid overflow for very negative `η`
    end
end

electronic_entropy(ε, μ, β) =
    (fermi_dirac(ε, μ, β) * (ε - μ) - electronic_energy(ε, μ, β)) * β

occupations(dm::AbstractMatrix) = eigvals(dm)

function rescale_mu(μ, 𝛆)
    εₘᵢₙ, εₘₐₓ = extrema(𝛆)
    @assert εₘₐₓ > εₘᵢₙ
    @assert εₘᵢₙ <= μ <= εₘₐₓ "μ must be in the range [εₘₐₓ, εₘᵢₙ]!"
    return (μ - εₘₐₓ) / (εₘᵢₙ - εₘₐₓ)
end

function rescale_beta(β, 𝛆)
    εₘᵢₙ, εₘₐₓ = extrema(𝛆)
    @assert εₘₐₓ > εₘᵢₙ
    return β * (εₘᵢₙ - εₘₐₓ)
end

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
    E = eigen(A)
    Λ, V = E.values, E.vectors
    if isunitary(V)
        return V * Diagonal(f.(Λ)) * V'
    end
    return V * Diagonal(f.(Λ)) * inv(V)  # `Diagonal` is faster than `diagm`
end

function rescale_zero_one(𝐱)
    min, max = extrema(𝐱)
    @assert min < max
    k, b = inv(max - min), min / (min - max)
    @assert !iszero(k)

    rescaler(x::Number) = k * x + b  # `x` can be out of the range [min, max]
    rescaler(X::AbstractMatrix) = k * X + b * I  # Map `max` to 1, `min` to 0

    rescaler⁻¹(y::Number) = y * (max - min) + min
    rescaler⁻¹(Y::AbstractMatrix) = (Y - b * I) / k
    Base.inv(::typeof(rescaler)) = rescaler⁻¹

    return rescaler
end
rescale_zero_one(𝐱...) = rescale_zero_one(𝐱)

function rescale_one_zero(𝐱)
    min, max = extrema(𝐱)
    @assert min < max
    k, b = inv(min - max), max / (max - min)
    @assert !iszero(k)

    rescaler(x::Number) = k * x + b  # `x` can be out of the range [min, max]
    rescaler(X::AbstractMatrix) = k * X + b * I  # Map `max` to 0, `min` to 1

    rescaler⁻¹(y::Number) = y * (min - max) + max
    rescaler⁻¹(Y::AbstractMatrix) = (Y - b * I) / k
    Base.inv(::typeof(rescaler)) = rescaler⁻¹

    return rescaler
end
rescale_one_zero(𝐱...) = rescale_one_zero(𝐱)
