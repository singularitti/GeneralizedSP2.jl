using GershgorinDiscs: eigvals_extrema
using LinearAlgebra: I, Diagonal, eigen, eigvals
using IsApprox: isunitary

export fermi_dirac,
    rescaled_fermi_dirac,
    rescaled_fermi_dirac2,
    electronic_energy,
    electronic_entropy,
    occupations

function fermi_dirac(ε, μ, β)
    η = exp((ε - μ) * β)
    return inv(oneunit(η) + η)
end
function fermi_dirac(𝐇::AbstractMatrix, μ, β)
    η = exp((𝐇 - μ * I) * β)
    return inv(oneunit(η) + η)
end

function rescaled_fermi_dirac(𝐇::AbstractMatrix, μ, β, (εₘᵢₙ, εₘₐₓ)=eigvals_extrema(𝐇))
    𝐇′ = -𝐇 + (εₘₐₓ * (oneunit(μ) - μ) + μ * εₘᵢₙ) * I
    β′ = β / (εₘₐₓ - εₘᵢₙ)
    η = exp(𝐇′ * β′)
    return inv(oneunit(η) + η)
end
function rescaled_fermi_dirac2(𝐇::AbstractMatrix, μ, β, (εₘᵢₙ, εₘₐₓ)=eigvals_extrema(𝐇))
    E = eigen(𝐇)
    𝛌_scaled = rescale_one_zero(εₘᵢₙ, εₘₐₓ).(E.values)
    Λ = map(𝛌_scaled) do λ_scaled
        fermi_dirac(λ_scaled, μ, β)
    end
    return E.vectors * Diagonal(Λ) * E.vectors'
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
