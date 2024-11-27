using AffineScaler: rescale_one_zero
using LinearAlgebra: Diagonal, eigen, eigvals
using IsApprox: isunitary

export fermi_dirac,
    rescaled_fermi_dirac,
    fermi_dirac_deriv,
    electronic_energy,
    electronic_entropy,
    rescale_mu,
    recover_mu,
    rescale_beta,
    recover_beta

function fermi_dirac(ε, μ, β)
    η = exp((ε - μ) * β)
    return inv(oneunit(η) + η)
end
fermi_dirac(H::AbstractMatrix, μ, β) = matrix_function(ε -> fermi_dirac(ε, μ, β), H)

function rescaled_fermi_dirac(H::AbstractMatrix, μ, β, 𝛜=extrema(H))
    μ′ = rescale_mu(μ, 𝛜)
    β′ = rescale_beta(β, 𝛜)
    f = rescale_one_zero(𝛜)
    return matrix_function(H) do ε
        ε′ = f(ε)
        fermi_dirac(ε′, μ′, β′)
    end
end

function fermi_dirac_deriv(ε, μ, β)
    ρ = fermi_dirac(ε, μ, β)
    return -β * ρ * (oneunit(ρ) - ρ)
end
fermi_dirac_deriv(DM::AbstractMatrix, β) = -β * DM * (oneunit(DM) - DM)

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

function rescale_mu(μ, 𝛜)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(𝛜)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    @assert ϵₘᵢₙ <= μ <= ϵₘₐₓ "μ must be in the range [εₘₐₓ, εₘᵢₙ]!"
    return (μ - ϵₘₐₓ) / (ϵₘᵢₙ - ϵₘₐₓ)
end

function recover_mu(μ′, 𝛜)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(𝛜)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    @assert zero(μ′) <= μ′ <= oneunit(μ′) "rescaled μ must be in the range [0, 1]!"
    return (oneunit(μ′) - μ′) * ϵₘₐₓ + μ′ * ϵₘᵢₙ
end

function rescale_beta(β, 𝛜)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(𝛜)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    return β * (ϵₘᵢₙ - ϵₘₐₓ)
end

function recover_beta(β′, 𝛜)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(𝛜)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    return β′ / (ϵₘᵢₙ - ϵₘₐₓ)
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
