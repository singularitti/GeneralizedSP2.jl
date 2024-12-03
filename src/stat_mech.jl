using AffineScaler: Scaler, rescale_one_zero
using LinearAlgebra: Diagonal, eigen, eigvals
using IsApprox: Approx, isunitary

export fermi_dirac,
    rescaled_fermi_dirac,
    fermi_dirac_deriv,
    electronic_energy,
    electronic_entropy,
    rescale_mu,
    rescale_beta

function fermi_dirac(ε, μ, β)
    η = exp((ε - μ) * β)
    return inv(oneunit(η) + η)
end
fermi_dirac(H::AbstractMatrix, μ, β) = matrix_function(ε -> fermi_dirac(ε, μ, β), H)

function rescaled_fermi_dirac(H::AbstractMatrix, μ, β, spectral_bounds=extrema(H))
    μ′ = rescale_mu(spectral_bounds)(μ)
    β′ = rescale_beta(spectral_bounds)(β)
    f = rescale_one_zero(spectral_bounds)
    return matrix_function(H) do ε
        ε′ = f(ε)
        fermi_dirac(ε′, μ′, β′)
    end
end

function fermi_dirac_deriv(ε, μ, β)
    ρ = fermi_dirac(ε, μ, β)
    return -β * ρ * (oneunit(ρ) - ρ)
end
fermi_dirac_deriv(D::AbstractMatrix, β) = -β * D * (oneunit(D) - D)

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

function rescale_mu(spectral_bounds)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(spectral_bounds)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    k = inv(ϵₘᵢₙ - ϵₘₐₓ)
    b = -ϵₘₐₓ * k
    return Scaler(k, b)
end

function rescale_beta(spectral_bounds)
    ϵₘᵢₙ, ϵₘₐₓ = extrema(spectral_bounds)
    @assert ϵₘₐₓ > ϵₘᵢₙ
    k = ϵₘᵢₙ - ϵₘₐₓ
    b = zero(k)
    return Scaler(k, b)
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
    if isunitary(V, Approx(; rtol=isapprox_rtol()))
        return V * Diagonal(f.(Λ)) * V'
    end
    return V * Diagonal(f.(Λ)) * inv(V)  # `Diagonal` is faster than `diagm`
end

const ISAPPROX_RTOL = Ref(√eps())

isapprox_rtol() = ISAPPROX_RTOL[]
# See https://github.com/KristofferC/OhMyREPL.jl/blob/8b0fc53/src/BracketInserter.jl#L44-L45
set_isapprox_rtol(rtol) = ISAPPROX_RTOL[] = rtol
