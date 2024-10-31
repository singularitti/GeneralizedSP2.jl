using GershgorinDiscs: eigvals_extrema
using LinearAlgebra: I, Diagonal, eigen, eigvals

export fermi_dirac, rescaled_fermi_dirac, electronic_energy, electronic_entropy, occupations

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
