using LinearAlgebra: Diagonal, eigvals, hermitianpart, ishermitian, qr
using Random: randexp

function rand_hamiltonian(syssize, α=2)
    Q, _ = qr(randn(syssize, syssize))
    𝛌 = α * randexp(syssize)
    Λ = Diagonal(𝛌)
    return collect(hermitianpart(Q * Λ * Q')), 𝛌  # Use `collect` to avoid Hermitian-specific algorithms
end

function rescaled_hamiltonian(H::AbstractMatrix)
    εₘᵢₙ, εₘₐₓ = minimum(eigvals(H)) - 10, maximum(eigvals(H)) + 10
    return rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H), εₘᵢₙ, εₘₐₓ
end

@testset "Test `eigvals` will return a random order of eigenvalues" begin
    H, 𝛌 = rand_hamiltonian(1024, 0.54)
    @assert ishermitian(H)
    @test !≈(eigvals(H), 𝛌)  # Not sorted
    @test sort(eigvals(H)) ≈ sort(𝛌)
    @testset "Test on the Fermi–Dirac function" begin
        μ, β = 0.35, 4
        D = fermi_dirac(H, μ, β)
        @test !≈(eigvals(D), fermi_dirac.(eigvals(H), μ, β))  # Not sorted
        @test sort(eigvals(D)) ≈ fermi_dirac.(reverse(sort(eigvals(H))), μ, β)  # We need to reverse it since it is Fermi–Dirac
    end
    @testset "Test on the rescaled Fermi–Dirac function" begin
        μ, β = 0.35, 4
        H_scaled, εₘᵢₙ, εₘₐₓ = rescaled_hamiltonian(H)
        D = rescaled_fermi_dirac2(H, μ, β, (εₘᵢₙ, εₘₐₓ))
        D′ = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
        @test D ≈ D′
        @test eigvals(D) ≈ eigvals(D′)
    end
end
