using LinearAlgebra: Diagonal, eigvals, hermitianpart, ishermitian, qr
using Random: randexp

function rand_hamiltonian(syssize, α=2)
    Q, _ = qr(randn(syssize, syssize))
    𝛌 = α * randexp(syssize)
    Λ = Diagonal(𝛌)
    return collect(hermitianpart(Q * Λ * Q')), 𝛌  # Use `collect` to avoid Hermitian-specific algorithms
end

function rescale_hamiltonian(H::AbstractMatrix)
    εₘᵢₙ, εₘₐₓ = minimum(eigvals(H)) - 10, maximum(eigvals(H)) + 10
    return rescale_one_zero(εₘᵢₙ, εₘₐₓ)(H), εₘᵢₙ, εₘₐₓ
end

@testset "Test `rescale_zero_one`" begin
    𝐱 = 1:5
    X = [1.0 2.0; 3.0 4.0]
    rescaler = rescale_zero_one(𝐱)
    @testset "Test number rescaling" begin
        @test rescaler(1.0) == 0
        @test rescaler(5.0) == 1
        @test rescaler(2) == 1 / 4
        @test rescaler(3.0) == 1 / 2
        @test rescaler(4) == 3 / 4
    end
    @testset "Test number rescaling" begin
        @test rescaler(X) == [
            0 1/2
            3/4 3/4
        ]
    end
    @test_throws ArgumentError rescale_zero_one(3, 3.0)
end

@testset "Test `rescale_one_zero`" begin
    𝐱 = 5:-1:1
    X = [1.0 2.0; 3.0 4.0]
    rescaler = rescale_one_zero(𝐱)
    @testset "Test number rescaling" begin
        @test rescaler(1.0) == 1
        @test rescaler(5.0) == 0
        @test rescaler(4) == 1 / 4
        @test rescaler(3.0) == 1 / 2
        @test rescaler(2) == 3 / 4
    end
    @testset "Test matrix rescaling" begin
        @test rescaler(X) == [
            1 -1/2
            -3/4 1/4
        ]
    end
    @test_throws ArgumentError rescale_one_zero(3, 3.0)
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
        H_scaled, εₘᵢₙ, εₘₐₓ = rescale_hamiltonian(H)
        D = rescaled_fermi_dirac2(H, μ, β, (εₘᵢₙ, εₘₐₓ))
        D′ = rescaled_fermi_dirac(H, μ, β, (εₘᵢₙ, εₘₐₓ))
        @test D ≈ D′
        @test eigvals(D) ≈ eigvals(D′)
    end
end
