@testset "Test computing the model's gradients" begin
    function compute_model_gradients(f′, 𝐱, 𝝷)
        npoints = length(𝐱)
        nlayers = size(𝝷, 2)

        𝝷̄ = Array{Float64}(undef, size(𝐱)..., size(𝝷)...)
        𝐲 = zeros(eltype(𝐱), nlayers + 1)
        for j in 1:npoints
            # Forward calculation
            𝐲[1] = 𝐱[j]
            Y = zero(eltype(𝐲))
            for i in 1:nlayers
                Y += 𝝷[4, i] * 𝐲[i]
                𝐲[i + 1] = 𝝷[1, i] * 𝐲[i]^2 + 𝝷[2, i] * 𝐲[i] + 𝝷[3, i]
            end
            Y += 𝐲[nlayers + 1]
            α = f′(Y)
            # Backward calculation
            z = one(eltype(𝝷)) # zₗₐₛₜ
            for i in nlayers:-1:1
                # zᵢ₊₁
                𝝷̄[j, 1, i] = α * z * 𝐲[i]^2
                𝝷̄[j, 2, i] = α * z * 𝐲[i]
                𝝷̄[j, 3, i] = α * z
                𝝷̄[j, 4, i] = α * 𝐲[i]
                z = 𝝷[4, i] + z * (2𝝷[1, i] * 𝐲[i] + 𝝷[2, i])  # zᵢ
            end
        end
        return 𝝷̄
    end

    𝝷₀ = rand(4, 10)
    f(x) = x^3 + 2.5x^2 + 4x
    f′(x) = 3x^2 + 5x + 4
    𝐱 = rand(100)
    𝝷̄ = compute_model_gradients(f′, 𝐱, 𝝷₀)  # Benchmark result
    @test autodiff_model(f, 𝐱, 𝝷₀) ≈ 𝝷̄
end
