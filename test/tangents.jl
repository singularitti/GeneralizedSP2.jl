@testset "Test computing the model's gradients" begin
    model = Model(rand(4, 10))
    𝐱 = rand(100)

    f(x) = x^3 + 2.5x^2 + 4x
    f′(x) = 3x^2 + 5x + 4
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄

    f(y) = exp(y)
    f′(y) = exp(y)
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄

    f(y) = sin(y)
    f′(y) = cos(y)
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄

    f(y) = log(y)
    f′(y) = 1 / y
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄

    f(y) = tanh(y)
    f′(y) = 1 - tanh(y)^2
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄

    f(y) = 1 / y
    f′(y) = -1 / y^2
    𝐱 = rand(100)
    𝝝̄ = manualdiff_model(f′, 𝐱, model)  # Benchmark result
    @test autodiff_model(f, 𝐱, model) ≈ 𝝝̄
end
