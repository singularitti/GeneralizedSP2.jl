using LinearAlgebra: I
using DifferentiationInterface
using Enzyme

export apply_model!,
    apply_model, fermi_dirac_model, entropy_model, rescale_zero_one, fermi_dirac_jacobian

function apply_model(f, T, 𝐱, 𝛉)
    result = similar(𝐱, T)
    apply_model!(f, result, 𝐱, 𝛉)
    return result
end
function apply_model(f, 𝐱, 𝛉)
    T = typeof(f(first(𝛉) * first(𝐱)))
    return apply_model(f, T, 𝐱, 𝛉)
end

function apply_model!(f, result::AbstractVector, 𝐱::AbstractVector, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        Y = zero(eltype(result))  # Accumulator of the summation
        for 𝛉 in eachcol(𝝷)
            Y += 𝛉[4] * y
            y = 𝛉[1] * y^2 + 𝛉[2] * y + 𝛉[3]
        end
        Y += y
        f(Y)
    end
    return result
end
function apply_model!(f, result::AbstractMatrix, 𝐗::AbstractMatrix, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    𝐘 = 𝐗
    map!(zero, result, result)  # `result` as the accumulator
    for 𝛉 in eachcol(𝝷)
        result += 𝛉[4] * 𝐘
        𝐘 = 𝛉[1] * 𝐘^2 + 𝛉[2] * 𝐘 + 𝛉[3] * oneunit(𝐘)  # Note this is not element-wise!
    end
    result += 𝐘
    return f(result)
end
apply_model!(f, result, 𝐱, 𝛉::AbstractVector) =
    apply_model!(f, result, 𝐱, reshape(𝛉, LAYER_WIDTH, :))

transform_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number at a time

transform_entropy(Y) = 4log(2) * (Y - Y^2)  # Applies to 1 number at a time

fermi_dirac_model!(result, 𝐱, 𝛉) = apply_model!(transform_fermi_dirac, result, 𝐱, 𝛉)

fermi_dirac_model(𝐱, 𝛉) = apply_model(transform_fermi_dirac, 𝐱, 𝛉)

entropy_model!(result, 𝐱, 𝛉) = apply_model!(transform_entropy, result, 𝐱, 𝛉)

entropy_model(𝐱, 𝛉) = apply_model(transform_entropy, 𝐱, 𝛉)

function fermi_dirac_jacobian(x, θ)
    f(x) = fermi_dirac_model(x, θ)
    return jacobian(f, AutoEnzyme(), x)
end

function compute_model_gradients!(f′, 𝗝, 𝐱, 𝛉)
    npoints = length(𝐱)
    θ = reshape(𝛉, LAYER_WIDTH, :)
    nlayers = size(θ, 2)
    𝗝 = reshape(𝗝, npoints, LAYER_WIDTH, nlayers)
    y = zeros(eltype(𝐱), nlayers + 1)

    for j in 1:npoints
        # Forward calculation
        y[1] = 𝐱[j]
        Y = zero(eltype(𝗝))
        for i in 1:nlayers
            Y += θ[4, i] * y[i]
            y[i + 1] = θ[1, i] * y[i]^2 + θ[2, i] * y[i] + θ[3, i]
        end
        Y += y[nlayers + 1]
        α = f′(Y)
        # Backward calculation
        z = 1 # z_{n+1}
        for i in nlayers:-1:1
            # z = z_{i+1}
            𝗝[j, 1, i] = α * z * y[i]^2
            𝗝[j, 2, i] = α * z * y[i]
            𝗝[j, 3, i] = α * z
            𝗝[j, 4, i] = α * y[i]

            z = θ[4, i] + z * (2θ[1, i] * y[i] + θ[2, i])
        end
    end
    return 𝗝
end

transform_fermi_dirac_derivative(Y) = -one(Y)  # Applies to 1 number at a time

transform_entropy_derivative(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_jacobian!(J, x, θ) = jacobian!(transform_fermi_dirac_derivative, J, x, θ)

entropy_jacobian!(J, x, θ) = jacobian!(transform_entropy_derivative, J, x, θ)

function rescale_zero_one(x1, x2)
    if x1 == x2
        throw(ArgumentError("inputs cannot be the same!"))
    end
    min, max = extrema((x1, x2))
    rescale(x::Number) = (x - max) / (min - max)  # `x` can be out of the range [min, max]
    function rescale(A::AbstractMatrix)
        k, b = inv(min - max), max / (max - min)
        return k * A + b * I  # Map `max` to 0, `min` to 1
    end
    return rescale
end
