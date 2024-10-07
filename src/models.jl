using LinearAlgebra: I

export apply_model!, apply_model, fermi_dirac_model, entropy_model, rescale_zero_one

function apply_model(f, T, 𝛉, 𝐱)
    result = similar(𝐱, T)
    apply_model!(f, result, 𝛉, 𝐱)
    return result
end
function apply_model(f, 𝛉, 𝐱)
    T = typeof(f(first(𝛉) * first(𝐱)))
    return apply_model(f, T, 𝛉, 𝐱)
end

function apply_model!(f, result::AbstractVector, 𝝷::AbstractMatrix, 𝐱::AbstractVector)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        Y = zero(eltype(result))  # Accumulator of the summation
        for θᵢ in eachcol(𝝷)
            Y += θᵢ[4] * y
            y = θᵢ[1] * y^2 + θᵢ[2] * y + θᵢ[3]
        end
        Y += y
        f(Y)
    end
    return result
end
function apply_model!(f, result::AbstractMatrix, 𝝷::AbstractMatrix, 𝐗::AbstractMatrix)
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
apply_model!(f, result, 𝛉::AbstractVector, 𝐱) =
    apply_model!(f, result, reshape(𝛉, LAYER_WIDTH, :), 𝐱)

transform_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number at a time

transform_entropy(Y) = 4log(2) * (Y - Y^2)  # Applies to 1 number at a time

fermi_dirac_model!(result, 𝛉, 𝐱) = apply_model!(transform_fermi_dirac, result, 𝛉, 𝐱)

fermi_dirac_model(𝛉, 𝐱) = apply_model(transform_fermi_dirac, 𝛉, 𝐱)

entropy_model!(result, 𝛉, 𝐱) = apply_model!(transform_entropy, result, 𝛉, 𝐱)

entropy_model(𝛉, 𝐱) = apply_model(transform_entropy, 𝛉, 𝐱)

function jacobian!(J::AbstractMatrix, x, θ, df_dY)
    npoints = length(x)
    θ = reshape(θ, LAYER_WIDTH, :)
    nlayers = size(θ, 2)

    J = reshape(J, npoints, LAYER_WIDTH, nlayers)
    y = zeros(eltype(x), nlayers + 1)

    for j in 1:npoints

        # forward calculation
        y[1] = x[j]
        Y = zero(eltype(J))
        for i in 1:nlayers
            Y += θ[4, i] * y[i]
            y[i + 1] = θ[1, i] * y[i]^2 + θ[2, i] * y[i] + θ[3, i]
        end
        Y += y[nlayers + 1]
        α = df_dY(Y)

        # backward calculation
        z = 1 # z_{n+1}
        for i in nlayers:-1:1
            # z = z_{i+1}
            J[j, 1, i] = α * z * y[i]^2
            J[j, 2, i] = α * z * y[i]
            J[j, 3, i] = α * z
            J[j, 4, i] = α * y[i]

            z = θ[4, i] + z * (2θ[1, i] * y[i] + θ[2, i])
        end
    end
end

transform_fermi_dirac_derivative(Y) = -one(Y)  # Applies to 1 number at a time

transform_entropy_derivative(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_jacobian!(J, x, θ) = jacobian!(J, x, θ, transform_fermi_dirac_derivative)

entropy_jacobian!(J, x, θ) = jacobian!(J, x, θ, transform_entropy_derivative)

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
