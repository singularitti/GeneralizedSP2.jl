using LinearAlgebra: I
using Enzyme: Reverse, Const, Duplicated, autodiff

export apply_model!,
    apply_model,
    autodiff_model!,
    autodiff_model,
    fermi_dirac_model,
    entropy_model,
    rescale_zero_one,
    autodiff_model

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

function autodiff_model(f, 𝐱, 𝝷)
    𝗝 = Array{eltype(𝝷)}(undef, size(𝐱)..., size(𝝷)...)
    return autodiff_model!(f, 𝗝, 𝐱, 𝝷)
end

function autodiff_model!(f, 𝗝, 𝐱, 𝝷)
    function _apply_model!(𝐲, 𝐱, 𝝷)
        apply_model!(f, 𝐲, 𝐱, 𝝷)
        return nothing
    end

    foreach(enumerate(𝐱)) do (i, x)
        y = zeros(size([x]))
        ȳ = ones(size(y))
        𝝷̄ = zero(𝝷)
        autodiff(Reverse, _apply_model!, Duplicated(y, ȳ), Const([x]), Duplicated(𝝷, 𝝷̄))
        𝗝[i, :, :] = 𝝷̄
    end
    return 𝗝
end

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
