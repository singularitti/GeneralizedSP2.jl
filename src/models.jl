using LinearAlgebra: I, checksquare
using Enzyme: Reverse, Const, Duplicated, autodiff

export apply_model!,
    apply_model,
    autodiff_model!,
    autodiff_model,
    manualdiff_model!,
    manualdiff_model,
    fermi_dirac_model,
    entropy_model,
    rescale_zero_one,
    rescale_back

function apply_model(f, T, 𝐱, 𝛉)
    result = similar(𝐱, T)
    apply_model!(f, result, 𝐱, 𝛉)
    return result
end
function apply_model(f, 𝐱, 𝛉)
    T = typeof(f(first(𝛉) * first(𝐱)))
    return apply_model(f, T, 𝐱, 𝛉)
end

function apply_model(x, 𝝷::AbstractMatrix{T}) where {T}
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    y = x  # `x` and `y` are 2 numbers
    accumulator = zero(typeof(x * oneunit(T)))  # Accumulator of the summation
    for 𝛉 in eachcol(𝝷)
        accumulator += 𝛉[4] * y
        y = 𝛉[1] * y^2 + 𝛉[2] * y + 𝛉[3] * oneunit(y)
    end
    accumulator += y
    return accumulator
end
function apply_model(𝗫::AbstractMatrix{X}, 𝝷::AbstractMatrix{T}) where {X,T}
    checksquare(𝗫)  # See https://discourse.julialang.org/t/120556/2
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    accumulator = zeros(typeof(oneunit(X) * oneunit(T)), size(𝗫))
    𝗬 = 𝗫
    for 𝛉 in eachcol(𝝷)
        accumulator += 𝛉[4] * 𝗬
        𝗬 = 𝛉[1] * 𝗬^2 + 𝛉[2] * 𝗬 + 𝛉[3] * oneunit(𝗬)  # Note this is not element-wise!
    end
    accumulator += 𝗬
    return accumulator
end

function apply_model!(f, result::AbstractVector, 𝐱::AbstractVector, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        accumulator = zero(eltype(result))  # Accumulator of the summation
        for 𝛉 in eachcol(𝝷)
            accumulator += 𝛉[4] * y
            y = 𝛉[1] * y^2 + 𝛉[2] * y + 𝛉[3] * oneunit(y)
        end
        accumulator += y
        f(accumulator)
    end
    return result
end
function apply_model!(f, result::AbstractMatrix, 𝐗::AbstractMatrix, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    T = typeof(f(first(𝝷) * first(𝐗)))
    accumulator = zeros(T, size(𝐗))  # Remeber to make it zero matrix!
    𝐘 = 𝐗
    for 𝛉 in eachcol(𝝷)
        accumulator += 𝛉[4] * 𝐘
        𝐘 = 𝛉[1] * 𝐘^2 + 𝛉[2] * 𝐘 + 𝛉[3] * oneunit(𝐘)  # Note this is not element-wise!
    end
    accumulator += 𝐘
    copy!(result, f(accumulator))
    return result
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
    𝝝̄ = Array{eltype(𝝷)}(undef, size(𝐱)..., size(𝝷)...)
    return autodiff_model!(f, 𝝝̄, 𝐱, 𝝷)
end

function autodiff_model!(f, 𝝝̄, 𝐱, 𝝷)
    function _apply_model!(𝐲, 𝐱, 𝝷)
        apply_model!(f, 𝐲, 𝐱, 𝝷)
        return nothing
    end

    foreach(enumerate(𝐱)) do (i, x)
        y = zeros(1)
        ȳ = ones(1)
        𝝷̄ = zero(𝝷)
        autodiff(Reverse, _apply_model!, Duplicated(y, ȳ), Const([x]), Duplicated(𝝷, 𝝷̄))
        𝝝̄[i, :, :] = 𝝷̄
    end
    return 𝝝̄
end

function manualdiff_model(f′, 𝐱, 𝝷)
    𝝷 = reshape(𝝷, LAYER_WIDTH, :)
    𝝝̄ = Array{Float64}(undef, size(𝐱)..., size(𝝷)...)
    return manualdiff_model!(f′, 𝝝̄, 𝐱, 𝝷)
end

function manualdiff_model!(f′, 𝝝̄, 𝐱, 𝝷)
    npoints = length(𝐱)
    𝝷 = reshape(𝝷, LAYER_WIDTH, :)
    nlayers = size(𝝷, 2)
    𝝝̄ = reshape(𝝝̄, size(𝐱)..., size(𝝷)...)
    𝐲 = zeros(eltype(𝐱), nlayers + 1)
    for j in 1:npoints
        # Forward calculation
        𝐲[1] = 𝐱[j]
        Y = zero(eltype(𝐲))
        for i in 1:nlayers
            Y += 𝝷[4, i] * 𝐲[i]
            𝐲[i + 1] = 𝝷[1, i] * 𝐲[i]^2 + 𝝷[2, i] * 𝐲[i] + 𝝷[3, i] * oneunit(𝐲[i])
        end
        Y += 𝐲[nlayers + 1]
        α = f′(Y)
        # Backward calculation
        z = one(eltype(𝝷)) # zₗₐₛₜ
        for i in nlayers:-1:1
            # zᵢ₊₁
            𝝝̄[j, 1, i] = α * z * 𝐲[i]^2
            𝝝̄[j, 2, i] = α * z * 𝐲[i]
            𝝝̄[j, 3, i] = α * z
            𝝝̄[j, 4, i] = α * 𝐲[i]
            z = 𝝷[4, i] * oneunit(𝐲[i]) + z * (2𝝷[1, i] * 𝐲[i] + 𝝷[2, i] * oneunit(𝐲[i]))  # zᵢ
        end
    end
    return 𝝝̄
end

transform_fermi_dirac_derivative(Y) = -one(Y)  # Applies to 1 number at a time

transform_entropy_derivative(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_derivatives!(𝝝̄, 𝐱, 𝝷) =
    manualdiff_model!(transform_fermi_dirac_derivative, 𝝝̄, 𝐱, 𝝷)
# fermi_dirac_derivatives!(𝝝̄, 𝐱, 𝝷) = autodiff_model!(transform_fermi_dirac, 𝝝̄, 𝐱, 𝝷)

entropy_derivatives!(𝝝̄, 𝐱, 𝝷) = manualdiff_model!(transform_entropy_derivative, 𝝝̄, 𝐱, 𝝷)
# entropy_derivatives!(𝝝̄, 𝐱, 𝝷) = autodiff_model!(transform_entropy, 𝝝̄, 𝐱, 𝝷)

function rescale_zero_one(x1, x2)
    if x1 == x2
        throw(ArgumentError("inputs cannot be the same!"))
    end
    min, max = extrema((x1, x2))
    rescale(x::Number) = (x - min) / (max - min)  # `x` can be out of the range [min, max]
    function rescale(A::AbstractMatrix)
        k, b = inv(max - min), -min / (max - min)
        return k * A + b * I  # Map `max` to 1, `min` to 0
    end
    return rescale
end

function rescale_back(x1, x2)
    min, max = extrema((x1, x2))
    rescale(y::Number) = y * (max - min) + min
    rescale(A::AbstractMatrix) = (max - min) * A + min * I
    return rescale
end
