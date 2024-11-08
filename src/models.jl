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
    rescale_one_zero

# See https://github.com/PainterQubits/Unitful.jl/blob/6bf6f99/src/utils.jl#L241-L247
struct DimensionError{X,Y} <: Exception
    x::X
    y::Y
end

Base.showerror(io::IO, e::DimensionError) =
    print(io, "DimensionError: $(e.x) and $(e.y) are not dimensionally compatible.")

function apply_model(x, Θ::AbstractMatrix{T}) where {T}
    if size(Θ, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    accumulator = zero(x * oneunit(T))  # Accumulator of the summation
    y = x  # `x` and `y` are 2 numbers
    for 𝛉 in eachcol(Θ)
        accumulator += 𝛉[4] * y
        y = 𝛉[1] * y^2 + 𝛉[2] * y + 𝛉[3] * oneunit(y)
    end
    accumulator += oneunit(T) * y
    return accumulator
end
function apply_model(X::AbstractMatrix{S}, Θ::AbstractMatrix{T}) where {S,T}
    checksquare(X)  # See https://discourse.julialang.org/t/120556/2
    if size(Θ, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    accumulator = zeros(typeof(oneunit(S) * oneunit(T)), size(X))
    Y = X
    for 𝛉 in eachcol(Θ)
        accumulator += 𝛉[4] * Y
        Y = 𝛉[1] * Y^2 + 𝛉[2] * Y + 𝛉[3] * oneunit(Y)  # Note this is not element-wise!
    end
    accumulator += oneunit(T) * Y
    return accumulator
end
apply_model(𝐱, 𝛉::AbstractVector) = apply_model(𝐱, reshape(𝛉, LAYER_WIDTH, :))

function apply_model!(
    result::AbstractVector{R}, 𝐱::AbstractVector{S}, Θ::AbstractMatrix{T}
) where {R,S,T}
    if size(Θ, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    if !isa(oneunit(S) * oneunit(T), R)
        throw(DimensionError(oneunit(S) * oneunit(T), oneunit(R)))
    end
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        accumulator = zero(eltype(result))  # Accumulator of the summation
        for 𝛉 in eachcol(Θ)
            accumulator += 𝛉[4] * y
            y = 𝛉[1] * y^2 + 𝛉[2] * y + 𝛉[3] * oneunit(y)
        end
        accumulator += oneunit(T) * y
    end
    return result
end
function apply_model!(
    result::AbstractMatrix{R}, X::AbstractMatrix{S}, 𝝷::AbstractMatrix{T}
) where {R,S,T}
    checksquare(X)  # See https://discourse.julialang.org/t/120556/2
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    if !isa(oneunit(S) * oneunit(T), R)
        throw(DimensionError(oneunit(S) * oneunit(T), oneunit(R)))
    end
    map!(zero, result, result)
    Y = X
    for 𝛉 in eachcol(𝝷)
        result += 𝛉[4] * Y
        Y = 𝛉[1] * Y^2 + 𝛉[2] * Y + 𝛉[3] * oneunit(Y)  # Note this is not element-wise!
    end
    result += oneunit(T) * Y
    return result
end
apply_model!(result, 𝐱, 𝛉::AbstractVector) =
    apply_model!(result, 𝐱, reshape(𝛉, LAYER_WIDTH, :))

finalize_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number/matrix at a time

function fermi_dirac_model(𝐱::AbstractVector, Θ)
    return map(𝐱) do x
        finalize_fermi_dirac(apply_model(x, Θ))  # This is element-wise!
    end
end
function fermi_dirac_model(X::AbstractMatrix, Θ)
    intermediate = apply_model(X, Θ)
    return finalize_fermi_dirac(intermediate)  # Note this is not element-wise!
end

function fermi_dirac_model!(result::AbstractVector, 𝐱::AbstractVector, Θ)
    return map!(result, 𝐱) do x
        finalize_fermi_dirac(apply_model(x, Θ))  # This is element-wise!
    end
end
function fermi_dirac_model!(result::AbstractMatrix, X::AbstractMatrix, Θ)
    intermediate = apply_model(X, Θ)
    copy!(result, finalize_fermi_dirac(intermediate))  # Note this is not element-wise!
    return result
end

finalize_entropy(Y) = 4log(2) * (Y - Y^2)  # Applies to 1 number/matrix at a time

function entropy_model(𝐱::AbstractVector, Θ)
    return map(𝐱) do x
        finalize_entropy(apply_model(x, Θ))  # This is element-wise!
    end
end
function entropy_model(X::AbstractMatrix, Θ)
    intermediate = apply_model(X, Θ)
    return finalize_entropy(intermediate)  # Note this is not element-wise!
end

function entropy_model!(result::AbstractVector, 𝐱::AbstractVector, Θ)
    return map!(result, 𝐱) do x
        finalize_entropy(apply_model(x, Θ))  # This is element-wise!
    end
end
function entropy_model!(result::AbstractMatrix, X::AbstractMatrix, Θ)
    intermediate = apply_model(X, Θ)
    copy!(result, finalize_entropy(intermediate))  # Note this is not element-wise!
    return result
end

function autodiff_model(f, 𝐱, Θ)
    𝝝̄ = Array{eltype(Θ)}(undef, size(𝐱)..., size(Θ)...)
    return autodiff_model!(f, 𝝝̄, 𝐱, Θ)
end

function autodiff_model!(f, 𝝝̄, 𝐱, Θ)
    function _apply_model!(𝐲, 𝐱, Θ)
        apply_model!(f, 𝐲, 𝐱, Θ)
        return nothing
    end

    foreach(enumerate(𝐱)) do (i, x)
        y = zeros(1)
        ȳ = ones(1)
        Θ̄ = zero(Θ)
        autodiff(Reverse, _apply_model!, Duplicated(y, ȳ), Const([x]), Duplicated(Θ, Θ̄))
        𝝝̄[i, :, :] = Θ̄
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

function rescale_one_zero(x1, x2)
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
