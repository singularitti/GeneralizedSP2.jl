using LinearAlgebra: checksquare, axpy!, axpby!, mul!
using LinearAlgebra.BLAS: scal!
# using Enzyme: Reverse, Const, Duplicated, autodiff

export basis, electronic_entropy, fermi_dirac!

const FOUR_LOG_TWO = 4log(2)

# See https://github.com/PainterQubits/Unitful.jl/blob/6bf6f99/src/utils.jl#L241-L247
struct DimensionError{X,Y} <: Exception
    x::X
    y::Y
end

Base.showerror(io::IO, e::DimensionError) =
    print(io, "DimensionError: $(e.x) and $(e.y) are not dimensionally compatible.")

function basis(model::AbstractModel)
    function _get(x)
        y = x  # `x` and `y` are 2 numbers
        collector = Vector{typeof(oneunit(x) * oneunit(eltype(model)))}(
            undef, numlayers(model) + 1
        )
        for (i, 𝐦) in enumerate(eachlayer(model))
            collector[i] = 𝐦[4] * y
            y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
        end
        collector[end] = oneunit(eltype(model)) * y
        return collector
    end
    return _get
end

function (model::AbstractModel)(x)
    y = x  # `x` and `y` are 2 numbers (not big numbers)
    𝟏, 𝟏′ = oneunit(eltype(model)), oneunit(y)
    accumulator = zero(𝟏 * x)  # Accumulator of the summation
    for 𝐦 in eachlayer(model)
        accumulator += 𝐦[4] * y
        y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * 𝟏′
    end
    accumulator += 𝟏 * y
    return accumulator
end
function (model::AbstractModel)(X::AbstractMatrix)
    result = similar(X, typeof(oneunit(eltype(model)) * oneunit(eltype(X))))  # Prepare for in-place result
    return model(result, X)
end
function (model::AbstractModel)(result::AbstractMatrix, X::AbstractMatrix)
    checksquare(X)  # See https://discourse.julialang.org/t/120556/2
    if !iszero(X)  # Very fast
        map!(zero, result, result)
    end
    Y = deepcopy(X)  # Modifying `Y` does not change `X` now
    Y² = similar(Y)
    𝟙 = oneunit(Y)  # Identity matrix
    for 𝐦 in eachlayer(model)  # All operations are in-place, significantly reducing allocations.
        axpy!(𝐦[4], Y, result)  # result .+= 𝐦[4] * Y
        mul!(Y², Y, Y)  # Y² .= Y^2
        axpby!(𝐦[1], Y², 𝐦[2], Y)  # Y .+= 𝐦[1] * Y^2 + 𝐦[2] * Y
        axpy!(𝐦[3], 𝟙, Y)  # Y .+= 𝐦[3] * 𝟙
    end
    axpy!(oneunit(eltype(model)), Y, result)  # result .+= oneunit(eltype(model)) * Y
    return result
end

function Base.map!(model::AbstractModel, result::AbstractVector, 𝐱::AbstractVector)
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        accumulator = zero(eltype(result))  # Accumulator of the summation
        for 𝐦 in eachlayer(model)
            accumulator += 𝐦[4] * y
            y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
        end
        accumulator += oneunit(eltype(model)) * y
    end
    return result
end

_finalize_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number/matrix at a time
_finalize_fermi_dirac!(Y::AbstractMatrix) = axpby!(1, oneunit(Y), -1, Y)  # This is the fastest, except for `axpy!(-1, result, oneunit(Y))`, which we cannot use here.

fermi_dirac(model::AbstractModel) = _finalize_fermi_dirac ∘ model
fermi_dirac!(model::AbstractModel) = _finalize_fermi_dirac! ∘ model

_finalize_electronic_entropy(Y) = FOUR_LOG_TWO * (Y - Y^2)  # Applies to 1 number/matrix at a time
function _finalize_electronic_entropy!(Y::AbstractMatrix)
    Y² = similar(Y)
    mul!(Y², Y, Y)  # Y² .= Y^2
    axpy!(-1, Y², Y)  # Y .= Y - Y²
    scal!(FOUR_LOG_TWO, Y)  # Y .= 4log(2) * Y
    return Y
end

electronic_entropy(model::AbstractModel) = _finalize_electronic_entropy ∘ model

electronic_entropy!(model::AbstractModel, result::AbstractVector, 𝐱::AbstractVector) =
    map!(electronic_entropy(model), result, 𝐱)
electronic_entropy!(model::AbstractModel) = _finalize_electronic_entropy! ∘ model
