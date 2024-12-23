using LinearAlgebra: I, checksquare, axpy!, axpby!, mul!
# using Enzyme: Reverse, Const, Duplicated, autodiff

export basis, electronic_entropy, fermi_dirac!

# See https://github.com/PainterQubits/Unitful.jl/blob/6bf6f99/src/utils.jl#L241-L247
struct DimensionError{X,Y} <: Exception
    x::X
    y::Y
end

Base.showerror(io::IO, e::DimensionError) =
    print(io, "DimensionError: $(e.x) and $(e.y) are not dimensionally compatible.")

function basis(M::AbstractModel)
    function _get(x)
        y = x  # `x` and `y` are 2 numbers
        collector = Vector{typeof(oneunit(x) * oneunit(eltype(M)))}(undef, numlayers(M) + 1)
        for (i, 𝐦) in enumerate(eachlayer(M))
            collector[i] = 𝐦[4] * y
            y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
        end
        collector[end] = oneunit(eltype(M)) * y
        return collector
    end
    return _get
end

function (M::AbstractModel)(x)
    𝟏 = oneunit(eltype(M))
    accumulator = zero(x * 𝟏)  # Accumulator of the summation
    y = x  # `x` and `y` are 2 numbers
    for 𝐦 in eachlayer(M)
        accumulator += 𝐦[4] * y
        y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
    end
    accumulator += 𝟏 * y
    return accumulator
end
function (M::AbstractModel)(X::AbstractMatrix)
    result = similar(X, typeof(oneunit(eltype(M)) * oneunit(eltype(X))))  # Prepare for in-place result
    return M(result, X)
end
function (M::AbstractModel)(result::AbstractMatrix, X::AbstractMatrix)
    checksquare(X)  # See https://discourse.julialang.org/t/120556/2
    map!(zero, result, result)
    Y = copy(X)  # Modifying `Y` does not change `X` now
    Y² = similar(Y)
    I = oneunit(Y)  # Identity matrix
    for 𝐦 in eachlayer(M)  # All operations are in-place, significantly reducing allocations.
        axpy!(𝐦[4], Y, result)  # result .+= 𝐦[4] * Y
        mul!(Y², Y, Y)  # Y² .= Y^2
        axpby!(𝐦[1], Y², 𝐦[2], Y)  # Y .+= 𝐦[1] * Y^2 + 𝐦[2] * Y
        axpy!(𝐦[3], I, Y)  # Y .+= 𝐦[3] * I
    end
    result .+= oneunit(eltype(M)) * Y
    return result
end

function Base.map!(
    M::AbstractModel{T}, result::AbstractVector{R}, 𝐱::AbstractVector{S}
) where {R,S,T}
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        accumulator = zero(eltype(result))  # Accumulator of the summation
        for 𝐦 in eachlayer(M)
            accumulator += 𝐦[4] * y
            y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
        end
        accumulator += oneunit(T) * y
    end
    return result
end

_finalize_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number/matrix at a time

fermi_dirac(M::AbstractModel) = _finalize_fermi_dirac ∘ M

fermi_dirac!(M::AbstractModel, result::AbstractVector, 𝐱::AbstractVector) =
    map!(fermi_dirac(M), result, 𝐱)
fermi_dirac!(M::AbstractModel, result::AbstractMatrix, X::AbstractMatrix) =
    copy!(result, fermi_dirac(M)(X))  # Note this is not element-wise!

_finalize_electronic_entropy(Y) = 4log(2) * (Y - Y^2)  # Applies to 1 number/matrix at a time

electronic_entropy(M::AbstractModel) = _finalize_electronic_entropy ∘ M

electronic_entropy!(M::AbstractModel, result::AbstractVector, 𝐱::AbstractVector) =
    map!(electronic_entropy(M), result, 𝐱)
electronic_entropy!(M::AbstractModel, result::AbstractMatrix, X::AbstractMatrix) =
    copy!(result, electronic_entropy(M)(X))  # Note this is not element-wise!
