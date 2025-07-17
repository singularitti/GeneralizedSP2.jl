using LinearAlgebra: checksquare, axpy!, axpby!, mul!
using LinearAlgebra.BLAS: scal!
# using Enzyme: Reverse, Const, Duplicated, autodiff

export basis, apply, apply!, fermi_dirac!, electronic_entropy, electronic_entropy!

const FOUR_LOG_TWO = 4log(2)

function basis(model::AbstractModel)
    function _collect(x)
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
    return _collect
end

function apply(model, x)
    𝟏ₘ, 𝟏ₓ = oneunit(eltype(model)), oneunit(x)
    accumulator = zero(𝟏ₘ * 𝟏ₓ)  # Accumulator of the summation
    y = x  # `x` and `y` are 2 numbers
    for 𝐦 in eachlayer(model)
        accumulator += 𝐦[4] * y
        y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * 𝟏ₓ
    end
    accumulator += 𝟏ₘ * y
    return accumulator
end
# function apply(model, x)
#     T = promote_type(typeof(x^2), typeof(x))
#     𝟏ₘ, 𝟏ₓ = oneunit(eltype(model)), oneunit(T)
#     accumulator = zero(𝟏ₘ * 𝟏ₓ)  # Accumulator of the summation
#     y = x  # `x` and `y` are 2 numbers
#     for 𝐦 in eachlayer(model)
#         accumulator += 𝐦[4] * y
#         y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * 𝟏ₓ
#     end
#     accumulator += 𝟏ₘ * y
#     return accumulator
# end

function apply!(𝐲::AbstractVector, model, x)
    if length(𝐲) != numlayers(model) + 1
        throw(DimensionMismatch("the length of 𝐲 and the model do not match!"))
    end
    layers = eachlayer(model)
    layerindices = eachindex(layers)
    𝐲[begin] = x
    accumulator = zero(eltype(𝐲))
    for (i, 𝐦) in zip(layerindices, layers)
        y = 𝐲[i]
        accumulator += 𝐦[4] * y
        𝐲[i + 1] = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
    end
    accumulator += 𝐲[end]
    return accumulator
end
# function apply!(𝐲::AbstractVector, model, x)
#     @assert isempty(𝐲)
#     𝟏ₘ, 𝟏ₓ = oneunit(eltype(model)), oneunit(x)
#     accumulator = zero(𝟏ₘ * 𝟏ₓ)  # Accumulator of the summation
#     y = x  # `x` and `y` are 2 numbers
#     push!(𝐲, y)
#     for 𝐦 in eachlayer(model)
#         accumulator += 𝐦[4] * y
#         y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * 𝟏ₓ
#         push!(𝐲, y)
#     end
#     accumulator += 𝟏ₘ * y
#     return accumulator
# end
function apply!(result::AbstractMatrix, model, X::AbstractMatrix)
    checksquare(X)  # See https://discourse.julialang.org/t/120556/2
    if size(result) != size(X)
        throw(DimensionMismatch("the size of `result` is not compatible with `X`!"))
    end
    if !iszero(X)  # Very fast
        map!(zero, result, result)
    end
    Y = deepcopy(X)  # Modifying `Y` does not change `X` now
    Y² = similar(Y)
    𝟙 = oneunit(Y)  # Identity matrix
    for 𝐦 in eachlayer(model)  # All operations are in-place, significantly reducing allocations.
        m₁, m₂, m₃, m₄ = 𝐦
        axpy!(m₄, Y, result)  # result .+= m₄ * Y
        mul!(Y², Y, Y)  # Y² .= Y^2, not elementwise square!
        axpby!(m₁, Y², m₂, Y)  # Y .+= m₁ * Y² + m₂ * Y
        axpy!(m₃, 𝟙, Y)  # Y .+= m₃ * 𝟙
    end
    axpy!(oneunit(eltype(model)), Y, result)  # result .+= oneunit(eltype(model)) * Y
    return result
end

(model::AbstractModel)(x) = apply(model, x)
(model::AbstractModel)(result::AbstractMatrix, X::AbstractMatrix) = apply!(result, model, X)
function (model::AbstractModel)(X::AbstractMatrix)
    result = similar(X, typeof(oneunit(eltype(model)) * oneunit(eltype(X))))  # Prepare for in-place result
    return apply!(result, model, X)
end

_finalize_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number/matrix at a time
_finalize_fermi_dirac!(Y::AbstractMatrix) = axpby!(1, oneunit(Y), -1, Y)  # This is the fastest, except for `axpy!(-1, result, oneunit(Y))`, which we cannot use here.

fermi_dirac(model::AbstractModel) = _finalize_fermi_dirac ∘ model

fermi_dirac!(rho, model, H) = _finalize_fermi_dirac!(apply!(rho, model, H))

_finalize_electronic_entropy(Y) = FOUR_LOG_TWO * (Y - Y^2)  # Applies to 1 number/matrix at a time
function _finalize_electronic_entropy!(Y::AbstractMatrix)
    Y² = similar(Y)
    mul!(Y², Y, Y)  # Y² .= Y^2
    axpy!(-1, Y², Y)  # Y .= Y - Y²
    scal!(FOUR_LOG_TWO, Y)  # Y .= 4log(2) * Y
    return Y
end

electronic_entropy(model::AbstractModel) = _finalize_electronic_entropy ∘ model

electronic_entropy!(entropy, model, distribution) =
    _finalize_electronic_entropy!(apply!(entropy, model, distribution))
