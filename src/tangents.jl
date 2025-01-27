using DifferentiationInterface: Constant, derivative, prepare_derivative

export Manual, Auto, autodiff_model, autodiff_model!, manualdiff_model, manualdiff_model!

abstract type Strategy end
struct Manual <: Strategy end
struct Auto{T} <: Strategy
    backend::T
end

function _modify_apply!(parameter, model::AbstractModel, index, x)
    model[index] = parameter
    return model(x)
end

function autodiff_model(model::AbstractModel, x, backend)
    derivatives = similar(model)
    return autodiff_model!(derivatives, model, x, backend)
end
function autodiff_model!(derivatives, model::AbstractModel, x, backend)
    if length(derivatives) != length(model)
        throw(ArgumentError("The length of derivatives and the model are not equal!"))
    end
    return map!(derivatives, eachindex(model)) do i
        contexts = Constant(model), Constant(i), Constant(x)
        prep = prepare_derivative(_modify_apply!, backend, model[i], contexts...)
        derivative(_modify_apply!, prep, backend, model[i], contexts...)
    end
end

function manualdiff_model(f′, model, x)
    derivatives = similar(model)
    return manualdiff_model!(f′, derivatives, model, x)
end
function manualdiff_model!(f′, derivatives::AbstractVecOrMat, model, x)
    if length(derivatives) != length(model)
        throw(ArgumentError("The length of derivatives and the model are not equal!"))
    end
    # FIXME: Only works for `derivatives` which supports linear indices
    model = Model(model)
    layers = eachlayer(model)
    layerindices = eachindex(layers)
    𝐲 = zeros(eltype(x), numlayers(model) + 1)
    # Forward calculation
    𝐲[begin] = x
    accumulator = zero(eltype(𝐲))
    for (i, 𝐦) in zip(layerindices, layers)
        y = 𝐲[i]
        accumulator += 𝐦[4] * y
        𝐲[i + 1] = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
    end
    accumulator += 𝐲[end]
    α = f′(accumulator)
    # Backward calculation
    z = one(eltype(model)) # zₗₐₛₜ
    for (i, 𝐦) in Iterators.reverse(zip(layerindices, layers))
        y = 𝐲[i]
        𝟏 = oneunit(y)
        # zᵢ₊₁
        derivatives[linear_index(1, i)] = α * z * y^2
        derivatives[linear_index(2, i)] = α * z * y
        derivatives[linear_index(3, i)] = α * z
        derivatives[linear_index(4, i)] = α * y
        z = 𝐦[4] * 𝟏 + z * (2𝐦[1] * y + 𝐦[2] * 𝟏)  # zᵢ
    end
    return derivatives
end

linear_index(j, i) = j + 4 * (i - 1)  # The linear index of the j-th element in the i-th layer

_finalize_fermi_dirac_grad(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_grad(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

function fermi_dirac_grad!(derivatives, 𝐱, M, ::Manual)
    if size(derivatives) != (length(𝐱), length(M))
        throw(DimensionMismatch("the size of derivatives is not compatible with 𝐱 and M!"))
    end
    for (i, x) in enumerate(𝐱)
        manualdiff_model!(_finalize_fermi_dirac_grad, @view(derivatives[i, :]), M, x)
    end
    return derivatives
end
function fermi_dirac_grad!(derivatives, 𝐱, M, strategy::Auto)
    if size(derivatives) != (length(𝐱), length(M))
        throw(DimensionMismatch("the size of derivatives is not compatible with 𝐱 and M!"))
    end
    for (i, x) in enumerate(𝐱)
        autodiff_model!(@view(derivatives[i, :]), M, x, strategy.backend)
    end
    return derivatives
end

electronic_entropy_grad!(𝐌̄, 𝐱, M) =
    manualdiff_model!(_finalize_electronic_entropy_grad, 𝐌̄, M, 𝐱)
