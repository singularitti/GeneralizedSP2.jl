using DifferentiationInterface: Constant, gradient!

export Manual, Auto, autodiff_model, autodiff_model!, manualdiff_model, manualdiff_model!

abstract type DiffStrategy end
struct Manual <: DiffStrategy end
struct Auto{T} <: DiffStrategy
    backend::T
end

_apply(x) = model -> model(x)

function autodiff_model(f, model, x, backend)
    derivatives = similar(parent(model))
    return autodiff_model!(f, derivatives, model, x, backend)
end
function autodiff_model!(f, derivatives, model, x, backend)
    if length(derivatives) != length(model)
        throw(DimensionMismatch("the length of derivatives and the model are not equal!"))
    end
    model = Model(model)
    g = f ∘ _apply(x)
    return gradient!(g, derivatives, backend, model)
end

function manualdiff_model(f′, model, x)
    derivatives = similar(parent(model))
    return manualdiff_model!(f′, derivatives, model, x)
end
function manualdiff_model!(f′, derivatives::AbstractVecOrMat, model, x)
    if length(derivatives) != length(model)
        throw(DimensionMismatch("the length of derivatives and the model are not equal!"))
    end
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
    linear_indices = LinearIndices(model)
    for (i, 𝐦) in Iterators.reverse(zip(layerindices, layers))
        y = 𝐲[i]
        𝟏 = oneunit(y)
        # zᵢ₊₁
        derivatives[linear_indices[1, i]] = α * z * y^2
        derivatives[linear_indices[2, i]] = α * z * y
        derivatives[linear_indices[3, i]] = α * z
        derivatives[linear_indices[4, i]] = α * y
        z = 𝐦[4] * 𝟏 + z * (2𝐦[1] * y + 𝐦[2] * 𝟏)  # zᵢ
    end
    return derivatives
end

_finalize_fermi_dirac_jac(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_jac(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

function compute_jac(f_or_f′, model, 𝐱, strategy::DiffStrategy)
    derivatives = similar(parent(model), length(𝐱), length(model))
    return compute_jac!(f_or_f′, derivatives, model, 𝐱, strategy)
end
function compute_jac!(f′, derivatives, model, 𝐱, ::Manual)
    if size(derivatives) != (length(𝐱), length(model))
        throw(DimensionMismatch("the size of `derivatives` is not compatible with `𝐱` & `model`!"))
    end
    for (i, x) in enumerate(𝐱)
        manualdiff_model!(f′, @view(derivatives[i, :]), model, x)
    end
    return derivatives
end
function compute_jac!(f, derivatives, model, 𝐱, strategy::Auto)
    if size(derivatives) != (length(𝐱), length(model))
        throw(DimensionMismatch("the size of `derivatives` is not compatible with `𝐱` & `model`!"))
    end
    for (i, x) in enumerate(𝐱)
        autodiff_model!(
            f,
            @view(derivatives[i, :]),  # Must use `@view` or `derivatives` will not be updated
            model,
            x,
            strategy.backend,
        )
    end
    return derivatives
end

fermi_dirac_jac(model, 𝐱, ::Manual) =
    compute_jac(_finalize_fermi_dirac_jac, model, 𝐱, Manual())
fermi_dirac_jac(model, 𝐱, strategy::Auto) =
    compute_jac(_finalize_fermi_dirac, model, 𝐱, strategy)
fermi_dirac_jac!(derivatives, model, 𝐱, ::Manual) =
    compute_jac!(_finalize_fermi_dirac_jac, derivatives, model, 𝐱, Manual())
fermi_dirac_jac!(derivatives, model, 𝐱, strategy::Auto) =
    compute_jac!(_finalize_fermi_dirac, derivatives, model, 𝐱, strategy)

electronic_entropy_jac(model, 𝐱, ::Manual) =
    compute_jac(_finalize_electronic_entropy_jac, model, 𝐱, Manual())
electronic_entropy_jac(model, 𝐱, strategy::Auto) =
    compute_jac(_finalize_electronic_entropy, model, 𝐱, strategy)
electronic_entropy_jac!(derivatives, model, 𝐱, ::Manual) =
    compute_jac!(_finalize_electronic_entropy_jac, derivatives, model, 𝐱, Manual())
electronic_entropy_jac!(derivatives, model, 𝐱, strategy) =
    compute_jac!(_finalize_electronic_entropy, derivatives, model, 𝐱, strategy)
