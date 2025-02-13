using DifferentiationInterface: gradient!

export Manual,
    Auto,
    NoDiff,
    CustomAutoEnzyme,
    autodiff_model,
    autodiff_model!,
    manualdiff_model,
    manualdiff_model!,
    deriv

abstract type DiffStrategy end
struct Manual <: DiffStrategy end
struct Auto{T} <: DiffStrategy
    backend::T
end
struct NoDiff <: DiffStrategy end  # Only for debugging

struct CustomAutoEnzyme end

function autodiff_model(f, model, x, backend)
    grad = similar(parent(model))
    return autodiff_model!(f, grad, model, x, backend)
end
function autodiff_model!(f, grad, model, x, backend)
    if length(grad) != length(model)
        throw(DimensionMismatch("the length of gradient and the model are not equal!"))
    end
    g(model′) = f(model′(x))
    return gradient!(g, grad, backend, model)
end

function manualdiff_model(f′, model, x)
    grad = similar(parent(model))
    return manualdiff_model!(f′, grad, model, x)
end
function manualdiff_model!(f′, grad::AbstractVecOrMat, model, x)
    𝐲 = zeros(eltype(x), numlayers(model) + 1)
    return manualdiff_model!(f′, 𝐲, grad, model, x)
end
function manualdiff_model!(f′, 𝐲, grad::AbstractVecOrMat, model::Model, x)
    if length(grad) != length(model)
        throw(DimensionMismatch("the length of gradient and the model are not equal!"))
    end
    if length(𝐲) != numlayers(model) + 1
        throw(DimensionMismatch("the length of 𝐲 and the model do not match!"))
    end
    layers = eachlayer(model)
    layerindices = eachindex(layers)
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
        𝟏 = one(y)  # Not `oneunit(y)`
        # zᵢ₊₁
        grad[linear_indices[1, i]] = α * z * y^2
        grad[linear_indices[2, i]] = α * z * y
        grad[linear_indices[3, i]] = α * z
        grad[linear_indices[4, i]] = α * y
        z = 𝐦[4] * 𝟏 + z * (2𝐦[1] * y + 𝐦[2] * 𝟏)  # zᵢ
    end
    return grad
end
manualdiff_model!(f′, 𝐲, grad::AbstractVecOrMat, model::FlatModel, x) =
    manualdiff_model!(f′, 𝐲, grad, Model(model), x)

_finalize_fermi_dirac_jac(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_jac(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

function compute_jac(f_or_f′, model, 𝐱, strategy::DiffStrategy)
    jac = similar(parent(model), length(𝐱), length(model))
    return compute_jac!(f_or_f′, jac, model, 𝐱, strategy)
end
# function compute_jac!(f′, jac, model, 𝐱, ::Manual)
#     if size(jac) != (length(𝐱), length(model))
#         throw(DimensionMismatch("the size of `jac` is not compatible with `𝐱` & `model`!"))
#     end
#     for (i, x) in enumerate(𝐱)
#         manualdiff_model!(f′, @view(jac[i, :]), model, x)
#     end
#     return jac
# end
function compute_jac!(f′, jac, model, 𝐱, ::Manual)
    if size(jac) != (length(𝐱), length(model))
        throw(DimensionMismatch("the size of `jac` is not compatible with `𝐱` & `model`!"))
    end
    𝐲 = similar(𝐱, numlayers(model) + 1)
    for (i, x) in enumerate(𝐱)
        manualdiff_model!(f′, 𝐲, @view(jac[i, :]), model, x)
    end
    return jac
end
function compute_jac!(f, jac, model, 𝐱, strategy::Auto)
    if size(jac) != (length(𝐱), length(model))
        throw(DimensionMismatch("the size of `jac` is not compatible with `𝐱` & `model`!"))
    end
    for (i, x) in enumerate(𝐱)
        autodiff_model!(
            f,
            @view(jac[i, :]),  # Must use `@view` or `jac` will not be updated
            model,
            x,
            strategy.backend,
        )
    end
    return jac
end

fermi_dirac_jac(model, 𝐱, ::Manual) =
    compute_jac(_finalize_fermi_dirac_jac, model, 𝐱, Manual())
fermi_dirac_jac(model, 𝐱, strategy::Auto) =
    compute_jac(_finalize_fermi_dirac, model, 𝐱, strategy)

fermi_dirac_jac!(jac, model, 𝐱, ::Manual) =
    compute_jac!(_finalize_fermi_dirac_jac, jac, model, 𝐱, Manual())
fermi_dirac_jac!(jac, model, 𝐱, strategy::Auto) =
    compute_jac!(_finalize_fermi_dirac, jac, model, 𝐱, strategy)

electronic_entropy_jac(model, 𝐱, ::Manual) =
    compute_jac(_finalize_electronic_entropy_jac, model, 𝐱, Manual())
electronic_entropy_jac(model, 𝐱, strategy::Auto) =
    compute_jac(_finalize_electronic_entropy, model, 𝐱, strategy)

electronic_entropy_jac!(jac, model, 𝐱, ::Manual) =
    compute_jac!(_finalize_electronic_entropy_jac, jac, model, 𝐱, Manual())
electronic_entropy_jac!(jac, model, 𝐱, strategy) =
    compute_jac!(_finalize_electronic_entropy, jac, model, 𝐱, strategy)

function deriv(model::AbstractModel, x)
    y = x
    ∂y➗∂x = one(x)  # ∂y/∂x
    𝟏, 𝟏′, 𝟏″ = oneunit(eltype(model)), oneunit(x), one(y)
    accumulator = zero(𝟏 * x)  # For ∂Y/∂x
    for 𝐦 in eachlayer(model)
        accumulator += 𝐦[4] * ∂y➗∂x
        ∂y➗∂x = (2𝐦[1] * y + 𝐦[2] * 𝟏″) * ∂y➗∂x
        y = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * 𝟏′
    end
    accumulator += 𝟏 * ∂y➗∂x
    return accumulator
end
