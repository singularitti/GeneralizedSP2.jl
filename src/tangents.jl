using DifferentiationInterface: prepare_jacobian, jacobian

export autodiff_model, manualdiff_model

function autodiff_model(model::Model, x, backend)
    prep = prepare_jacobian(Base.Fix1(map, model), backend, x)
    return jacobian(Base.Fix1(map, model), prep, backend, x)
end
function autodiff_model(model, x, backend)
    model = Model(FlattendModel(model))
    prep = prepare_jacobian(Base.Fix1(map, model), backend, x)
    return jacobian(Base.Fix1(map, model), prep, backend, x)
end

function manualdiff_model(f′, 𝐱, M)
    M = Model(FlattendModel(M))
    𝐌̄ = Array{typeof(oneunit(eltype(M)) / oneunit(eltype(𝐱)))}(
        undef, size(𝐱)..., size(M)...
    )
    return manualdiff_model!(f′, 𝐌̄, 𝐱, M)
end

function manualdiff_model!(
    f′, derivatives::AbstractArray{T,3}, 𝐱::AbstractVector, model::AbstractModel
) where {T}
    if size(derivatives) != (size(𝐱)..., size(model)...)
        throw(DimensionMismatch("the derivatives do not have the correct size!"))
    end
    for (i, x) in enumerate(𝐱)
        manualdiff_model!(f′, derivatives[i, :, :], x, model)  # Single-point calculation
    end
    return derivatives
end
function manualdiff_model!(f′, derivatives::AbstractMatrix, x, model::AbstractModel)
    if size(model) != size(derivatives)
        throw(DimensionMismatch("the model and its derivatives must have the same size!"))
    end
    nlayers = numlayers(model)
    𝐲 = zeros(eltype(x), nlayers + 1)
    # Forward calculation
    𝐲[1] = x
    accumulator = zero(eltype(𝐲))
    for (i, 𝐦) in enumerate(eachlayer(model))
        y = 𝐲[i]
        accumulator += 𝐦[4] * y
        𝐲[i + 1] = 𝐦[1] * y^2 + 𝐦[2] * y + 𝐦[3] * oneunit(y)
    end
    accumulator += 𝐲[nlayers + 1]
    α = f′(accumulator)
    # Backward calculation
    z = one(eltype(model)) # zₗₐₛₜ
    for (i, 𝐦) in Iterators.reverse(enumerate(eachlayer(model)))
        y = 𝐲[i]
        𝟏 = oneunit(y)
        # zᵢ₊₁
        derivatives[1, i] = α * z * y^2
        derivatives[2, i] = α * z * y
        derivatives[3, i] = α * z
        derivatives[4, i] = α * y
        z = 𝐦[4] * 𝟏 + z * (2𝐦[1] * y + 𝐦[2] * 𝟏)  # zᵢ
    end
    return derivatives
end

_finalize_fermi_dirac_grad(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_grad(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_grad!(𝐌̄, 𝐱, M) = manualdiff_model!(_finalize_fermi_dirac_grad, 𝐌̄, 𝐱, M)

electronic_entropy_grad!(𝐌̄, 𝐱, M) =
    manualdiff_model!(_finalize_electronic_entropy_grad, 𝐌̄, 𝐱, M)
