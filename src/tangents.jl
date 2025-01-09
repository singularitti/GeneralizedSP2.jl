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

function manualdiff_model!(f′, 𝐌̄, 𝐱::AbstractVector, model::Model)
    npoints = length(𝐱)
    𝐌̄ = reshape(𝐌̄, size(𝐱)..., size(model)...)
    for j in 1:npoints
        # Call manualdiff_model_single! for each x in 𝐱
        manualdiff_model!(f′, 𝐌̄[j, :, :], 𝐱[j], model)
    end
    return 𝐌̄
end
function manualdiff_model!(f′, 𝐌̄, x, model::Model)
    nlayers = numlayers(model)
    𝐲 = zeros(eltype(x), nlayers + 1)
    𝐌̄ = reshape(𝐌̄, size(model)...)
    # Forward calculation
    𝐲[1] = x
    Y = zero(eltype(𝐲))
    for i in 1:nlayers
        Y += model[4, i] * 𝐲[i]
        𝐲[i + 1] = model[1, i] * 𝐲[i]^2 + model[2, i] * 𝐲[i] + model[3, i] * oneunit(𝐲[i])
    end
    Y += 𝐲[nlayers + 1]
    α = f′(Y)
    # Backward calculation
    z = one(eltype(model)) # zₗₐₛₜ
    for i in nlayers:-1:1
        # zᵢ₊₁
        𝐌̄[1, i] = α * z * 𝐲[i]^2
        𝐌̄[2, i] = α * z * 𝐲[i]
        𝐌̄[3, i] = α * z
        𝐌̄[4, i] = α * 𝐲[i]
        z =
            model[4, i] * oneunit(𝐲[i]) +
            z * (2model[1, i] * 𝐲[i] + model[2, i] * oneunit(𝐲[i]))  # zᵢ
    end
    return 𝐌̄
end
manualdiff_model!(f′, 𝐌̄, 𝐱, M) = manualdiff_model!(f′, 𝐌̄, 𝐱, Model(FlattendModel(M)))

_finalize_fermi_dirac_grad(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_grad(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_grad!(𝐌̄, 𝐱, M) = manualdiff_model!(_finalize_fermi_dirac_grad, 𝐌̄, 𝐱, M)

electronic_entropy_grad!(𝐌̄, 𝐱, M) =
    manualdiff_model!(_finalize_electronic_entropy_grad, 𝐌̄, 𝐱, M)
