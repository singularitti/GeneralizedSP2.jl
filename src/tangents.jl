using DifferentiationInterface: prepare_gradient, gradient

export autodiff_model, manualdiff_model

function autodiff_model(model::AbstractModel, x, backend)
    prep = prepare_gradient(model, backend, x)
    gradient(model, prep, backend, x)
end

function manualdiff_model(f′, 𝐱, M)
    M = Model(FlattendModel(M))
    𝐌̄ = Array{typeof(oneunit(eltype(M)) / oneunit(eltype(𝐱)))}(
        undef, size(𝐱)..., size(M)...
    )
    return manualdiff_model!(f′, 𝐌̄, 𝐱, M)
end

function manualdiff_model!(f′, 𝐌̄, 𝐱, model::Model)
    npoints = length(𝐱)
    nlayers = numlayers(model)
    𝐌̄ = reshape(𝐌̄, size(𝐱)..., size(model)...)
    𝐲 = zeros(eltype(𝐱), nlayers + 1)
    for j in 1:npoints
        # Forward calculation
        𝐲[1] = 𝐱[j]
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
            𝐌̄[j, 1, i] = α * z * 𝐲[i]^2
            𝐌̄[j, 2, i] = α * z * 𝐲[i]
            𝐌̄[j, 3, i] = α * z
            𝐌̄[j, 4, i] = α * 𝐲[i]
            z = model[4, i] * oneunit(𝐲[i]) + z * (2model[1, i] * 𝐲[i] + model[2, i] * oneunit(𝐲[i]))  # zᵢ
        end
    end
    return 𝐌̄
end
manualdiff_model!(f′, 𝐌̄, 𝐱, M) = manualdiff_model!(f′, 𝐌̄, 𝐱, Model(FlattendModel(M)))

_finalize_fermi_dirac_grad(Y) = -one(Y)  # Applies to 1 number at a time

_finalize_electronic_entropy_grad(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_grad!(𝐌̄, 𝐱, M) = manualdiff_model!(_finalize_fermi_dirac_grad, 𝐌̄, 𝐱, M)

electronic_entropy_grad!(𝐌̄, 𝐱, M) =
    manualdiff_model!(_finalize_electronic_entropy_grad, 𝐌̄, 𝐱, M)
