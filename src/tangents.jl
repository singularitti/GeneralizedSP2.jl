export manualdiff_model

function manualdiff_model(f′, 𝐱, M)
    M = Model(FlattendModel(M))
    𝐌̄ = Array{Float64}(undef, size(𝐱)..., size(M)...)
    return manualdiff_model!(f′, 𝐌̄, 𝐱, M)
end

function manualdiff_model!(f′, 𝐌̄, 𝐱, M::Model)
    npoints = length(𝐱)
    nlayers = numlayers(M)
    𝐌̄ = reshape(𝐌̄, size(𝐱)..., size(M)...)
    𝐲 = zeros(eltype(𝐱), nlayers + 1)
    for j in 1:npoints
        # Forward calculation
        𝐲[1] = 𝐱[j]
        Y = zero(eltype(𝐲))
        for i in 1:nlayers
            Y += M[4, i] * 𝐲[i]
            𝐲[i + 1] = M[1, i] * 𝐲[i]^2 + M[2, i] * 𝐲[i] + M[3, i] * oneunit(𝐲[i])
        end
        Y += 𝐲[nlayers + 1]
        α = f′(Y)
        # Backward calculation
        z = one(eltype(M)) # zₗₐₛₜ
        for i in nlayers:-1:1
            # zᵢ₊₁
            𝐌̄[j, 1, i] = α * z * 𝐲[i]^2
            𝐌̄[j, 2, i] = α * z * 𝐲[i]
            𝐌̄[j, 3, i] = α * z
            𝐌̄[j, 4, i] = α * 𝐲[i]
            z = M[4, i] * oneunit(𝐲[i]) + z * (2M[1, i] * 𝐲[i] + M[2, i] * oneunit(𝐲[i]))  # zᵢ
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
