using Enzyme: Reverse, Active, Duplicated, Const, autodiff

export manualdiff_model

function autodiff_model()
    θ = randn(nlayers, 4)
    dθ = zero(θ)
    A = zeros(nlayers, npts)
    dA = zero(A)
    F = zero(X)
    dF = zero(F)
    L = zeros(1)
    dL = zero(L)

    function f(
        θ::Matrix{Float64},
        𝐱::Vector{Float64},
        𝐲̂::Vector{Float64},
        A::Matrix{Float64},
        F::Vector{Float64},
        L::Vector{Float64},
    )
        F .= 0

        for layer in 1:size(θ, 1)
            for i in eachindex(𝐱)
                if layer == 1
                    A[layer, i] = θ[layer, 1] * 𝐱[i]^2 + θ[layer, 2] * 𝐱[i] + θ[layer, 3]
                else
                    A[layer, i] =
                        θ[layer, 1] * A[layer - 1, i]^2 +
                        θ[layer, 2] * A[layer - 1, i] +
                        θ[layer, 3]
                end
                F[i] += θ[layer, 4] * A[layer, i]
            end
        end

        L[1] = 0
        for i in eachindex(𝐱)
            L[1] += (F[i] - 𝐲̂[i])^2
        end
        return L[1]
    end

    return autodiff(
        Reverse,
        f,
        Active,
        Duplicated(θ, dθ),
        Const(X),
        Const(Y),
        Duplicated(A, dA),
        Duplicated(F, dF),
        Duplicated(L, dL),
    )
end

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
