using DifferentiationInterface: Constant, derivative

export autodiff_model

function apply!(t, model, i, x)
    model[i] = t
    return model(x)
end

function autodiff_model(model::AbstractModel, x, backend)
    return map(eachindex(model)) do i
        derivative(apply!, backend, model[i], Constant(model), Constant(i), Constant(x))
    end
end
function autodiff_model(model::AbstractModel, 𝐱::AbstractVector, backend)
    return transpose(hcat(map(𝐱) do x
        autodiff_model(model, x, backend)
    end...))
end

function manualdiff_model!(f′, derivatives::AbstractVecOrMat, model, x)
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

function fermi_dirac_grad!(derivatives, 𝐱, M)
    for (i, x) in enumerate(𝐱)
        manualdiff_model!(_finalize_fermi_dirac_grad, derivatives[i, :], M, x)
    end
    return derivatives
end

electronic_entropy_grad!(𝐌̄, 𝐱, M) =
    manualdiff_model!(_finalize_electronic_entropy_grad, 𝐌̄, M, 𝐱)
