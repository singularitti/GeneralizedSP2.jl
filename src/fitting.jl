using LsqFit: curve_fit, coef

export generate_model, model!, model

### Generalized model

# Postprocessing for final model output, and derivative
fermi_transf_1(Y) = 1 - Y
fermi_transf_2(Y) = -1.0
entropy_transf_1(Y) = 4log(2) * (Y - Y^2)
entropy_transf_2(Y) = 4log(2) * (1 - 2Y)

model_fermi(x, θ) = model(x, θ, fermi_transf_1)
model_entropy(x, θ) = model(x, θ, entropy_transf_1)

function model!(f, result, 𝐱, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    map!(result, 𝐱) do x
        y = x
        Y = zero(eltype(result))
        for θᵢ in eachcol(𝝷)
            Y += θᵢ[4] * y
            y = θᵢ[1] * y^2 + θᵢ[2] * y + θᵢ[3]
        end
        Y += y
        f(Y)
    end
    return result
end
model!(f, result, 𝐱, 𝛉::AbstractVector) = model!(f, result, 𝐱, reshape(𝛉, LAYER_WIDTH, :))

function model(f, 𝐱, 𝛉)
    T = typeof(f(first(𝛉) * first(𝐱)))
    result = similar(𝐱, T)
    model!(f, result, 𝐱, 𝛉)
    return result
end

model_inplace_fermi(res, x, θ) = model!(res, x, θ, fermi_transf_1)

model_inplace_entropy(res, x, θ) = model!(res, x, θ, entropy_transf_1)

function jacobian_inplace(J::Array{Float64,2}, x, θ, df_dY)
    npts = length(x)
    θ = reshape(θ, LAYER_WIDTH, :)
    nlayers = size(θ, 2)

    J = reshape(J, npts, LAYER_WIDTH, nlayers)
    y = zeros(eltype(x), nlayers + 1)

    for j in 1:npts

        # forward calculation
        y[1] = x[j]
        Y = zero(eltype(J))
        for i in 1:nlayers
            Y += θ[4, i] * y[i]
            y[i + 1] = θ[1, i] * y[i]^2 + θ[2, i] * y[i] + θ[3, i]
        end
        Y += y[nlayers + 1]
        α = df_dY(Y)

        # backward calculation
        z = 1 # z_{n+1}
        for i in nlayers:-1:1
            # z = z_{i+1}
            J[j, 1, i] = α * z * y[i]^2
            J[j, 2, i] = α * z * y[i]
            J[j, 3, i] = α * z
            J[j, 4, i] = α * y[i]

            z = θ[4, i] + z * (2θ[1, i] * y[i] + θ[2, i])
        end
    end
end

jacobian_inplace_fermi(J, x, θ) = jacobian_inplace(J, x, θ, fermi_transf_2)

jacobian_inplace_entropy(J, x, θ) = jacobian_inplace(J, x, θ, entropy_transf_2)

function generate_model(;
    β, μ, max_iter, npts_scale=1.0, nlayers=round(Int64, 4.75log(β) - 6.6)
)

    # Sample points more densely near x=μ
    npts = npts_scale * 80log(β)
    w = sqrt(β)
    sample_density(x) = (npts / 2) + (npts / 2) * (w / 2) * sech(w * (x - μ))^2
    x = sample_by_density(μ, 0, 1, sample_density)
    weight = sample_weights(x)

    # Initialize model with SP2
    θ = init_params(μ, nlayers)

    fit_fermi = curve_fit(
        model_inplace_fermi,
        jacobian_inplace_fermi,
        x,  # xdata
        fermi_dirac.(x, β, μ),  # ydata
        θ;  # p0
        maxIter=max_iter,
        inplace=true,
    )
    fit_entropy = curve_fit(
        model_inplace_entropy,
        jacobian_inplace_entropy,
        x,
        entropyof.(x, β, μ),
        θ;
        maxIter=max_iter,
        inplace=true,
    )

    return (; θ, θ_fermi=coef(fit_fermi), θ_entropy=coef(fit_entropy), x)
end
