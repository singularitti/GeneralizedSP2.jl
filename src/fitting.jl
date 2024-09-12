using LsqFit: curve_fit, coef

export fit_model, model!, model, fermi_dirac_model, entropy_model, fit_residuals

sp2model(y, 𝝷) = @. 𝝷[1] * y .^ 2 + 𝝷[2] * y + 𝝷[3]

function fit_residuals(𝐱, 𝐲, nlayers=4; max_iter=100)
    θ = ones(3)
    𝝷 = [θ]
    total_output = zeros(size(𝐱))
    residual = 𝐲 - total_output
    predicted = 𝐱
    for _ in 1:nlayers
        fitted_fermi = curve_fit(
            sp2model,
            predicted,  # xdata
            residual,  # ydata
            θ;  # p0
            maxIter=max_iter,
        )
        θ = coef(fitted_fermi)
        push!(𝝷, θ)
        # Update `predicted` with the new model output based on the fitted parameters
        predicted = sp2model(predicted, θ)
        # Important: We add the *newly predicted* values to the total output
        # *after* updating `predicted` with the current model predictions.
        # If we add `predicted` before updating it, we'd be adding the old
        # predictions (or even just the input `𝐱` in the first iteration), which
        # would corrupt the total output with incorrect values.
        total_output += predicted
        residual = 𝐲 - total_output  # This progressively reduces the residual as the predictions improve.
    end
    return 𝝷, total_output
end

function model!(f, result, 𝐱::AbstractVector, 𝝷::AbstractMatrix)
    if size(𝝷, 1) != LAYER_WIDTH
        throw(ArgumentError("input coefficients matrix must have $LAYER_WIDTH rows!"))
    end
    map!(result, 𝐱) do x
        y = x  # `x` and `y` are 2 numbers
        Y = zero(eltype(result))  # Accumulator of the summation
        for θᵢ in eachcol(𝝷)
            Y += θᵢ[4] * y
            y = θᵢ[1] * y^2 + θᵢ[2] * y + θᵢ[3]
        end
        Y += y
        f(Y)
    end
    return result
end
model!(f, result, 𝐱::AbstractVector, 𝛉::AbstractVector) =
    model!(f, result, 𝐱, reshape(𝛉, LAYER_WIDTH, :))

fermi_dirac_model!(result, 𝐱, 𝛉) = model!(transform_fermi_dirac, result, 𝐱, 𝛉)

entropy_model!(result, 𝐱, 𝛉) = model!(transform_entropy, result, 𝐱, 𝛉)

function model(f, 𝐱, 𝛉)
    T = typeof(f(first(𝛉) * first(𝐱)))
    result = similar(𝐱, T)
    model!(f, result, 𝐱, 𝛉)
    return result
end

transform_fermi_dirac(Y) = oneunit(Y) - Y  # Applies to 1 number at a time

transform_entropy(Y) = 4log(2) * (Y - Y^2)  # Applies to 1 number at a time

fermi_dirac_model(x, θ) = model(transform_fermi_dirac, x, θ)

entropy_model(x, θ) = model(transform_entropy, x, θ)

function jacobian!(J::AbstractMatrix, x, θ, df_dY)
    npoints = length(x)
    θ = reshape(θ, LAYER_WIDTH, :)
    nlayers = size(θ, 2)

    J = reshape(J, npoints, LAYER_WIDTH, nlayers)
    y = zeros(eltype(x), nlayers + 1)

    for j in 1:npoints

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

transform_fermi_dirac_derivative(Y) = -one(Y)  # Applies to 1 number at a time

transform_entropy_derivative(Y) = 4log(2) * (oneunit(Y) - 2Y)  # Applies to 1 number at a time

fermi_dirac_jacobian!(J, x, θ) = jacobian!(J, x, θ, transform_fermi_dirac_derivative)

entropy_jacobian!(J, x, θ) = jacobian!(J, x, θ, transform_entropy_derivative)

function fit_model(𝐱, μ, β; max_iter=100, nlayers=round(Int64, 4.75log(β) - 6.6))
    # Initialize model with SP2
    θ = init_params(μ, nlayers)

    fitted_fermi = curve_fit(
        fermi_dirac_model!,
        fermi_dirac_jacobian!,
        𝐱,  # xdata
        fermi_dirac.(𝐱, μ, β),  # ydata
        θ;  # p0
        maxIter=max_iter,
        inplace=true,
    )
    fitted_entropy = curve_fit(
        entropy_model!,
        entropy_jacobian!,
        𝐱,
        electronic_entropy.(𝐱, μ, β),
        θ;
        maxIter=max_iter,
        inplace=true,
    )

    return coef(fitted_fermi), coef(fitted_entropy)
end
