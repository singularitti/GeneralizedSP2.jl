using LeastSquaresOptim: LevenbergMarquardt, optimize

export fit_residuals0, fit_residuals, sp2model

# Define the quadratic model
sp2model(y, 𝛉) = 𝛉[1] * y .^ 2 + 𝛉[2] * y + 𝛉[3] * oneunit.(y)

# Custom loss function with regularization for residual fitting
function residuals_with_regularization(𝛉, x, y; λ₁=2, λ₂=2)
    # Residuals: differences between model predictions and actual values
    residuals = sp2model(x, 𝛉) - y
    # Regularization: penalize lower-order terms (𝛉[1] for x^2, 𝛉[2] for x)
    regularization = λ₁ * 𝛉[1]^2 + λ₂ * 𝛉[2]^2
    # Return the residuals, with regularization as an additional penalty term
    return vcat(residuals, regularization)
end

# Main function for fitting residuals with regularization
function fit_residuals(𝐱, 𝐲̂, nlayers=4; λ₁=2, λ₂=2)
    𝛉 = ones(3)  # Initial guess for parameters
    𝝷 = []
    𝐲 = zeros(size(𝐱))  # Start with no prediction
    𝐫 = 𝐲̂ - 𝐲
    𝚫𝐲 = collect(𝐱)  # Ensure that `prediction` is an array
    residuals = [𝐫]
    predictions = [𝚫𝐲]
    for _ in 1:nlayers
        # Define the objective function for the current layer, using residuals with regularization
        obj_func(𝛉) = residuals_with_regularization(𝛉, 𝚫𝐲, 𝐫; λ₁, λ₂)
        # Use LeastSquaresOptim for fitting
        result = optimize(obj_func, 𝛉, LevenbergMarquardt())  # or Dogleg()
        # Extract the fitted parameters
        𝛉 = result.minimizer
        # Store the fitted parameters
        push!(𝝷, 𝛉)
        # Update the predictions using the new fitted model
        𝚫𝐲 = sp2model(𝚫𝐲, 𝛉)
        push!(predictions, 𝚫𝐲)
        # Accumulate the predictions to get the full model prediction
        𝐲 += 𝚫𝐲
        # Update residual (reduce it layer by layer)
        𝐫 = 𝐲̂ - 𝐲
        push!(residuals, 𝐫)
    end
    return 𝝷, 𝐲, predictions, residuals
end

function fit_residuals0(𝐱, 𝐲̂, nlayers=4; max_iter=100)
    𝛉 = ones(3)
    𝝷 = []
    𝐲 = zeros(size(𝐱))
    𝐫 = 𝐲̂ - 𝐲
    𝚫𝐲 = 𝐱
    residuals = [𝐫]
    predictions = [𝚫𝐲]
    for _ in 1:nlayers
        result = curve_fit(
            sp2model,
            𝚫𝐲,  # xdata
            𝐫,  # ydata
            𝛉;  # p0
            maxIter=max_iter,
        )
        𝛉 = coef(result)
        push!(𝝷, 𝛉)
        # Update `predicted` with the new model output based on the fitted parameters
        𝚫𝐲 = sp2model(𝚫𝐲, 𝛉)
        push!(predictions, 𝚫𝐲)
        # Important: We add the *newly predicted* values to the total output
        # *after* updating `predicted` with the current model predictions.
        # If we add `predicted` before updating it, we'd be adding the old
        # predictions (or even just the input `𝐱` in the first iteration), which
        # would corrupt the total output with incorrect values.
        𝐲 += 𝚫𝐲
        𝐫 = 𝐲̂ - 𝐲  # This progressively reduces the residual as the predictions improve.
        push!(residuals, 𝐫)
    end
    return 𝝷, 𝐲, predictions, residuals
end
