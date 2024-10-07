using LsqFit: curve_fit, coef

export fit_model

function fit_model(𝐱, μ, β, nlayers=round(Int64, 4.75log(β) - 6.6); max_iter=100)
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
