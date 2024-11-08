using LsqFit: curve_fit, coef, stderror, vcov

export fit_fermi_dirac, fit_entropy

function fit_fermi_dirac(𝐱, μ, β, nlayers=20; max_iter=1000, rtol=NaN)
    # Initialize model with SP2
    𝛉 = init_params(μ, nlayers)
    fitted = curve_fit(
        fermi_dirac_model!,
        fermi_dirac_derivatives!,
        𝐱,  # xdata
        fermi_dirac.(𝐱, μ, β),  # ydata
        𝛉;  # p0
        maxIter=max_iter,
        inplace=true,
    )
    return coef(fitted), stderror(fitted; rtol=rtol), vcov(fitted)
end

function fit_entropy(𝐱, μ, β, nlayers=20; max_iter=1000, rtol=NaN)
    # Initialize model with SP2
    𝛉 = init_params(μ, nlayers)
    fitted = curve_fit(
        entropy_model!,
        entropy_derivatives!,
        𝐱,
        electronic_entropy.(𝐱, μ, β),
        𝛉;
        maxIter=max_iter,
        inplace=true,
    )
    return coef(fitted), stderror(fitted; rtol=rtol), vcov(fitted)
end
