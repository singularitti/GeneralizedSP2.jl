using LsqFit: curve_fit, coef, stderror, vcov

export fit_fermi_dirac, fit_entropy

function fit_fermi_dirac(𝐱, μ, β, nlayers=20; max_iter=1000, rtol=NaN)
    _checkdomain(𝐱, μ, β)
    𝛉 = init_params(μ, nlayers)  # Initialize model with SP2
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
    _checkdomain(𝐱, μ, β)
    𝛉 = init_params(μ, nlayers)  # Initialize model with SP2
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

function _checkdomain(𝐱, μ, β)
    if zero(eltype(𝐱)) <= minimum(𝐱) <= oneunit(eltype(𝐱))
        throw(DomainError("𝐱 must be in the range [0, 1]!"))
    end
    if zero(μ) <= μ <= oneunit(μ)
        throw(DomainError("μ must be in the range [0, 1]!"))
    end
    if β < zero(β)
        throw(DomainError("β must be positive!"))
    end
end
