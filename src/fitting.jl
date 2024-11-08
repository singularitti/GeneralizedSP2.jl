using LsqFit: curve_fit, coef, stderror, vcov

export fit_fermi_dirac, fit_entropy

function fit_fermi_dirac(
    𝐱, μ, β, nlayers=20; max_iter=1000, rtol=NaN, check_domain=false, show_trace=false
)
    if check_domain
        _checkdomain(𝐱, μ, β)
    end
    𝛉 = init_params(μ, nlayers)  # Initialize model with SP2
    fitted = curve_fit(
        fermi_dirac_model!,
        fermi_dirac_derivatives!,
        𝐱,  # xdata
        fermi_dirac.(𝐱, μ, β),  # ydata
        𝛉;  # p0
        maxIter=max_iter,
        inplace=true,
        show_trace=show_trace,
    )
    return coef(fitted), stderror(fitted; rtol=rtol), vcov(fitted)
end

function fit_entropy(
    𝐱, μ, β, nlayers=20; max_iter=1000, rtol=NaN, check_domain=false, show_trace=false
)
    if check_domain
        _checkdomain(𝐱, μ, β)
    end
    𝛉 = init_params(μ, nlayers)  # Initialize model with SP2
    fitted = curve_fit(
        entropy_model!,
        entropy_derivatives!,
        𝐱,
        electronic_entropy.(𝐱, μ, β),
        𝛉;
        maxIter=max_iter,
        inplace=true,
        show_trace=show_trace,
    )
    return coef(fitted), stderror(fitted; rtol=rtol), vcov(fitted)
end

function _checkdomain(𝐱, μ, β)
    if minimum(𝐱) < zero(eltype(𝐱)) || maximum(𝐱) > oneunit(eltype(𝐱))
        throw(DomainError("rescaled 𝐱 must be in the range [0, 1]!"))
    end
    if μ < zero(μ) || μ > oneunit(μ)
        throw(DomainError("rescaled μ must be in the range [0, 1]!"))
    end
    if β >= zero(β)
        throw(DomainError("rescaled β must be negative!"))
    end
end
