using LsqFit: curve_fit, isconverged, coef, residuals, mse, stderror, vcov

import LsqFit: LMResults

export fit_fermi_dirac, fit_entropy

# See https://github.com/JuliaMath/Roots.jl/blob/bf0da62/src/utils.jl#L9-L11
struct ConvergenceFailed
    msg::String
end

function fit_fermi_dirac(
    𝐱,
    μ,
    β,
    nlayers=20;
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    is_rescaled=false,
    show_trace=false,
    kwargs...,
)
    if is_rescaled
        _checkdomain(𝐱, μ, β)
    end
    𝛉 = FlattendModel(init_model(μ, nlayers))  # Initialize model with SP2
    result = curve_fit(
        fermi_dirac!,
        fermi_dirac_grad!,
        𝐱,  # xdata
        fermi_dirac.(𝐱, μ, β),  # ydata
        𝛉;  # p0
        maxIter=max_iter,
        maxTime=max_time,
        x_tol=x_tol,
        g_tol=grad_tol,
        inplace=true,
        show_trace=show_trace,
        kwargs...,
    )
    if isconverged(result)
        return (
            model=FlattendModel(coef(result)),
            jac=result.jacobian,
            resid=residuals(result),
            rmse=sqrt(mse(result)),
            sigma=stderror(result; rtol=neg_rtol),
            covar=vcov(result),
        )
    end
    throw(ConvergenceFailed("the curve fitting did not converge!"))
end

function fit_entropy(
    𝐱,
    μ,
    β,
    nlayers=20;
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    is_rescaled=false,
    show_trace=false,
    kwargs...,
)
    if is_rescaled
        _checkdomain(𝐱, μ, β)
    end
    𝛉 = FlattendModel(init_model(μ, nlayers))  # Initialize model with SP2
    result = curve_fit(
        electronic_entropy!,
        electronic_entropy_grad!,
        𝐱,
        electronic_entropy.(𝐱, μ, β),
        𝛉;
        maxIter=max_iter,
        maxTime=max_time,
        x_tol=x_tol,
        g_tol=grad_tol,
        inplace=true,
        show_trace=show_trace,
        kwargs...,
    )
    if isconverged(result)
        return (
            model=FlattendModel(coef(result)),
            jac=result.jacobian,
            resid=residuals(result),
            rmse=sqrt(mse(result)),
            sigma=stderror(result; rtol=neg_rtol),
            covar=vcov(result),
        )
    end
    throw(ConvergenceFailed("the curve fitting did not converge!"))
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

LMResults(method, initial_x::AbstractModel, minimizer::AbstractModel, args...) =
    LMResults(method, convert(Vector, initial_x), convert(Vector, minimizer), args...)
