using LsqFit: curve_fit, isconverged, coef, residuals, mse, stderror, vcov

import LsqFit: LMResults

export fit_fermi_dirac, fit_electronic_entropy

_fermi_dirac!(result, X, A) = fermi_dirac!(FlattendModel(A), result, X)  # Only used for fitting

_electronic_entropy!(result, X, A) = electronic_entropy!(FlattendModel(A), result, X)  # Only used for fitting

function fit_fermi_dirac(
    𝛆′,
    μ′,
    β′,
    nlayers=20;
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    is_rescaled=false,
    show_trace=false,
    store_trace=true,
    kwargs...,
)
    if is_rescaled
        _checkdomain(𝛆′, μ′, β′)
    end
    model = init_model(μ′, nlayers)  # Initialize model with SP2
    fd = fermi_dirac.(𝛆′, μ′, β′)
    result = curve_fit(
        _fermi_dirac!,
        fermi_dirac_grad!,
        𝛆′,  # xdata
        fd,  # ydata
        model;  # p0
        maxIter=max_iter,
        maxTime=max_time,
        x_tol=x_tol,
        g_tol=grad_tol,
        inplace=true,
        show_trace=show_trace,
        store_trace=store_trace,
        kwargs...,
    )
    if !isconverged(result)
        @warn "the curve fitting did not converge!"
    end
    return (
        model=FlattendModel(coef(result)),
        jac=result.jacobian,
        resid=residuals(result),
        rmse=sqrt(mse(result)),
        sigma=stderror(result; rtol=neg_rtol),
        covar=vcov(result),
        trace=result.trace,
    )
end

function fit_electronic_entropy(
    𝛆′,
    μ′,
    β′,
    nlayers=20;
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    is_rescaled=false,
    show_trace=false,
    store_trace=true,
    kwargs...,
)
    if is_rescaled
        _checkdomain(𝛆′, μ′, β′)
    end
    model = init_model(μ′, nlayers)  # Initialize model with SP2
    𝐬 = electronic_entropy.(𝛆′, μ′, β′)
    result = curve_fit(
        _electronic_entropy!,
        electronic_entropy_grad!,
        𝛆′,  # xdata
        𝐬,  # ydata
        model;  # p0
        maxIter=max_iter,
        maxTime=max_time,
        x_tol=x_tol,
        g_tol=grad_tol,
        inplace=true,
        show_trace=show_trace,
        store_trace=store_trace,
        kwargs...,
    )
    if !isconverged(result)
        @warn "the curve fitting did not converge!"
    end
    return (
        model=FlattendModel(coef(result)),
        jac=result.jacobian,
        resid=residuals(result),
        rmse=sqrt(mse(result)),
        sigma=stderror(result; rtol=neg_rtol),
        covar=vcov(result),
        trace=result.trace,
    )
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
