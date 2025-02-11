using LsqFit: curve_fit, isconverged, coef, residuals, mse, stderror, vcov

import LsqFit: LMResults

export init_model, fit_fermi_dirac, fit_electronic_entropy

function fit_fermi_dirac(
    𝛆′,
    μ′,
    β′,
    model_init=init_model(μ′, 20);
    diff=Manual(),
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    show_trace=false,
    store_trace=true,
    kwargs...,
)
    fd = fermi_dirac.(𝛆′, μ′, β′)
    result = curve_fit(
        _fermi_dirac!,
        _fermi_dirac_jac!(diff),
        𝛆′,  # xdata
        fd,  # ydata
        model_init;  # p0
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
        model=Model(coef(result)),
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
    model_init=init_model(μ′, 20);
    diff=Manual(),
    max_iter=1000,
    max_time=Inf,
    x_tol=1e-8,
    grad_tol=1e-12,
    neg_rtol=NaN,
    show_trace=false,
    store_trace=true,
    kwargs...,
)
    𝐬 = electronic_entropy.(𝛆′, μ′, β′)
    result = curve_fit(
        _electronic_entropy!,
        _electronic_entropy_jac!(diff),
        𝛆′,  # xdata
        𝐬,  # ydata
        model_init;  # p0
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
        model=Model(coef(result)),
        jac=result.jacobian,
        resid=residuals(result),
        rmse=sqrt(mse(result)),
        sigma=stderror(result; rtol=neg_rtol),
        covar=vcov(result),
        trace=result.trace,
    )
end

function init_model(μ, nlayers)
    model = similar(Model{eltype(μ)}, LAYER_WIDTH, nlayers)
    branches = determine_branches(μ, nlayers)
    for (i, branch) in zip(1:nlayers, branches)
        if branch  # μᵢ < μ
            model[:, i] = [1, 0, 0, 0] # x' = x^2, increase μᵢ
        else
            model[:, i] = [-1, 2, 0, 0] # x' = 2x - x^2, decrease μᵢ
        end
    end
    return FlatModel(model)
end

_fermi_dirac!(result, 𝐱, M) = map!(fermi_dirac(FlatModel(M)), result, 𝐱)  # Only used for fitting

_electronic_entropy!(result, 𝐱, M) = map!(electronic_entropy(FlatModel(M)), result, 𝐱)  # Only used for fitting

_fermi_dirac_jac!(strategy::DiffStrategy) =
    (derivatives, 𝐱, M) -> fermi_dirac_jac!(derivatives, FlatModel(M), 𝐱, strategy)  # Only used for fitting

_electronic_entropy_jac!(strategy::DiffStrategy) =
    (derivatives, 𝐱, M) -> electronic_entropy_jac!(derivatives, FlatModel(M), 𝐱, strategy)  # Only used for fitting

LMResults(method, initial_x::FlatModel, minimizer::FlatModel, args...) =
    LMResults(method, convert(Vector, initial_x), convert(Vector, minimizer), args...)
