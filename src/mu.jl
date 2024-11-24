using LinearAlgebra: tr, diag

export newton_raphson_step, estimate_mu

function newton_raphson_step(target_occupation, DM, β; occ_tol=1e-4)
    @assert occ_tol >= zero(occ_tol) "occupation tolerance must be non-negative!"
    occupation = tr(DM)
    occupation_error = target_occupation - occupation
    derivative = tr(fermi_dirac_deriv(DM, β))
    Δμ = occupation_error / derivative
    converged = abs(occupation_error) <= occ_tol
    return Δμ, converged
end

function estimate_mu(
    target_occupation,
    H,
    β,
    𝛆′,
    𝛜=extrema(H),
    μ=sum(extrema(diag(H))) / 2,
    nlayers=20;
    is_rescaled=true,
    fitting_max_iter=1000,
    occ_tol=1e-4,
    kwargs...,
)
    H′ = rescale_one_zero(𝛜)(H)
    μ′ = rescale_mu(μ, 𝛜)
    β′ = rescale_beta(β, 𝛜)
    factor = inv(minimum(𝛜) - maximum(𝛜))
    history = [float(μ′)]
    converged = false
    while !converged
        fitted = fit_fermi_dirac(𝛆′, μ′, β′, nlayers; max_iter=fitting_max_iter, kwargs...)
        DM = fermi_dirac(fitted.model)(H′)
        Δμ, converged = newton_raphson_step(target_occupation, DM, β; occ_tol=occ_tol)
        Δμ′ = Δμ * factor
        μ′ -= Δμ′
        push!(history, μ′)
    end
    return is_rescaled ? history : map(Base.Fix2(recover_mu, 𝛜), history)
end
