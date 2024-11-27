using LinearAlgebra: tr, diag

export newton_raphson_step, estimate_mu

ojbective(D::AbstractMatrix, target_occupation) = target_occupation - tr(D)

ojbective_deriv(D::AbstractMatrix, β) = tr(fermi_dirac_deriv(D, β))

function newton_raphson_step(target_occupation, D::AbstractMatrix, β; occ_tol=1e-4)
    @assert occ_tol >= zero(occ_tol) "occupation tolerance must be non-negative!"
    occupation_error = ojbective(D, target_occupation)
    derivative = ojbective_deriv(D, β)
    Δμ = occupation_error / derivative
    converged = abs(occupation_error) <= occ_tol
    return Δμ, converged
end

function estimate_mu(
    target_occupation,
    H,
    β,
    𝛆′,
    spectral_bounds=extrema(H),
    μ=sum(extrema(diag(H))) / 2,
    nlayers=20;
    is_rescaled=true,
    fit_max_iter=1000,
    occ_tol=1e-4,
    kwargs...,
)
    H′ = rescale_one_zero(spectral_bounds)(H)
    μ′ = rescale_mu(spectral_bounds)(μ)
    β′ = rescale_beta(spectral_bounds)(β)
    factor = inv(minimum(spectral_bounds) - maximum(spectral_bounds))
    history = [float(μ′)]
    converged = false
    while !converged
        fitted = fit_fermi_dirac(𝛆′, μ′, β′, nlayers; max_iter=fit_max_iter, kwargs...)
        DM = fermi_dirac(fitted.model)(H′)
        Δμ, converged = newton_raphson_step(target_occupation, DM, β; occ_tol=occ_tol)
        Δμ′ = Δμ * factor
        μ′ -= Δμ′
        push!(history, μ′)
    end
    return is_rescaled ? history : map(inv(rescale_mu(spectral_bounds)), history)
end

function bisection(D::AbstractMatrix, lower, upper; tol=1e-6, max_iter=100)
    if ojbective(D, lower) * ojbective(D, upper) >= 0
        throw(DomainError("ojbective(a) and ojbective(b) must have opposite signs"))
    end
    left, right = lower, upper
    mid = (left + right) / 2
    for iter in 1:max_iter
        mid = (left + right) / 2
        g_mid = ojbective(D, mid)
        if abs(g_mid) < tol || (right - left) / 2.0 < tol
            return mid
        end
        if ojbective(D, left) * g_mid < 0
            right = mid
        else
            left = mid
        end
    end
    return error("Bisection method did not converge within max_iter iterations")
end
