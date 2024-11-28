using LinearAlgebra: tr, diag

export newton_raphson_step, estimate_mu, update_spectral_bounds, recover_mu_history

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
    max_iter=100,
    fit_max_iter=1000,
    occ_tol=1e-4,
    kwargs...,
)
    if max_iter <= zero(max_iter)
        throw(ArgumentError("`max_iter` must be positive!"))
    end
    μ′_history = typeof(one(float(μ)))[]  # Store μ′
    spectral_bounds_history = [extrema(spectral_bounds)]
    converged = false
    for _ in 1:max_iter
        μ′ = rescale_mu(spectral_bounds)(μ)
        push!(μ′_history, μ′)
        if converged
            break  # This order is important since I want to store the final μ′ or μ without doing unnecessary calculations!
        end
        β′ = rescale_beta(spectral_bounds)(β)
        fitted = fit_fermi_dirac(𝛆′, μ′, β′, nlayers; max_iter=fit_max_iter, kwargs...)
        H′ = rescale_one_zero(spectral_bounds)(H)
        D = fermi_dirac(fitted.model)(H′)
        Δμ, converged = newton_raphson_step(target_occupation, D, β; occ_tol=occ_tol)
        μ -= Δμ
        spectral_bounds = update_spectral_bounds(μ, spectral_bounds)
        push!(spectral_bounds_history, spectral_bounds)
    end
    return if is_rescaled
        μ′_history
    else
        recover_mu_history(μ′_history, spectral_bounds_history)
    end,
    spectral_bounds_history
end

function update_spectral_bounds(μ, spectral_bounds)
    if μ < minimum(spectral_bounds)
        return (floor(typeof(μ), μ), maximum(spectral_bounds))
    elseif μ > maximum(spectral_bounds)
        return (minimum(spectral_bounds), ceil(typeof(μ), μ))
    else
        return extrema(spectral_bounds)
    end
end

function recover_mu_history(μ′_history, spectral_bounds_history)
    return collect(
        inv(rescale_mu(spectral_bounds))(μ′) for
        (spectral_bounds, μ′) in zip(spectral_bounds_history, μ′_history)
    )
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
