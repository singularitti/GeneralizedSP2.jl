using LinearAlgebra: tr, diag

export newton_raphson_step, estimate_mu

function newton_raphson_step(DM, β, target_occupation; occ_atol=1e-4)
    @assert occ_atol >= zero(occ_atol)
    occupation = tr(DM)
    occupation_error = target_occupation - occupation
    derivative = tr(fermi_dirac_deriv(DM, β))
    Δμ = occupation_error / derivative
    return Δμ, abs(occupation_error) <= occ_atol
end

function estimate_mu(
    H,
    𝐱′,
    β,
    target_occupation,
    μ=sum(extrema(diag(H))) / 2,
    𝛆=extrema(H),
    nlayers=20;
    max_iter=1000,
    occ_atol=1e-4,
    kwargs...,
)
    H′ = rescale_one_zero(𝛆)(H)
    β′ = rescale_beta(β, 𝛆)
    μ′ = rescale_mu(μ, 𝛆)
    εₘᵢₙ, εₘₐₓ = extrema(𝛆)
    converged = false
    while !converged
        fitted = fit_fermi_dirac(𝐱′, μ′, β′, nlayers; max_iter=max_iter, kwargs...)
        DM = fermi_dirac(fitted.model)(H′)
        Δμ, converged = newton_raphson_step(DM, β, target_occupation; occ_atol=occ_atol)
        μ′ -= Δμ / (εₘᵢₙ - εₘₐₓ)
    end
    return μ′
end
