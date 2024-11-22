using LinearAlgebra: tr

export newton_raphson_iteration, estimate_mu

function newton_raphson_iteration(DM, β, occupation₀; occ_atol=1e-7)
    occupation = tr(DM)
    occupation_error = occupation₀ - occupation
    derivatives = fermi_dirac_deriv(DM, β)
    Δμ′ = occupation_error / tr(derivatives)
    return occupation_error > occ_atol ? Δμ′ : zero(Δμ′)
end

function estimate_mu(
    H,
    𝐱′,
    β,
    occupation₀,
    μ=sum(extrema(H)) / 2,
    𝛆=extrema(H),
    nlayers=20;
    max_iter=1000,
    occ_atol=1e-7,
    kwargs...,
)
    H′ = rescale_one_zero(𝛆)(H)
    β′ = rescale_beta(β, 𝛆)
    μ′ = rescale_mu(μ, 𝛆)
    Δμ′ = oneunit(μ′)  # Initialize with a non-zero value
    while !iszero(Δμ′)
        M = fit_fermi_dirac(𝐱′, μ′, β′, nlayers; max_iter=max_iter, kwargs...).model
        DM = fermi_dirac(M)(H′)
        Δμ′ = newton_raphson_iteration(DM, β, occupation₀; occ_atol=occ_atol)
        μ′ += Δμ′
    end
    return μ′
end
