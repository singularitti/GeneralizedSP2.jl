export fermi_dirac, energyof, entropyof

function fermi_dirac(ε, β, μ)
    η = exp(β * (ε - μ))
    return inv(oneunit(η) + η)
end

function energyof(ε, β, μ)
    η = β * (ε - μ)
    if η > -20oneunit(η)
        return -inv(β) * log1p(exp(-η))  # `log1p(x)` is accurate for `x` near zero
    else
        return -inv(β) * (log1p(exp(η)) - η)  # Avoid overflow for very negative `η`
    end
end

entropyof(ε, β, μ) = β * (fermi_dirac(ε, β, μ) * (ε - μ) - energyof(ε, β, μ))
