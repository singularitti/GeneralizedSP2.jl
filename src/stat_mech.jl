export fermi_dirac, energyof, entropyof

function fermi_dirac(ε, μ, β)
    η = exp((ε - μ) * β)
    return inv(oneunit(η) + η)
end

function energyof(ε, μ, β)
    η = (ε - μ) * β
    if η > -20oneunit(η)
        return -inv(β) * log1p(exp(-η))  # `log1p(x)` is accurate for `x` near zero
    else
        return -inv(β) * (log1p(exp(η)) - η)  # Avoid overflow for very negative `η`
    end
end

entropyof(ε, μ, β) = (fermi_dirac(ε, μ, β) * (ε - μ) - energyof(ε, μ, β)) * β
