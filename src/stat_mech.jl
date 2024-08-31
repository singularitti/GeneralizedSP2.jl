export fermi_dirac, energyof, entropyof

fermi_dirac(ε, β, μ) = 1 / (1 + exp(β * (ε - μ)))

function energyof(ε, β, μ)
    η = β * (ε - μ)
    if η > 0
        return -inv(β) * log(1 + exp(-η))
    else
        # Avoid overflow for very negative η
        return -inv(β) * (log(1 + exp(η)) - η)
    end
end

entropyof(ε, β, μ) = β * (fermi_dirac(ε, β, μ) * (ε - μ) - energyof(ε, β, μ))
