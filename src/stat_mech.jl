export fermi_dirac, energyof, entropyof

fermi_dirac(x, β, μ) = 1 / (1 + exp(β * (x - μ)))

function energyof(x, β, μ)
    η = β * (x - μ)
    if η > 0
        return -inv(β) * log(1 + exp(-η))
    else
        # Avoid overflow for very negative η
        return -inv(β) * (log(1 + exp(η)) - η)
    end
end

entropyof(x, β, μ) = β * (fermi_dirac(x, β, μ) * (x - μ) - energyof(x, β, μ))
