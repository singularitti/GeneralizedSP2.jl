using LinearAlgebra: norm

export determine_branches, backward_pass, forward_pass, init_params, myrun

function determine_branches(μ, nlayers)
    branches = Bool[]
    for _ in 1:nlayers
        μᵢ, _ = backward_pass(branches)  # Solve yᵢ(1 / 2) backwards, we get μᵢ by definition
        push!(branches, μᵢ < μ)
    end
    return branches
end

function backward_pass(branches)
    y′ = 1.0  # y′₀ = 1, accumulator
    y = 1 / 2  # yₙ(μₙ) = 1 / 2
    for branchᵢ₊₁ in Iterators.reverse(branches)  # Starts from the nth layer
        if branchᵢ₊₁  # μᵢ < μ
            y = sqrt(y)  # yᵢ(μₙ) = √yᵢ₊₁(μₙ), you must do this first to get yᵢ before the next line
            y′ *= 2y  # y′ᵢ₊₁ *= 2yᵢ, accumulate backwards
        else
            y = 1 - sqrt(1 - y)  # yᵢ(μₙ) = 1 - √(1 - yᵢ₊₁(μₙ)), you must do this first to get yᵢ before the next line
            y′ *= 2 - 2y  # y′ᵢ₊₁ *= -2yᵢ + 2, accumulate backwards
        end
    end
    return y, 4y′  # μₙ = y₀(1 / 2), β = 4y′ₙ
end

function forward_pass(branches, 𝐱)
    return map(𝐱) do x
        if zero(x) <= x <= oneunit(x)
            y = x
            for bᵢ in branches
                if bᵢ
                    y = y^2
                else
                    y = 2y - y^2
                end
            end
            return y
        end
        throw(ArgumentError("$x is not in the interval (0, 1)!"))
    end
end

function init_params(μ, nlayers)
    θ = zeros(LAYER_WIDTH, nlayers)
    branches = determine_branches(μ, nlayers)

    for (i, branch) in zip(1:nlayers, branches)
        if branch  # μᵢ < μ
            θ[:, i] = [1.0, 0.0, 0.0, 0.0] # x' = x^2, increase μᵢ
        else
            θ[:, i] = [-1.0, 2.0, 0.0, 0.0] # x' = 2x - x^2, decrease μᵢ
        end
    end

    return vec(θ)
end
