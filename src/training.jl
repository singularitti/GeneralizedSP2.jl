using LinearAlgebra
using Printf
using JLSO: JLSO

export determine_branches, backward_pass, forward_pass, init_params

### Vanilla SP2, targeting specific chemical potential

const LAYER_WIDTH = 4

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

function forward_pass(branches, x)
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
    throw(ArgumentError("x must be in the interval (0, 1)!"))
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

### Generalized model

# Postprocessing for final model output, and derivative
fermi_transf_1(Y) = 1 - Y
fermi_transf_2(Y) = -1.0
entropy_transf_1(Y) = 4log(2) * (Y - Y^2)
entropy_transf_2(Y) = 4log(2) * (1 - 2Y)

function model(x, θ, f)
    res = Vector{eltype(x)}(undef, length(x))
    model_inplace(res, x, θ, f)
    return res
end

model_fermi(x, θ) = model(x, θ, fermi_transf_1)
model_entropy(x, θ) = model(x, θ, entropy_transf_1)

#= 
function test_jacobian(x, θ, f, df_dY, idx)
    θp = copy(θ)
    θn = copy(θ)

    ϵ = 1e-6
    θp[idx] += ϵ
    θn[idx] -= ϵ
    J_idx = (model(x, θp, f) - model(x, θn, f)) ./ 2ϵ

    J = zeros(length(x), length(θ))
    jacobian_inplace(J, x, θ, df_dY)

    norm(J[:, idx] - J_idx)
end
test_jacobian(x, θ_fermi, fermi_transf_1,   fermi_transf_2,   22)
test_jacobian(x, θ_sp2,   entropy_transf_1, entropy_transf_2, 22)
=#

### Sample points

function sample_by_density(x0, xlo, xhi, density)
    pts = [x0]

    # add points below
    dx = 1 / density(x0)
    x = x0 - dx
    while x > xlo
        push!(pts, x)
        dx = 2 / density(x) - dx
        x -= dx
    end
    push!(pts, xlo)

    reverse!(pts)

    # add points above
    dx = 1 / density(x0)
    x = x0 + dx
    while x < xhi
        push!(pts, x)
        dx = 2 / density(x) - dx
        x += dx
    end
    push!(pts, xhi)

    return pts
end

function sample_weights(pts)
    @assert length(pts) ≥ 2
    weights = zeros(eltype(pts), length(pts))
    # endpoints get half the normal weight
    weights[1] = (pts[2] - pts[1]) / 2
    weights[end] = (pts[end] - pts[end - 1]) / 2
    # interior points collect half the weight from each of the two neighbor intervals
    weights[2:(end - 1)] = (pts[3:end] - pts[1:(end - 2)]) ./ 2
    return weights
end

### Model training routines

fermi_fn(x, β, μ) = 1 / (1 + exp(β * (x - μ)))

function energy_fn(x, β, μ)
    a = β * (x - μ)
    if a > 0
        -β^-1 * log(1 + exp(-a))
    else
        # avoid overflow for very negative a
        -β^-1 * (log(1 + exp(a)) - a)
    end
end

entropy_fn(x, β, μ) = -β * (energy_fn(x, β, μ) - fermi_fn(x, β, μ) * (x - μ))

function read_or_generate_models(filename, overwrite=false)
    need_calculations = overwrite || !isfile(filename)

    if need_calculations
        max_iters = [1_000, 4_000, 16_000, 64_000]
        βs = [40, 400, 4000]
        μs = [0.1, 0.5]

        all_models = map(Iterators.product(βs, μs, max_iters)) do (β, μ, max_iter)
            println("Generating β=$β, μ=$μ, max_iter=$max_iter.")
            (θ_sp2, θ_fermi, θ_entropy, x) = generate_model(; β, μ, max_iter, npts_scale=1)
            (; β, μ, max_iter, θ_sp2, θ_fermi, θ_entropy, x)
        end
        all_models = reshape(all_models, :)

        JLSO.save(filename, :models => all_models)
    else
        all_models = JLSO.load(filename)[:models]
    end

    return all_models
end

### Figuring out width of SP2 -- β vs nlayers

# Empirical fitting function:
# 4.75log(β) + 3.2log(sin(π*μ)) - 6.6

#= 
# μ = 1/2
[40 11; 400 22; 4000 33; 40_000 44]

# μ = 1/10
[40 8; 400 19; 4000 29; 40_000 40]

# μ = 1/100
[40 4; 400 11; 4000 22; 40_000 33]

# μ = 1/1000
[40 ?; 400 ?; 4000 ?; 40_000 25]
=#