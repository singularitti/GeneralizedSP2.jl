using LinearAlgebra
# using JLSO: JLSO

### Vanilla SP2, targeting specific chemical potential

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

function read_or_generate_models(filename, overwrite=false)
    need_calculations = overwrite || !isfile(filename)

    if need_calculations
        max_iters = [1_000, 4_000, 16_000, 64_000]
        βs = [40, 400, 4000]
        μs = [0.1, 0.5]

        all_models = map(Iterators.product(βs, μs, max_iters)) do (β, μ, max_iter)
            println("Generating β=$β, μ=$μ, max_iter=$max_iter.")
            (θ_sp2, θ_fermi, θ_entropy, x) = fit_model(; β, μ, max_iter, npts_scale=1)
            (; β, μ, max_iter, θ_sp2, θ_fermi, θ_entropy, x)
        end
        all_models = reshape(all_models, :)

        # JLSO.save(filename, :models => all_models)
    else
        # all_models = JLSO.load(filename)[:models]
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