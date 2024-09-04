export sample_by_pdf, bell_distribution, chebyshevnodes_1st, chebyshevnodes_2nd

function sample_by_pdf(pdf, start, (lower_bound, upper_bound))
    points = [start]

    # Add points below `start`
    dx = 1 / pdf(start)
    x = start - dx
    while x > lower_bound
        push!(points, x)
        dx = 2 / pdf(x) - dx
        x -= dx
    end
    push!(points, lower_bound)

    reverse!(points)

    # Add points above `start`
    dx = 1 / pdf(start)
    x = start + dx
    while x < upper_bound
        push!(points, x)
        dx = 2 / pdf(x) - dx
        x += dx
    end
    push!(points, upper_bound)

    return points
end

# Sample points more densely near x=μ
function bell_distribution(μ, β, npoints_scale=1.0)
    npoints = npoints_scale * 80log(β)
    w = sqrt(β)
    return ε -> npoints / 2 + npoints * w / 4 * sech(w * (ε - μ))^2
end

chebyshevnodes_1st(order, (lower_bound, upper_bound)) =
    (lower_bound + upper_bound) / 2 .+
    (lower_bound - upper_bound) / 2 * cospi.((2 * (1:order) .- 1) / 2order)  # Transforms Chebyshev nodes from [-1, 1] to [lower_bound, upper_bound]

chebyshevnodes_2nd(order, (lower_bound, upper_bound)) =
    (lower_bound + upper_bound) / 2 .+
    (lower_bound - upper_bound) / 2 * cospi.((0:(order - 1)) ./ (order - 1))  # Transforms Chebyshev nodes from [-1, 1] to [lower_bound, upper_bound]
