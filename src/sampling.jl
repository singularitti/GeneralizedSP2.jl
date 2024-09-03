export sample_by_pdf

function sample_by_pdf(pdf, start, lower_bound, upper_bound)
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
