using LinearAlgebra

const kBT = 0.25

function gershgorin(M)
    # find eigenvalue estimates of the matrix M from the Gershgorin circle theorem
    min_e = 0.0
    max_e = 0.0
    n = size(M, 1)

    for i in 1:n
        e = M[i, i]  # Gershgorin eigenvalue circle center
        r = 0.0

        for j in 1:n  # compute sum of abs. val of components in row i
            r += abs(M[i, j])
        end

        r -= abs(e)  # Gershgorin eigenvalue circle radius

        # update min and max eigenvalues as you loop over rows
        if e - r < min_e
            min_e = e - r
        elseif e + r > max_e
            max_e = e + r
        end
    end

    return min_e, max_e
end

function setup_hamiltonian(N, a=0.01)
    𝐇 = diagm(10 * rand(N))
    foreach(1:size(𝐇, 1)) do i
        foreach(1:size(𝐇, 2)) do j
            𝐇[i, j] = exp(-a * (i - j)^2)  # Mimic a non-metallic system or a metallic system at ﬁnite temperature
        end
    end
    return Symmetric(𝐇)
end

𝐇 = setup_hamiltonian(1600)
gershgorin(𝐇)
