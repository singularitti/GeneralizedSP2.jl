using CEnum: @cenum
using LinearAlgebra: diagind

export Precision,
    diagonalize, diagonalize!, fill_diagonal, fill_diagonal!, compute_exact_fermi_dirac!

struct CUDAError
    at::Symbol
    msg::String
end

Base.showerror(io::IO, e::CUDAError) = print(io, "CUDA error in `$(e.at)`: $(e.msg)")

@cenum Precision begin
    DOUBLE = 0  # uniform double precision
    SINGLE = 1  # uniform single precision
    MIXED = 2   # accumulate in single, compute in half
end

function diagonalize end

function diagonalize! end

function fill_diagonal!(A::AbstractMatrix{T}, D::AbstractVector{T}) where {T}
    M, N = size(A)
    if M != N  # See https://github.com/JuliaLang/LinearAlgebra.jl/blob/d2872f9/src/LinearAlgebra.jl#L300-L304
        throw(DimensionMismatch(lazy"matrix is not square: dimensions are $(size(A))"))
    end
    if size(D) != (N,)
        throw(DimensionMismatch(lazy"the diagonal vector size must match matrix size"))
    end
    A .= zero(T)  # Fill matrix with zeros
    diag_indices = diagind(A)  # Generate linear indices for diagonal elements
    A[diag_indices] .= D  # Fill diagonal
    return A
end
function fill_diagonal!(A::AbstractMatrix, D::AbstractVector)
    D′ = convert.(eltype(A), D)
    return fill_diagonal!(A, D′)
end
function fill_diagonal(D::AbstractVector)
    A = similar(D, length(D), length(D))
    return fill_diagonal!(A, D)
end
