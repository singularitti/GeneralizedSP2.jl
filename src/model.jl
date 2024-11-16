using StaticArrays: MArray, MMatrix, Size

export Model, numlayers, eachlayer

struct Model{T,N} <: AbstractMatrix{T}
    data::MMatrix{LAYER_WIDTH,N,T}
    function Model(data::AbstractMatrix)
        if size(data, 1) != LAYER_WIDTH
            throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
        end
        N, T = size(data, 2), eltype(data)
        return new{T,N}(MMatrix{LAYER_WIDTH,N,T}(data))
    end
end
Model(data::AbstractVector) = Model(reshape(data, LAYER_WIDTH, :))
Model(M::Model) = M

numlayers(M::Model) = size(M, 2)

eachlayer(M::Model) = eachcol(M)

Base.parent(M::Model) = M.data

Base.size(M::Model) = size(parent(M))

Base.getindex(M::Model, i::Int) = getindex(parent(M), i)

Base.setindex!(M::Model, v, i::Int) = setindex!(parent(M), v, i)

Base.IndexStyle(::Type{<:Model}) = IndexLinear()

# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta2/base/abstractarray.jl#L839
function Base.similar(M::Model, ::Type{T}, dims::Dims) where {T}
    if length(dims) == 2
        return Model(similar(parent(M), T, dims))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end
# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta1/base/abstractarray.jl#L874
function Base.similar(::Type{<:Model{T}}, dims::Dims) where {T}
    if length(dims) == 2
        return Model(MMatrix{dims...,T}(undef))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end

# Base.convert(::Type{T}, m::Model) where {T<:AbstractVector} = convert(T, vec(m))
# Base.convert(::Type{<:Model}, A::AbstractVector) = Model(A)
# Base.convert(::Type{<:Model{N}}, A::AbstractVector) where {N} =
#     Model(reshape(A, LAYER_WIDTH, N))
# Base.convert(::Type{<:Model{N}}, m::Model{N}) where {N} = m
# Base.convert(::Type{<:Model{N,T}}, m::Model{M,T}) where {N,M,T} =
#     throw(DimensionMismatch(""))
