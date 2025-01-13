export Model, numlayers, eachlayer

struct Model{T} <: AbstractMatrix{T}
    data::Matrix{T}
    function Model{T}(data::AbstractMatrix{S}) where {T,S}
        if size(data, 1) != LAYER_WIDTH
            throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
        end
        return S <: T ? new{S}(data) : new(convert(Matrix{T}, data))  # Reduce allocations
    end
end
Model(A::AbstractMatrix) = Model{eltype(A)}(A)
Model(A::AbstractVector) = Model(reshape(parent(A), LAYER_WIDTH, :))
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
    if length(dims) in (1, 2)
        return Model(similar(parent(M), T, dims))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end
# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta1/base/abstractarray.jl#L874
function Base.similar(::Type{<:Model{T}}, dims::Dims) where {T}
    N = length(dims)
    if N in (1, 2)
        return Model(Array{T,N}(undef, dims))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end

# See https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#When-is-convert-called?
Base.convert(::Type{Model{S}}, M::Model{T}) where {S,T} = Model(convert(Matrix{S}, M))
Base.convert(::Type{Model}, M::Model{T}) where {T} = convert(Model{eltype(M)}, M)
