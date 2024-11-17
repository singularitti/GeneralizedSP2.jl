export Model, FlattendModel, numlayers, eachlayer

abstract type AbstractModel{T,N} <: AbstractArray{T,N} end

struct Model{T} <: AbstractModel{T,2}
    data::Matrix{T}
    function Model(data::AbstractMatrix)
        if size(data, 1) != LAYER_WIDTH
            throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
        end
        return new{eltype(data)}(data)
    end
end
Model(M::Model) = M

struct FlattendModel{T} <: AbstractModel{T,1}
    data::Vector{T}
    function FlattendModel(data::AbstractVector)
        if !iszero(length(data) % LAYER_WIDTH)
            throw(DimensionMismatch("flattend model must have 4N elements!"))
        end
        return new{eltype(data)}(data)
    end
end
FlattendModel(M::Model) = FlattendModel(vec(M))
FlattendModel(M::FlattendModel) = M

Model(M::FlattendModel) = Model(reshape(parent(M), LAYER_WIDTH, :))

numlayers(M::Model) = size(M, 2)
numlayers(M::FlattendModel) = size(Model(M))

eachlayer(M::Model) = eachcol(M)
eachlayer(M::FlattendModel) = eachcol(Model(M))

Base.parent(M::AbstractModel) = M.data

Base.size(M::AbstractModel) = size(parent(M))

Base.getindex(M::AbstractModel, i::Int) = getindex(parent(M), i)

Base.setindex!(M::AbstractModel, v, i::Int) = setindex!(parent(M), v, i)

Base.IndexStyle(::Type{<:AbstractModel}) = IndexLinear()

# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta2/base/abstractarray.jl#L839
function Base.similar(M::AbstractModel, ::Type{T}, dims::Dims) where {T}
    if length(dims) == 1
        return FlattendModel(similar(parent(M), T, dims))
    elseif length(dims) == 2
        return Model(similar(parent(M), T, dims))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end
# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta1/base/abstractarray.jl#L874
function Base.similar(::Type{<:AbstractModel{T}}, dims::Dims) where {T}
    if length(dims) == 1
        return FlattendModel(Vector{T}(undef, dims))
    elseif length(dims) == 2
        return Model(Matrix{T}(undef, dims))
    else
        return throw(DimensionMismatch("invalid dimensions `$dims` for `Model`!"))
    end
end

# See https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#When-is-convert-called?
Base.convert(::Type{<:FlattendModel{T}}, M::Model{T}) where {T} = FlattendModel(M)
Base.convert(::Type{<:Model{T}}, M::FlattendModel{T}) where {T} = Model(M)
