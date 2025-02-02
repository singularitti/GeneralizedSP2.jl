export Model, layerwidth, numlayers, eachlayer

const LAYER_WIDTH = 4

abstract type AbstractModel{T,N} <: AbstractArray{T,N} end
const Model2D{T} = AbstractModel{T,2}
const Model1D{T} = AbstractModel{T,1}

struct Model{T} <: Model2D{T}
    data::Matrix{T}
    function Model{T}(data::AbstractMatrix{S}) where {T,S}
        if size(data, 1) != LAYER_WIDTH
            throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
        end
        return S <: T ? new{S}(data) : new(convert(Matrix{T}, data))  # Reduce allocations
    end
end
Model(A::AbstractMatrix) = Model{eltype(A)}(A)
function Model(A::AbstractVector{<:AbstractVector})
    if any(map(length, A) .!= 4)
        throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))
    end
    return Model(hcat(A...))
end
Model(A::AbstractVector) = Model(reshape(parent(A), LAYER_WIDTH, :))
Model(model::Model) = model
Model{T}(::UndefInitializer, dims::Dims{2}) where {T} = Model(Matrix{T}(undef, dims))

struct FlattendModel{T} <: Model1D{T}
    data::Vector{T}
    function FlattendModel{T}(data::AbstractVector{S}) where {T,S}
        if !iszero(length(data) % LAYER_WIDTH)
            throw(DimensionMismatch("flattend model must have $LAYER_WIDTH×N elements!"))
        end
        return S <: T ? new{S}(data) : new(convert(Vector{T}, data))  # Reduce allocations
    end
end
FlattendModel(A::AbstractVector) = FlattendModel{eltype(A)}(A)
function FlattendModel(A::AbstractVector{<:AbstractVector})
    if any(map(length, A) .!= 4)
        throw(DimensionMismatch("model must have $LAYER_WIDTH×N elements!"))
    end
    return FlattendModel(collect(Iterators.flatten(A)))
end
FlattendModel(A::AbstractMatrix) = FlattendModel(vec(A))
FlattendModel(model::FlattendModel) = model
FlattendModel{T}(::UndefInitializer, dims::Dims{1}) where {T} =
    FlattendModel(Vector{T}(undef, dims))

layerwidth(::AbstractModel) = LAYER_WIDTH

numlayers(model::Model) = size(model, 2)
numlayers(model::FlattendModel) = numlayers(Model(model))

eachlayer(model::Model) = eachcol(model)
eachlayer(model::FlattendModel) = eachcol(Model(model))

Base.parent(model::AbstractModel) = model.data

Base.size(model::AbstractModel) = size(parent(model))

Base.getindex(model::AbstractModel, i::Int) = getindex(parent(model), i)

Base.setindex!(model::AbstractModel, v, i::Int) = setindex!(parent(model), v, i)

Base.IndexStyle(::Type{<:AbstractModel}) = IndexLinear()

# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta2/base/abstractarray.jl#L839
function Base.similar(model::AbstractModel, ::Type{T}, dims::Dims) where {T}
    if dims isa Dims{1}
        return FlattendModel(similar(parent(model), T, dims))
    elseif dims isa Dims{2}
        return Model(similar(parent(model), T, dims))
    else
        return similar(parent(model), T, dims)
    end
end
# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta1/base/abstractarray.jl#L874
function Base.similar(::Type{<:AbstractModel{T}}, dims::Dims{N}) where {T,N}
    if dims isa Dims{1}
        return FlattendModel(Vector{T}(undef, dims))
    elseif dims isa Dims{2}
        return Model(Matrix{T}(undef, dims))
    else
        return similar(Array{T,N}, dims)
    end
end

# See https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#When-is-convert-called?
Base.convert(::Type{Model{S}}, model::Model{T}) where {S,T} = Model{S}(model)
Base.convert(::Type{Model}, model::Model{T}) where {T} = model
Base.convert(::Type{FlattendModel{S}}, model::FlattendModel{T}) where {S,T} =
    FlattendModel{S}(model)
Base.convert(::Type{FlattendModel}, model::FlattendModel{T}) where {T} = model
Base.convert(::Type{Model{S}}, model::FlattendModel{T}) where {S,T} =
    Model(convert(Matrix{S}, model))
Base.convert(::Type{Model}, model::FlattendModel{T}) where {T} = Model(model)
Base.convert(::Type{FlattendModel{S}}, model::Model{T}) where {S,T} =
    FlattendModel(convert(Vector{S}, model))
Base.convert(::Type{FlattendModel}, model::Model{T}) where {T} = FlattendModel(model)
