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

struct FlatModel{T} <: Model1D{T}
    data::Vector{T}
    function FlatModel{T}(data::AbstractVector{S}) where {T,S}
        if !iszero(length(data) % LAYER_WIDTH)
            throw(DimensionMismatch("flattend model must have $LAYER_WIDTH×N elements!"))
        end
        return S <: T ? new{S}(data) : new(convert(Vector{T}, data))  # Reduce allocations
    end
end
FlatModel(A::AbstractVector) = FlatModel{eltype(A)}(A)
function FlatModel(A::AbstractVector{<:AbstractVector})
    if any(map(length, A) .!= 4)
        throw(DimensionMismatch("model must have $LAYER_WIDTH×N elements!"))
    end
    return FlatModel(collect(Iterators.flatten(A)))
end
FlatModel(A::AbstractMatrix) = FlatModel(vec(A))
FlatModel(model::FlatModel) = model
FlatModel{T}(::UndefInitializer, dims::Dims{1}) where {T} =
    FlatModel(Vector{T}(undef, dims))

layerwidth(::AbstractModel) = LAYER_WIDTH

numlayers(model::Model) = size(model, 2)
numlayers(model::FlatModel) = numlayers(Model(model))

eachlayer(model::Model) = eachcol(model)
eachlayer(model::FlatModel) = eachcol(Model(model))

Base.parent(model::AbstractModel) = model.data

Base.size(model::AbstractModel) = size(parent(model))

Base.getindex(model::AbstractModel, i::Int) = getindex(parent(model), i)

Base.setindex!(model::AbstractModel, v, i::Int) = setindex!(parent(model), v, i)

Base.IndexStyle(::Type{<:AbstractModel}) = IndexLinear()

# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta2/base/abstractarray.jl#L839
Base.similar(model::AbstractModel, ::Type{T}, dims::Dims{1}) where {T} =
    FlatModel(similar(parent(model), T, dims))
Base.similar(model::AbstractModel, ::Type{T}, dims::Dims{2}) where {T} =
    Model(similar(parent(model), T, dims))
Base.similar(model::AbstractModel, ::Type{T}, dims::Dims) where {T} =
    similar(parent(model), T, dims)
# Override https://github.com/JuliaLang/julia/blob/v1.10.0-beta1/base/abstractarray.jl#L874
Base.similar(::Type{<:AbstractModel{T}}, dims::Dims{1}) where {T} =
    FlatModel(Vector{T}(undef, dims))
Base.similar(::Type{<:AbstractModel{T}}, dims::Dims{2}) where {T} =
    Model(Matrix{T}(undef, dims))
Base.similar(::Type{<:AbstractModel{T}}, dims::Dims{N}) where {T,N} =
    similar(Array{T,N}, dims)

# See https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#When-is-convert-called?
Base.convert(::Type{Model{S}}, model::Model{T}) where {S,T} = Model{S}(model)
Base.convert(::Type{Model}, model::Model{T}) where {T} = model
Base.convert(::Type{FlatModel{S}}, model::FlatModel{T}) where {S,T} = FlatModel{S}(model)
Base.convert(::Type{FlatModel}, model::FlatModel{T}) where {T} = model
Base.convert(::Type{Model{S}}, model::FlatModel{T}) where {S,T} =
    Model(convert(Matrix{S}, model))
Base.convert(::Type{Model}, model::FlatModel{T}) where {T} = Model(model)
Base.convert(::Type{FlatModel{S}}, model::Model{T}) where {S,T} =
    FlatModel(convert(Vector{S}, model))
Base.convert(::Type{FlatModel}, model::Model{T}) where {T} = FlatModel(model)
