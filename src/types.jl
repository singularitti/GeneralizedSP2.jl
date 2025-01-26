using ArraysOfArrays: AbstractVectorOfSimilarVectors, VectorOfSimilarVectors, innersize

export Model, numlayers, eachlayer

struct Model{T} <: AbstractVectorOfSimilarVectors{T}
    data::VectorOfSimilarVectors{T}
end
function Model(A::AbstractMatrix)
    if size(A, 1) != LAYER_WIDTH
        throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
    end
    return Model(eachcol(A))
end
function Model(A::AbstractVector{<:AbstractVector})
    if innersize(A) != (LAYER_WIDTH,)
        throw(DimensionMismatch("invalid dimensions for `Model`!"))  # See https://discourse.julialang.org/t/120556/2
    end
    return Model(VectorOfSimilarVectors(A))
end
Model(A::AbstractVector) = Model(collect(Iterators.partition(A, LAYER_WIDTH)))
Model(model::Model) = model

numlayers(model::Model) = size(model)

eachlayer(model::Model) = (model[i] for i in eachindex(model))

elementtype(::Type{Model{T}}) where {T} = T
elementtype(model::Model) = elementtype(typeof(model))

Base.parent(model::Model) = model.data

Base.size(model::Model) = size(parent(model))

Base.getindex(model::Model, i::Int) = getindex(parent(model), i)

Base.setindex!(model::Model, v, i::Int) = setindex!(parent(model), v, i)

Base.IndexStyle(::Type{<:Model}) = IndexLinear()
