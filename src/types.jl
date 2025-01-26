using StaticArrays: SVector, MVector

export Model, numlayers, eachlayer

struct Model{S,T<:AbstractVector{S},N}
    data::SVector{N,T}
end
function Model(A::AbstractMatrix)
    if size(A, 1) != LAYER_WIDTH
        throw(DimensionMismatch("model matrix must have $LAYER_WIDTH rows!"))  # See https://discourse.julialang.org/t/120556/2
    end
    return Model(eachcol(A))
end
function Model(A::AbstractVector{T}) where {S,T<:AbstractVector{S}}
    if innersize(A) != (LAYER_WIDTH,)
        throw(DimensionMismatch("invalid dimensions for `Model`!"))  # See https://discourse.julialang.org/t/120556/2
    end
    return Model{S,T,length(A)}(A)
end
Model(A::AbstractVector) = Model(collect(Iterators.partition(A, LAYER_WIDTH)))
Model(model::Model) = model

numlayers(model::Model) = length(model)

Base.iterate(model::Model, state=firstindex(model)) = iterate(parent(model), state)

Base.IteratorSize(::Type{<:Model}) = Base.HasLength()

Base.length(::Model{S,T,N}) where {S,T,N} = N

Base.eltype(::Type{<:Model{T}}) where {T} = T
Base.eltype(model::Model) = eltype(typeof(model))

Base.parent(model::Model) = model.data

Base.size(model::Model) = size(parent(model))

Base.getindex(model::Model, i) = getindex(parent(model), i)

Base.setindex!(model::Model, v, i) = setindex!(parent(model), v, i)

Base.firstindex(model::Model) = firstindex(parent(model))

Base.lastindex(model::Model) = lastindex(parent(model))

struct EachLayer{N,T}
    data::MVector{N,SVector{4,T}}
end

eachlayer(model::Model) = EachLayer{length(model),eltype(model)}(parent(model))

# Similar to https://github.com/JuliaCollections/IterTools.jl/blob/0ecaa88/src/IterTools.jl#L1028-L1032
function Base.iterate(iter::EachLayer, state=1)
    if state > length(iter)
        return nothing
    else
        return iter.data[state], state + 1
    end
end

Base.IteratorSize(::Type{<:EachLayer}) = Base.HasLength()

Base.length(::EachLayer{N}) where {N} = N

Base.eltype(::Type{EachLayer{N,T}}) where {N,T} = SVector{4,T}
