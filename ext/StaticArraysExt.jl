module StaticArraysExt

using StaticArrays: StaticArray
using GeneralizedSP2: AbstractModel, FlatModel, Model

import StaticArrays: Size, similar_type

function similar_type(
    ::Type{<:AbstractModel{R,N,A}}, ::Type{T}, s::Size{S}
) where {R,N,A<:StaticArray,T,S}
    if length(s) == 1
        return FlatModel{T,similar_type(A, T, s)}
    elseif length(s) == 2
        return Model{T,similar_type(A, T, s)}
    else
        return similar_type(A, T, s)
    end
end
similar_type(::Type{M}, size::Size) where {M<:AbstractModel} =
    similar_type(M, eltype(M), size)
similar_type(::Type{M}, ::Type{T}) where {R,N,A<:StaticArray,M<:AbstractModel{R,N,A},T} =
    similar_type(M, T, Size(M))
similar_type(::Type{M}) where {M<:AbstractModel} = M

function Base.similar(
    ::Type{<:AbstractModel{R,N,A}}, ::Type{T}, s::Size{S}
) where {R,N,A<:StaticArray,T,S}
    if length(S) == 1
        return FlatModel(similar(A, T, s))
    elseif length(S) == 2
        return Model(similar(A, T, s))
    else
        return similar(A, T, s)
    end
end
Base.similar(::Type{M}, ::Type{T}) where {R,N,A<:StaticArray,M<:AbstractModel{R,N,A},T} =
    similar(M, T, Size(M))
Base.similar(::M, ::Type{T}) where {R,N,A<:StaticArray,M<:AbstractModel{R,N,A},T} =
    similar(M, Size(M))
Base.similar(::Type{M}) where {R,N,A<:StaticArray,M<:AbstractModel{R,N,A}} =
    similar(M, Size(M))

Size(::Type{M}) where {R,N,A<:StaticArray,M<:AbstractModel{R,N,A}} = Size(A)

end
