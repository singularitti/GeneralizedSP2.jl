struct FoldList{L,T,F}
    f::F
    seed::T
    list::L
end

foldlist(f, seed, list) = FoldList(f, seed, list)

Base.IteratorEltype(::Type{<:FoldList}) = Base.EltypeUnknown()

Base.IteratorSize(::Type{<:FoldList{L}}) where {L} = Base.IteratorSize(L)

function Base.iterate(it::FoldList)
    next = iterate(it.list)
    if next === nothing
        return nothing
    end
    item, state = next
    val = it.f(it.seed, item)
    return val, (val, state)
end
function Base.iterate(it::FoldList, (val, state))
    next = iterate(it.list, state)
    if next === nothing
        return nothing
    end
    newitem, state = next
    newval = it.f(val, newitem)
    return newval, (newval, state)
end

Base.length(it::FoldList) = length(it.list)

Base.size(it::FoldList) = size(it.list)
