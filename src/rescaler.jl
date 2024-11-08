using LinearAlgebra: I

export rescale_zero_one, rescale_one_zero

struct Rescaler{K,B}
    k::K
    b::B
    function Rescaler(k::K, b::B) where {K,B}
        if iszero(k)
            throw(ArgumentError("The slope `k` must be non-zero!"))
        end
        return new{K,B}(k, b)
    end
end

struct Inverse{T<:Rescaler}
    rescaler::T
end

(r::Rescaler)(x::Number) = r.k * x + r.b  # `x` can be out of the range [min, max]
(r::Rescaler)(X::AbstractMatrix) = r.k * X + r.b * I

(i::Inverse{<:Rescaler})(y::Number) = (y - i.rescaler.b) / i.rescaler.k
(i::Inverse{<:Rescaler})(Y::AbstractMatrix) = (Y - i.rescaler.b * I) / i.rescaler.k

Base.inv(r::Rescaler) = Inverse(r)

function rescale_zero_one(𝐱)  # Map `max` to 1, `min` to 0
    min, max = extrema(𝐱)
    @assert min < max
    k, b = inv(max - min), min / (min - max)
    return Rescaler(k, b)
end
rescale_zero_one(𝐱...) = rescale_zero_one(𝐱)

function rescale_one_zero(𝐱)  # Map `max` to 0, `min` to 1
    min, max = extrema(𝐱)
    @assert min < max
    k, b = inv(min - max), max / (max - min)
    return Rescaler(k, b)
end
rescale_one_zero(𝐱...) = rescale_one_zero(𝐱)

function Base.show(io::IO, ::MIME"text/plain", r::Rescaler)
    k, b = r.k, r.b
    if b < zero(b)
        print(io, "y = $k x - $(abs(b))")
    else
        print(io, "y = $k x + $b")
    end
    return nothing
end
