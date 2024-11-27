using GeneralizedSP2
using Test

@testset "GeneralizedSP2.jl" begin
    # Write your tests here.
    include("primals.jl")
    include("tangents.jl")
end
