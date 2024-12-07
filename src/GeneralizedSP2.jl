module GeneralizedSP2

const LAYER_WIDTH = 4

include("stat_mech.jl")
include("types.jl")
include("primals.jl")
include("tangents.jl")
include("sampling.jl")
include("sp2.jl")
include("fitting.jl")
include("mu.jl")
include("io.jl")
include("cuda.jl")

end
