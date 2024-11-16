module GeneralizedSP2

const LAYER_WIDTH = 4

include("rescaler.jl")
include("stat_mech.jl")
include("model.jl")
include("models.jl")
include("fitting.jl")
include("sp2.jl")
include("sampling.jl")
include("io.jl")

end
