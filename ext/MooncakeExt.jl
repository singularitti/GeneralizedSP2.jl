module MooncakeExt

using DifferentiationInterface: AutoMooncake, gradient!
using GeneralizedSP2: AbstractModel
using Mooncake: AsPrimal, Config, FriendlyTangentCache

import GeneralizedSP2: autodiff_model!
import Mooncake: friendly_tangent_cache

friendly_tangent_cache(model::AbstractModel) = FriendlyTangentCache{AsPrimal}(deepcopy(model))

_friendly_config(::Nothing) = Config(; friendly_tangents = true)
function _friendly_config(config::Config)
    config.friendly_tangents && return config
    names = propertynames(config)
    values = map(names) do name
        return name === :friendly_tangents ? true : getproperty(config, name)
    end
    return Config(; NamedTuple{names}(values)...)
end

function autodiff_model!(f, grad, model, x, backend::AutoMooncake)
    if length(grad) != length(model)
        throw(DimensionMismatch("the length of gradient and the model are not equal!"))
    end
    g(model′) = f(model′(x))
    backend′ = AutoMooncake(; config = _friendly_config(backend.config))
    return gradient!(g, grad, backend′, model)
end

end
