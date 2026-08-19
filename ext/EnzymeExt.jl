module EnzymeExt

using DifferentiationInterface: AutoEnzyme
using EnzymeCore:
    Reverse,
    ReverseMode,
    NoPrimal,
    Const,
    Active,
    Duplicated,
    autodiff,
    make_zero

import GeneralizedSP2: autodiff_model!

function autodiff_model!(
    f,
    grad,
    model,
    x,
    backend::AutoEnzyme{<:Union{Nothing,ReverseMode},<:Union{Nothing,Const}},
)
    if length(grad) != length(model)
        throw(DimensionMismatch("the length of gradient and the model are not equal!"))
    end
    shadow = make_zero(model)
    g(model′) = f(model′(x)) # Do not use a `` to construct a `ComposedFunction`, it will add allocations!
    mode = isnothing(backend.mode) ? Reverse : NoPrimal(backend.mode)
    autodiff(mode, g, Active, Duplicated(model, shadow))
    return copyto!(grad, shadow)
end

end
