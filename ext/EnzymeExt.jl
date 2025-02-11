module EnzymeExt

using EnzymeCore: Reverse, Active, Duplicated, autodiff
using GeneralizedSP2: AbstractModel, CustomAutoEnzyme, _apply

import GeneralizedSP2: autodiff_model!

function autodiff_model!(f, grad, model, x, ::CustomAutoEnzyme)
    if length(grad) != length(model)
        throw(DimensionMismatch("the length of gradient and the model are not equal!"))
    end
    if !iszero(grad)
        map!(zero, grad, grad)  # Faster than `make_zero!(grad)`
    end
    g(model′) = f(model′(x))  # Do not use a `` to construct a `ComposedFunction`, it will add allocations!
    autodiff(Reverse, g, Active, Duplicated(model, grad))  # In-place
    return grad
end

end
