using NPZ: npzread, npzwrite

export loadmodel, savemodel

function loadmodel(file)
    ext = lowercase(splitext(file)[2])
    if ext in (".npy", ".npz")
        A = npzread(file)
        if A isa AbstractVector
            return FlattendModel(A)
        elseif A isa AbstractMatrix
            return Model(A)
        else
            error("this is not a model!")
        end
    else
        throw(ArgumentError("unrecognized file extension $ext."))
    end
end

function savemodel(file, model::AbstractModel)
    ext = lowercase(splitext(file)[2])
    if ext in (".npy", ".npz")
        npzwrite(file, model)
    else
        throw(ArgumentError("unrecognized file extension $ext."))
    end
end
