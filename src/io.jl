using NPZ: npzread, npzwrite

export loadmodel, savemodel

function loadmodel(file)
    ext = lowercase(splitext(file)[2])
    if ext in (".npy", ".npz")
        model = npzread(file)
        if model isa AbstractMatrix
            return model
        else
            error("this is not a model!")
        end
    else
        throw(ArgumentError("unrecognized file extension $ext."))
    end
end

function savemodel(file, model::AbstractMatrix)
    ext = lowercase(splitext(file)[2])
    if ext in (".npy", ".npz")
        npzwrite(file, model)
    else
        throw(ArgumentError("unrecognized file extension $ext."))
    end
end
