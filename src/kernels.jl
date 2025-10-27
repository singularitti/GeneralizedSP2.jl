using KernelAbstractions: @Const, @kernel, @index, get_backend, synchronize

export fermi_dirac!, workgroup_size, set_workgroup_size

@kernel function kernel_fermi_dirac!(result, @Const(ε), μ, β)
    i = @index(Global)
    if i <= length(result)
        @inbounds begin
            η = exp((ε[i] - μ) * β)
            result[i] = inv(oneunit(eltype(result)) + η)
        end
    end
end

function fermi_dirac!(result, ε, μ, β)
    @assert size(result) == size(ε)
    backend = get_backend(result)
    kernel = kernel_fermi_dirac!(backend, workgroup_size())
    kernel(result, ε, μ, β; ndrange=length(result))
    synchronize(backend)
    return result
end

const WorkgroupSize = Ref(128)

workgroup_size() = WorkgroupSize[]
# See https://github.com/KristofferC/OhMyREPL.jl/blob/8b0fc53/src/BracketInserter.jl#L44-L45
set_workgroup_size(size::Integer) = WorkgroupSize[] = size
