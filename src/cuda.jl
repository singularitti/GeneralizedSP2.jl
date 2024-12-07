export diagonalize, diagonalize!

struct CUDAError
    at::Symbol
    msg::String
end

Base.showerror(io::IO, e::CUDAError) = print(io, "CUDA error in `$(e.at)`: $(e.msg)")

function diagonalize end

function diagonalize! end
