module FiniteElementMethodAMDGPUExt

# The AMDGPU specifics of device assembly; the assembly itself is backend-neutral, in FiniteElementMethodKAExt.
using FiniteElementMethod: FiniteElementMethod as FEM
using AMDGPU: ROCBackend
using AMDGPU.rocSPARSE: ROCSparseMatrixCSC

FEM.device_matrix_type(::ROCBackend, Tv, Ti) = ROCSparseMatrixCSC{Tv, Ti}
FEM.supports_float64(::ROCBackend) = true

end # module
