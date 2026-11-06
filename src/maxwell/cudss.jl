# The cuDSS sparse direct solver of `solve_sweep`, implemented by the FiniteElementMethodCUDSSExt extension. The
# host-side pieces (the CSR layout of a CSC matrix and the per-frequency value gather) live here so that they are
# testable without a GPU.

"""
    CuDSSSolver(; structure = "G")

A [`solve_sweep`](@ref) solver that factorises `A` on an NVIDIA GPU with cuDSS: `solve_sweep(model, omegas;
solver = CuDSSSolver())`. It needs `using CUDA, CUDSS` (the FiniteElementMethodCUDSSExt extension) and a
functional CUDA device.

Assembly stays on the CPU. The first frequency uploads `A` in CSR layout and runs the cuDSS analysis
(reordering and symbolic factorisation) and the numerical factorisation; every later frequency copies only the
new nonzero values to the device and refactorises. All port excitations (the columns of `B`) are solved at once
on the device and returned to the host.

`structure = "G"` (the default) is cuDSS's general LDU factorisation. `structure = "S"` is its symmetric LDLᵀ,
which for complex values treats `A` as complex symmetric (`transpose(A) == A`, true of the Maxwell operator);
it halves the factor storage but is documented for complex values only from CUDSS.jl 1.0 (cuDSS 1.0).
"""
struct CuDSSSolver
    structure::String
    function CuDSSSolver(; structure::AbstractString = "G")
        structure in ("G", "S") ||
            throw(ArgumentError("CuDSSSolver structure must be \"G\" or \"S\", got \"$structure\""))
        return new(String(structure))
    end
end

function (::CuDSSSolver)(A, B, cache)
    loaded = Base.get_extension(@__MODULE__, :FiniteElementMethodCUDSSExt) !== nothing
    msg = loaded ?
        "CuDSSSolver solves a SparseMatrixCSC with Float32, Float64, ComplexF32 or ComplexF64 values and a dense right-hand side of the same element type; got $(typeof(A)) and $(typeof(B))" :
        "CuDSSSolver needs the CUDA and CUDSS packages: run `using CUDA, CUDSS` first"
    throw(ArgumentError(msg))
end

# csr_layout(A::SparseMatrixCSC) -> (rowptr, colval, perm)
#
# The compressed-sparse-row arrays of `A` with `Int32` indices (1-based, as cuSPARSE's `CuSparseMatrixCSR` stores
# them) and the permutation with `A.nzval[perm]` the values in that row order. The pattern of the sweep operator
# never changes, so `perm` is computed once and every later frequency only gathers its values.
function csr_layout(A::SparseMatrixCSC)
    nnz(A) < typemax(Int32) ||
        throw(ArgumentError("CuDSSSolver uses Int32 indices; the matrix has $(nnz(A)) stored entries"))
    positions = SparseMatrixCSC(size(A)..., A.colptr, A.rowval, collect(1:nnz(A)))
    byrow = copy(transpose(positions))
    return Int32.(byrow.colptr), Int32.(byrow.rowval), byrow.nzval
end

# csr_values!(vals, A::SparseMatrixCSC, perm)
#
# Overwrite `vals` with the nonzero values of `A` in the row order of `csr_layout`.
function csr_values!(vals, A::SparseMatrixCSC, perm)
    length(perm) == nnz(A) ||
        throw(DimensionMismatch("the sparsity pattern changed: $(nnz(A)) stored entries, $(length(perm)) expected"))
    vals .= view(nonzeros(A), perm)
    return vals
end
