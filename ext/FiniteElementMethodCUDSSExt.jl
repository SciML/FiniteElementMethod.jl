module FiniteElementMethodCUDSSExt

using FiniteElementMethod: CuDSSSolver, csr_layout, csr_values!
using CUDA: CuMatrix, CuVector, cuSPARSE
using CUDSS: CudssSolver, cudss, cudss_update
using SparseArrays: SparseMatrixCSC, nonzeros

const CudssFloat = Union{Float32, Float64, ComplexF32, ComplexF64}

# What a CuDSSSolver keeps between frequencies: the host CSR value buffer and its gather permutation, the
# device matrix (whose value array cuDSS reads in place) and the cuDSS analysis and factors.
struct CuDSSCache{T, S}
    perm::Vector{Int}
    vals::Vector{T}
    dA::cuSPARSE.CuSparseMatrixCSR{T, Int32}
    solver::S
end

function (s::CuDSSSolver)(A::SparseMatrixCSC{T}, B::AbstractMatrix{T}, cache) where {T <: CudssFloat}
    dB = CuMatrix{T}(B)
    dX = similar(dB)
    if cache === nothing
        rowptr, colval, perm = csr_layout(A)
        vals = nonzeros(A)[perm]
        dA = cuSPARSE.CuSparseMatrixCSR{T, Int32}(CuVector(rowptr), CuVector(colval), CuVector(vals), size(A))
        solver = CudssSolver(dA, s.structure, 'F')
        cudss("analysis", solver, dX, dB)
        cudss("factorization", solver, dX, dB)
        cache = CuDSSCache(perm, vals, dA, solver)
    else
        csr_values!(cache.vals, A, cache.perm)
        copyto!(cache.dA.nzVal, cache.vals)
        cudss_update(cache.solver, cache.dA)
        cudss("refactorization", cache.solver, dX, dB)
    end
    cudss("solve", cache.solver, dX, dB; asynchronous = false)
    return Array(dX), cache
end

end # module
