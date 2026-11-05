module FiniteElementMethodGenericSparseArraysExt

# The sparse matrix of backends without a sparse matrix library of their own (JLArrays here; Metal in Ferrite): a
# GenericSparseMatrixCSC whose storage vectors live on the backend. Vendor extensions name more specific types.
using FiniteElementMethod: FiniteElementMethod as FEM
using Adapt: adapt
using GenericSparseArrays: GenericSparseMatrixCSC
using KernelAbstractions: KernelAbstractions as KA
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, rowvals

FEM.device_matrix_type(::KA.GPU, Tv, Ti) = GenericSparseMatrixCSC{Tv, Ti}

function FEM.device_matrix(::Type{<:GenericSparseMatrixCSC{Tv, Ti}}, backend, A) where {Tv, Ti}
    storage(v, S) = adapt(backend, Vector{S}(v))
    return GenericSparseMatrixCSC(size(A)..., storage(SparseArrays.getcolptr(A), Ti), storage(rowvals(A), Ti), storage(nonzeros(A), Tv))
end

end # module
