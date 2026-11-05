# The device-independent pieces of the cuDSS sweep solver: the CSR layout of the sweep operator, the
# per-frequency value gather, and the error without the extension. The GPU solve itself is in test/GPU.
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Test
using FiniteElementMethod: csr_layout, csr_values!

include(joinpath(@__DIR__, "..", "shared", "wr90.jl"))

# The CSR arrays of `A` (rows of the transpose's columns), built independently of csr_layout.
function reference_csr(A)
    rowptr = ones(Int, size(A, 1) + 1)
    colval, vals = Int[], eltype(A)[]
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            if any(==(i), rowvals(A)[nzrange(A, j)])
                push!(colval, j)
                push!(vals, A[i, j])
            end
        end
        rowptr[i + 1] = length(colval) + 1
    end
    return rowptr, colval, vals
end

@testset "CSR layout of a nonsymmetric matrix with stored zeros" begin
    A = sparse([1, 3, 2, 1, 3, 2], [1, 1, 2, 3, 3, 3], ComplexF64[1 + 2im, 3, 0, 4im, 5, 6], 3, 3)
    rowptr, colval, perm = csr_layout(A)
    @test rowptr isa Vector{Int32} && colval isa Vector{Int32}
    ref_rowptr, ref_colval, ref_vals = reference_csr(A)
    @test rowptr == ref_rowptr && colval == ref_colval
    @test nonzeros(A)[perm] == ref_vals
    @test length(colval) == nnz(A)  # the stored zero A[2, 2] stays in the pattern
    vals = similar(nonzeros(A))
    A.nzval .*= 2im
    @test csr_values!(vals, A, perm) == 2im .* ref_vals
    @test_throws DimensionMismatch csr_values!(vals, sparse(A[1:2, 1:2]), perm)
end

@testset "CSR layout of the WR-90 sweep operator" begin
    model = wr90(5.0, 20.0)
    A = maxwell_operator(model, F10)
    Ferrite.apply!(A, model.ch)
    rowptr, colval, perm = csr_layout(A)
    csr = SparseMatrixCSC(size(A)..., Int.(rowptr), Int.(colval), nonzeros(A)[perm])  # CSC of transpose(A)
    @test transpose(csr) == A
    vals = similar(nonzeros(A))
    maxwell_operator!(A, model, 1.1 * F10)  # a new frequency, the same pattern
    Ferrite.apply!(A, model.ch)
    @test transpose(SparseMatrixCSC(size(A)..., Int.(rowptr), Int.(colval), csr_values!(vals, A, perm))) == A
end

@testset "CuDSSSolver without CUDA" begin
    @test CuDSSSolver().structure == "G"
    @test CuDSSSolver(; structure = "S").structure == "S"
    @test_throws "must be \"G\" or \"S\"" CuDSSSolver(; structure = "H")
    @test_throws "using CUDA, CUDSS" solve_sweep(wr90(5.0, 20.0), [F10]; solver = CuDSSSolver())
end
