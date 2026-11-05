# Every prepared kernel of the real PDESystems in `common/cases.jl`, evaluated in a KernelAbstractions kernel on
# KernelAbstractions.CPU() and on JLArrays, equals the host Float64 kernel. Both run on the host: they test host-array
# discipline, not that a kernel compiles for a GPU (see README.md).
using JLArrays: JLArray, JLBackend
using GenericSparseArrays: GenericSparseMatrixCSC
using LinearAlgebra: mul!
using SparseArrays: sparse
include(joinpath(@__DIR__, "common", "backend.jl"))

GPUArraysCore.allowscalar(false)

@testset "KernelAbstractions.CPU()" begin
    test_backend(KernelAbstractions.CPU(), identity)
end

@testset "JLArrays" begin
    test_backend(JLBackend(), JLArray)
end

@testset "a GenericSparseMatrixCSC with JLArray storage multiplies a JLArray without scalar indexing" begin
    A = sparse([1, 2, 3, 1], [1, 2, 3, 3], Float32[1, 2, 3, 4])
    G = GenericSparseMatrixCSC(size(A)..., JLArray(A.colptr), JLArray(A.rowval), JLArray(A.nzval))
    y = JLArray(zeros(Float32, 3))
    mul!(y, G, JLArray(ones(Float32, 3)))
    @test Array(y) == A * ones(Float32, 3)
end
