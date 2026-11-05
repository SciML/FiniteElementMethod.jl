# The cuDSS sweep solver against UMFPACK on the WR-90 sweep. Without a functional CUDA device only the
# extension's loading and inference are checked.
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Test
using CUDA, CUDSS, JET

include(joinpath(@__DIR__, "..", "shared", "wr90.jl"))

const EXT = Base.get_extension(FiniteElementMethod, :FiniteElementMethodCUDSSExt)

@testset "extension loads and infers" begin
    @test EXT !== nothing
    model = wr90(5.0, 20.0)
    A = maxwell_operator(model, F10)
    B = FiniteElementMethod.port_excitations(model, F10)
    solver = CuDSSSolver()
    @test_opt target_modules = (EXT,) solver(A, B, nothing)
    @test_call target_modules = (EXT,) solver(A, B, nothing)
    Cache = EXT.CuDSSCache{ComplexF64, CUDSS.CudssSolver{ComplexF64, Int32}}
    @test only(Base.return_types(solver, (typeof(A), typeof(B), Nothing))) == Tuple{Matrix{ComplexF64}, Cache}
    @test only(Base.return_types(solver, (typeof(A), typeof(B), Cache))) == Tuple{Matrix{ComplexF64}, Cache}
    @test_throws "SparseMatrixCSC" solver(Matrix(A), B, nothing)  # dense A is not for cuDSS
end

# The complex symmetric LDLᵀ is documented from CUDSS.jl 1.0 on.
const STRUCTURES = pkgversion(CUDSS) >= v"1" ? ["G", "S"] : ["G"]

if CUDA.functional()
    @testset "cuDSS sweep equals UMFPACK ($structure)" for structure in STRUCTURES
        model = wr90(3.0, 30.0)
        ωs = 2π .* [9.0e9, 10.0e9, 11.0e9]
        cpu = solve_sweep(model, ωs; fields = true)
        calls = Ref(0)
        cudss_solver = CuDSSSolver(; structure)
        counting(A, B, cache) = (calls[] += 1; cudss_solver(A, B, cache))
        gpu = solve_sweep(model, ωs; fields = true, solver = counting)
        @test calls[] == length(ωs)
        @test gpu.S ≈ cpu.S rtol = 1.0e-8
        for k in eachindex(ωs)
            @test gpu.fields[k] ≈ cpu.fields[k] rtol = 1.0e-8
        end
    end
else
    @info "CUDA is not functional here; the cuDSS solve against UMFPACK is skipped"
    @test_skip "cuDSS sweep equals UMFPACK"
end
