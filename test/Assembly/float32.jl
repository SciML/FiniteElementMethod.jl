# Element-type-generic assembly: the generated kernels of a heat problem assembled into Float32 storage, on a Float32
# grid with Float32 values and parameters, against the Float64 assembly. The kernels' Float64 literals promote
# inside each call, so the Float32 results are the Float64 ones rounded at the element level.
using FiniteElementMethod: weak_form, emit_kernels, AssemblyContext, assemble_linear, assemble_linear!,
    assemble_nonlinear, assemble_nonlinear!, NonlinearOperator, residual!, jacobian!
using Ferrite, LinearAlgebra, Random, SparseArrays, Symbolics, Test

include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse

relerr(A, B) = norm(A - B) / norm(B)

@variables t x y k
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

function context(T, n)
    grid = generate_grid(Triangle, (n, n), Vec{2, T}((-1, -1)), Vec{2, T}((1, 1)))
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    cv = CellValues(T, QuadratureRule{RefTriangle}(T, 2), ip)
    fv = FacetValues(T, FacetQuadratureRule{RefTriangle}(T, 2), ip)
    return AssemblyContext(dh, [:u], cv, fv)
end

linear_kernels() = emit_kernels(
    weak_form(
        analyse(
            [Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + (1 + t) * x * y + 1], [];
            depvars = [U], t, sivs = [x, y], params = [k]
        )
    )
)
nonlinear_kernels() = emit_kernels(
    weak_form(
        analyse(
            [0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + 1], [];
            depvars = [U], t, sivs = [x, y]
        )
    )
)

@testset "Float32 assembly of M, K, F matches Float64" begin
    kernels = linear_kernels()
    ctx64, ctx32 = context(Float64, 8), context(Float32, 8)
    ref = assemble_linear(ctx64, kernels, [2.0]; t = 0.5)
    @test ref.M isa SparseMatrixCSC{Float64, Int} && ref.K isa SparseMatrixCSC{Float64, Int} && ref.F isa Vector{Float64}

    M = allocate_matrix(SparseMatrixCSC{Float32, Int}, ctx32.dh)
    K, F = copy(M), zeros(Float32, ndofs(ctx32.dh))
    @test assemble_linear!(M, K, F, ctx32, kernels, Float32[2]; t = 0.5f0) === (M, K, F)
    @test eltype(M) == eltype(K) == eltype(F) == Float32
    for (A, B) in ((M, ref.M), (K, ref.K), (F, ref.F))
        @test norm(B) > 0
        @test relerr(A, B) ≤ 1.0e-5
    end
end

@testset "M and K are each accumulated in their own element type" begin
    # A mass coefficient whose element entries overflow Float32 but not Float64.
    ρ = 1.0e40
    kernels = emit_kernels(
        weak_form(
            analyse([ρ * Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + 1], []; depvars = [U], t, sivs = [x, y])
        )
    )
    ctx = context(Float64, 4)
    ref = assemble_linear(ctx, kernels, [0.0])
    @test all(isfinite, nonzeros(ref.M)) && maximum(abs, nonzeros(ref.M)) > floatmax(Float32)
    for (TM, TK) in ((Float64, Float32), (Float32, Float64))
        M = allocate_matrix(SparseMatrixCSC{TM, Int}, ctx.dh)
        K = allocate_matrix(SparseMatrixCSC{TK, Int}, ctx.dh)
        F = zeros(ndofs(ctx.dh))
        assemble_linear!(M, K, F, ctx, kernels, [0.0])
        @test eltype(M) == TM && eltype(K) == TK
        @test all(isfinite, nonzeros(K)) && relerr(K, ref.K) ≤ 1.0e-6
        if TM == Float64
            @test M == ref.M
        else
            @test any(isinf, nonzeros(M))
        end
    end
end

@testset "Float32 nonlinear residual and Jacobian match Float64" begin
    kernels = nonlinear_kernels()
    ctx64, ctx32 = context(Float64, 8), context(Float32, 8)
    n = ndofs(ctx64.dh)
    u = 0.5 .* randn(Xoshiro(4), n)
    ref = assemble_nonlinear(ctx64, kernels, u, [0.0])
    @test ref.r isa Vector{Float64} && ref.J isa SparseMatrixCSC{Float64, Int}

    r, J = zeros(Float32, n), allocate_matrix(SparseMatrixCSC{Float32, Int}, ctx32.dh)
    assemble_nonlinear!(r, J, ctx32, kernels, Float32.(u), Float32[0])
    @test eltype(r) == eltype(J) == Float32
    @test relerr(r, ref.r) ≤ 1.0e-5
    @test relerr(J, ref.J) ≤ 1.0e-5

    # A NonlinearOperator with Float32 buffers, evaluated through its in-place entry points.
    op64 = NonlinearOperator(ctx64, kernels)
    @test typeof(op64.r) == Vector{Float64} && typeof(op64.Je) == Matrix{Float64}
    op = NonlinearOperator(ctx32, op64.volume, op64.groups, Float32.(op64.r), J, Float32.(op64.Je), Float32.(op64.re), Float32.(op64.u_elem))
    @test residual!(zeros(Float32, n), op, Float32.(u), Float32[0], 0.0f0) ≈ r
    @test jacobian!(similar(J), op, Float32.(u), Float32[0], 0.0f0) ≈ J
end
