# End to end, the reassembly policy (spec §8.1–8.3, red-cell m5, §10.1 "remake with a new k reassembles K") and
# the in-place nonlinear operator: a LinearCache reassembles only the stale operators, and NonlinearOperator
# matches assemble_nonlinear and central differences without allocating per cell.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))
using FiniteElementMethod: LinearCache, update!, NonlinearOperator, residual!, jacobian!, residual_jacobian!,
    assemble_linear!

@variables t x y k
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

# K depends on p only, F on t only, M on neither.
function linear_problem()
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + (1 + t) * x * y
    return chain([eq], [BC("left", U ~ 0.0)]; depvars = [U], t, sivs = [x, y], params = [k])
end

fresh(setup, c, p, t) = assemble_linear(setup.ctx, c.kernels, p; t)

@testset "LinearCache reassembles only stale operators" begin
    c = linear_problem()
    setup = mesh_setup(Triangle, 1, 8)
    cache = LinearCache(setup.ctx, c.wf, c.kernels, [2.0]; t = 0.0)
    counts() = (cache.assemblies[:M], cache.assemblies[:K], cache.assemblies[:F])
    @test counts() == (1, 1, 1)

    update!(cache, [2.0], 0.0)
    @test counts() == (1, 1, 1)                              # neither t nor p changed

    K_before = copy(cache.K)
    update!(cache, [2.0], 0.5)
    @test counts() == (1, 1, 2)                              # t-dependent source: F only
    @test cache.K == K_before
    ref = fresh(setup, c, [2.0], 0.5)
    @test relerr(cache.F, ref.F) ≤ 1.0e-14
    @test norm(cache.F - fresh(setup, c, [2.0], 0.0).F) > 0

    update!(cache, [3.0], 0.5)
    @test counts() == (1, 2, 2)                              # remake p: K, not F
    ref = fresh(setup, c, [3.0], 0.5)
    @test relerr(cache.K, ref.K) ≤ 1.0e-14
    @test relerr(cache.F, ref.F) ≤ 1.0e-14
    @test relerr(cache.M, ref.M) ≤ 1.0e-14
    @test relerr(cache.K, fresh(setup, c, [2.0], 0.5).K) > 0.1
end

@testset "LinearCache reassembles a time-dependent mass" begin
    c = chain(
        [(1 + t) * Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + 1], [BC("left", U ~ 0.0)];
        depvars = [U], t, sivs = [x, y]
    )
    setup = mesh_setup(Triangle, 1, 6)
    cache = LinearCache(setup.ctx, c.wf, c.kernels, P0; t = 0.0)
    update!(cache, P0, 1.0)
    @test (cache.assemblies[:M], cache.assemblies[:K], cache.assemblies[:F]) == (2, 1, 1)
    @test relerr(cache.M, fresh(setup, c, P0, 1.0).M) ≤ 1.0e-14
end

# k(u) = 1 + u² in the volume and a nonlinear natural condition on `right`.
function nonlinear_problem()
    eq = 0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + 1
    bcs = [BC("left", U ~ 1.0), BC("right", Dx(U) ~ -(U^4 - 1), RIGHT)]
    return chain([eq], bcs; depvars = [U], t, sivs = [x, y])
end

@testset "NonlinearOperator matches assemble_nonlinear and central differences" begin
    c = nonlinear_problem()
    setup = mesh_setup(Triangle, 1, 10)
    op = NonlinearOperator(setup.ctx, c.kernels)
    u = 0.5 .* randn(Xoshiro(5), ndofs(setup.dh))
    ref = assemble_nonlinear(setup.ctx, c.kernels, u, P0)
    r = zeros(ndofs(setup.dh))
    @test residual!(r, op, u, P0, 0.0) === r
    @test relerr(r, ref.r) ≤ 1.0e-14
    @test relerr(op(similar(r), u, P0, 0.0), ref.r) ≤ 1.0e-14
    J = similar(op.J)
    @test relerr(jacobian!(J, op, u, P0, 0.0), ref.J) ≤ 1.0e-14
    r2, J2 = residual_jacobian!(zeros(ndofs(setup.dh)), similar(op.J), op, u, P0, 0.0)
    @test relerr(r2, ref.r) ≤ 1.0e-14 && relerr(J2, ref.J) ≤ 1.0e-14
    R(v) = residual!(zeros(length(v)), op, v, P0, 0.0)
    @test tangent_error(R, J, u) ≤ 1.0e-9
end

@testset "residuals assemble into a view of a larger buffer" begin
    c = nonlinear_problem()
    setup = mesh_setup(Triangle, 1, 6)
    n = ndofs(setup.dh)
    op = NonlinearOperator(setup.ctx, c.kernels)
    u = 0.5 .* randn(Xoshiro(7), n)
    ref = assemble_nonlinear(setup.ctx, c.kernels, u, P0)
    buffer = fill(NaN, n + 4)
    residual!(view(buffer, 3:(n + 2)), op, u, P0, 0.0)
    @test buffer[3:(n + 2)] == residual!(zeros(n), op, u, P0, 0.0)
    @test relerr(buffer[3:(n + 2)], ref.r) ≤ 1.0e-14
    @test all(isnan, buffer[[1, 2, n + 3, n + 4]])
    buffer = fill(NaN, n + 4)
    _, J = residual_jacobian!(view(buffer, 3:(n + 2)), similar(op.J), op, u, P0, 0.0)
    @test relerr(buffer[3:(n + 2)], ref.r) ≤ 1.0e-14 && relerr(J, ref.J) ≤ 1.0e-14
    lin = linear_problem()
    A = fresh(setup, lin, [2.0], 0.5)
    M, K, Fbuffer = similar(A.M), similar(A.K), fill(NaN, n + 4)
    assemble_linear!(M, K, view(Fbuffer, 3:(n + 2)), setup.ctx, lin.kernels, [2.0]; t = 0.5)
    @test relerr(Fbuffer[3:(n + 2)], A.F) ≤ 1.0e-14
end

@testset "residual! allocation does not scale with the mesh" begin
    c = nonlinear_problem()
    allocations = map((8, 32)) do nel
        setup = mesh_setup(Triangle, 1, nel)
        op = NonlinearOperator(setup.ctx, c.kernels)
        u, r = rand(Xoshiro(6), ndofs(setup.dh)), zeros(ndofs(setup.dh))
        residual!(r, op, u, P0, 0.0)
        @allocated residual!(r, op, u, P0, 0.0)
    end
    @info "residual! allocations (bytes) for 8x8 and 32x32 meshes" allocations
    @test allocations[2] == allocations[1]
end
