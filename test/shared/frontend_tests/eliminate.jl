# End-to-end tests of `dirichlet = :eliminate` (spec §7.3): PDESystem -> DAEProblem on the free
# dofs -> solve, checked against exact solutions in the finite-element space and against
# `:algebraic`, with the full field rebuilt for every solution accessor.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, field_range, prescribed_dofs
using Ferrite: Triangle, Line

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using LinearAlgebra: norm

@variables x y u(..) v(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const tol = 1.0e-8
const P2_LINE = Dict(u => Lagrange{RefLine, 2}())

pieces(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
eliminated(args...; kw...) = fem_solve(args...; dirichlet = :eliminate, kw...)

# u* = a + b x + x(1 - x)(1 + t) solves u_t = u_xx + x(1 - x) + 2(1 + t) and is in the P2 space.
bubble_eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x))) + x * (1 - x) + 2 * (1 + t)
bubble(a, b) = (t_, x_) -> a + b * x_ + x_ * (1 - x_) * (1 + t_)

@testset verbose = true "dirichlet = :eliminate" begin
    @testset "homogeneous Dirichlet, 1D P2" begin
        Ω = MeshDomain(rectangle_grid(Line, (4,)))
        lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
        bcs = [u(t, lo) ~ 0, u(t, hi) ~ 0, u(0, x) ~ x * (1 - x)]
        sol, prob = eliminated([bubble_eq], bcs, Ω, (x,), t, [u(t, x)]; spaces = P2_LINE)
        @test SciMLBase.successful_retcode(sol)
        @test length(prob.u0) == 9 - 2
        @test nodal_error(sol, u(t, x), bubble(0, 0)) <= tol
    end

    @testset "nonzero constant Dirichlet, 1D P2" begin
        Ω = MeshDomain(rectangle_grid(Line, (4,)))
        lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
        bcs = [u(t, lo) ~ 2, u(t, hi) ~ 3, u(0, x) ~ 2 + x + x * (1 - x)]
        sol, _ = eliminated([bubble_eq], bcs, Ω, (x,), t, [u(t, x)]; spaces = P2_LINE)
        @test nodal_error(sol, u(t, x), bubble(2, 1)) <= tol
        @test sol[u(t, lo)] ≈ fill(2.0, length(sol.t), 1) atol = 1.0e-14
        @test sol[u(t, hi)] ≈ fill(3.0, length(sol.t), 1) atol = 1.0e-14
    end

    @testset "spatially varying constant Dirichlet with Neumann pieces, 2D P2" begin
        # u* = 1 + x + 2y + x(1 - x)(1 + t): Dirichlet in x, Neumann in y.
        Ω = MeshDomain(rectangle_grid(Triangle, (3, 3)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + x * (1 - x) + 2 * (1 + t)
        bcs = [
            u(t, Γl) ~ 1 + 2y, u(t, Γr) ~ 2 + 2y,
            NormalDerivative(Γb)(u(t, Γb)) ~ -2, NormalDerivative(Γt)(u(t, Γt)) ~ 2,
            u(0, x, y) ~ 1 + x + 2y + x * (1 - x),
        ]
        sol, _ = eliminated(
            [eq], bcs, Ω, (x, y), t, [u(t, x, y)]; spaces = Dict(u => Lagrange{RefTriangle, 2}())
        )
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> 1 + x_ + 2y_ + x_ * (1 - x_) * (1 + t_)) <= tol
    end

    # Constant Dirichlet data, a time-dependent source (the load is a function of t) and an
    # initial field that is not an exact solution.
    Ω = MeshDomain(rectangle_grid(Triangle, (6, 6)))
    Γl, Γr, Γb, Γt = pieces(Ω)
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + sin(3x + t) * (1 + y)
    top(x_) = 1 + x_ + x_ * (1 - x_)
    bcs = [[u(t, Γ) ~ 1 + x for Γ in (Γl, Γr, Γb)]; u(t, Γt) ~ top(x); u(0, x, y) ~ 1 + x + x * (1 - x) * y^2]
    solve_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10, saveat = 0.1)
    alg_sol, alg_prob = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; solve_kwargs)
    sol, prob = eliminated([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; solve_kwargs)

    @testset "agrees with :algebraic" begin
        @test SciMLBase.successful_retcode(sol)
        @test sol.t ≈ alg_sol.t
        @test maximum(abs, sol[u(t, x, y)] - alg_sol[u(t, x, y)]) <= 1.0e-7
    end

    @testset "reduced problem" begin
        space = sol.disc_data.space
        free = setdiff(1:length(alg_prob.u0), prescribed_dofs(space))
        @test length(prob.u0) == length(free) == size(prob.f.jac_prototype, 1)
        @test prob.u0 == alg_prob.u0[free]
        @test all(prob.differential_vars)
        res = similar(prob.u0)
        prob.f(res, prob.du0, prob.u0, prob.p, 0.0)
        @test norm(res, Inf) <= 1.0e-10
        @test length(sol.original_sol[only(sol.disc_data.fields).unknown][end]) == length(free)
    end

    @testset "solution interface returns full fields" begin
        U = sol[u(t, x, y)]
        @test size(U) == size(alg_sol[u(t, x, y)])
        for (Γ, g) in ((Γl, x_ -> 1 + x_), (Γt, top))
            nodes = piece_nodes(sol, Γ)
            expected = [g(sol[x][i]) for _ in sol.t, i in nodes]
            @test sol[u(t, Γ)] ≈ expected atol = 1.0e-14
            @test U[:, nodes] ≈ expected atol = 1.0e-14
            @test sol(0.35, Γ; dv = u(t, x, y)) ≈ expected[1, :] atol = 1.0e-14
        end
        @test sol(0.35, 0.0, 0.5; dv = u(t, x, y)) ≈ 1.0 atol = 1.0e-14
        @test sol(0.35, 0.3, 0.6; dv = u(t, x, y)) ≈ alg_sol(0.35, 0.3, 0.6; dv = u(t, x, y)) atol = 1.0e-6
        @test normal_flux(sol, u(t, x, y), Γr) ≈ normal_flux(alg_sol, u(t, x, y), Γr) atol = 1.0e-6
        @test sol[Dx(u(t, x, y))] ≈ alg_sol[Dx(u(t, x, y))] atol = 1.0e-6
        mktempdir() do dir
            pvd = write_vtk(sol, joinpath(dir, "heat"))
            @test isfile(pvd)
            @test count(endswith(".vtu"), readdir(dir)) == length(sol.t)
        end
    end

    @testset "algebraic field: structurally zero mass rows stay algebraic" begin
        # u_t = Δu + v with 0 = Δv: v* = 1 + x is exact in P1, and u is compared with :algebraic.
        parabolic = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + v(t, x, y)
        algebraic = 0 ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))
        Ω2 = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        bcs2 = [
            [u(t, Γ) ~ 1 + x + 2y for Γ in pieces(Ω2)]; [v(t, Γ) ~ 1 + x for Γ in pieces(Ω2)];
            u(0, x, y) ~ (1 + x + 2y) + x * (1 - x) * y * (1 - y)
        ]
        sol2, prob2 = eliminated(
            [parabolic, algebraic], bcs2, Ω2, (x, y), t, [u(t, x, y), v(t, x, y)];
            pairing = Dict(algebraic => v), tspan = (0.0, 5.0), solve_kwargs
        )
        alg2, _ = fem_solve(
            [parabolic, algebraic], bcs2, Ω2, (x, y), t, [u(t, x, y), v(t, x, y)];
            pairing = Dict(algebraic => v), tspan = (0.0, 5.0), solve_kwargs
        )
        @test SciMLBase.successful_retcode(sol2)
        space = sol2.disc_data.space
        nfree(name) = count(!in(prescribed_dofs(space)), field_range(space, name))
        @test prob2.differential_vars == [trues(nfree(:u)); falses(nfree(:v))]
        @test nodal_error(sol2, v(t, x, y), (t_, x_, y_) -> 1 + x_) <= tol
        @test maximum(abs, sol2[u(t, x, y)] - alg2[u(t, x, y)]) <= 1.0e-7
    end

    @testset "time-dependent Dirichlet data are rejected" begin
        td = [u(t, Γl) ~ 1 + x; u(t, Γr) ~ (2 + x) * (1 + t); [u(t, Γ) ~ 1 + x for Γ in (Γb, Γt)]; u(0, x, y) ~ 1 + x]
        @named pdesys = PDESystem([eq], td, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        @test_throws ArgumentError(
            "dirichlet = :eliminate needs Dirichlet data that are constant in time, but the condition on " *
                "boundary piece right depends on time; use the default dirichlet = :algebraic"
        ) discretize(pdesys, FEMDiscretization(Ω, t; dirichlet = :eliminate))
        @test discretize(pdesys, FEMDiscretization(Ω, t)) isa SciMLBase.DAEProblem
    end
end
