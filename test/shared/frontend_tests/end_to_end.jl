# End-to-end tests, one per M2 code path: PDESystem -> DAEProblem -> solve, checked against an exact
# solution. Each manufactured solution lies in the finite-element space and the quadrature is exact
# for its data, so the nodal error is at the level of the time-integration tolerance.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Triangle, Quadrilateral, Line

include(joinpath(@__DIR__, "..", "frontend.jl"))
include(joinpath(@__DIR__, "..", "lshape.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, @register_symbolic

smooth_coef(x) = 1 + x^2
@register_symbolic smooth_coef(x)
Symbolics.@register_derivative smooth_coef(x) 1 2x

@variables x y u(..) v(..) w(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const tol = 1.0e-8

# u* = (1 + x + 2y)(1 + t) solves u_t = Δu + (1 + x + 2y) and is in every P1 and Q1 space.
uexact(t_, x_, y_) = (1 + x_ + 2y_) * (1 + t_)
heat_eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x + 2y)
ic = u(0, x, y) ~ 1 + x + 2y

pieces(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
dirichlet_all(Ω) = [u(t, Γ) ~ (1 + x + 2y) * (1 + t) for Γ in pieces(Ω)]

@testset verbose = true "end to end" begin
    @testset "constant Dirichlet" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        steady = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
        bcs = [[u(t, Γ) ~ 1 + x + 2y for Γ in pieces(Ω)]; ic]
        sol, _ = fem_solve([steady], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> 1 + x_ + 2y_) <= tol
    end

    @testset "time-dependent Dirichlet, P1 triangles" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        sol, prob = fem_solve([heat_eq], [dirichlet_all(Ω); ic], Ω, (x, y), t, [u(t, x, y)])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
        space = sol.disc_data.space
        @test !any(prob.differential_vars[space.ch.prescribed_dofs])
        # Not index 2: the default algebraic_abstol = :auto sets no abstol.
        @test !haskey(prob.kwargs, :abstol)
        # An explicit algebraic_abstol applies to any system: here to the Dirichlet rows, which stay exact.
        sol, prob = fem_solve(
            [heat_eq], [dirichlet_all(Ω); ic], Ω, (x, y), t, [u(t, x, y)]; algebraic_abstol = 1.0e3,
            solve_kwargs = (; reltol = 1.0e-10), abstol = 1.0e-10
        )
        @test prob.kwargs[:abstol] == [d ? 1.0e-10 : 1.0e3 for d in prob.differential_vars]
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
        # The symbolic Dirichlet data agree with the ConstraintHandler, including at corners
        # shared by two pieces, where the condition added last wins.
        load = SciMLBase.SymbolicIndexingInterface.getp(prob.f.sys, prob.f.sys.Fop)(prob.p)
        np = length(space.ch.prescribed_dofs)
        @test FiniteElementMethod.dirichlet_values!(zeros(np), load, 0.3) ≈
            FiniteElementMethod.dirichlet_values!(zeros(np), space, 0.3) atol = 1.0e-14
    end

    @testset "discretize returns the DAEProblem and forwards keywords" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))
        @named pdesys = PDESystem([heat_eq], [dirichlet_all(Ω); ic], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        prob = discretize(pdesys, FEMDiscretization(Ω, t; saveat = 0.25); tstops = [0.5])
        @test prob isa SciMLBase.DAEProblem
        @test prob.kwargs[:saveat] == 0.25 && prob.kwargs[:tstops] == [0.5]
        sol = solve(prob, EXACT_ROWS)
        @test sol.t == 0:0.25:1
        @test sol.disc_data isa FEMMetadata
        @test discretize(pdesys, FEMDiscretization(Ω, t; saveat = 0.25); saveat = 0.5).kwargs[:saveat] == 0.5
        reserved = "keyword `name` is set by the FEM discretization and cannot be passed; the reserved keywords are iv, unknowns, du0, coordinate, name, parameters"
        @test_throws ArgumentError(reserved) FEMDiscretization(Ω, t; name = :other)
        @test_throws ArgumentError(replace(reserved, "`name`" => "`du0`")) discretize(pdesys, FEMDiscretization(Ω, t); du0 = zeros(3))
        @test sprint(show, sol.disc_data) == sprint(show, prob.problem_type) == "FEMMetadata(fields = [u], ndofs = 9, hasTime = true)"
        @test sprint(show, FEMDiscretization(Ω, t)) == "FEMDiscretization(8 cells, time = t, codegen = :numeric)"
        sys, tspan = SciMLBase.symbolic_discretize(pdesys, FEMDiscretization(Ω, t))
        @test tspan == (0.0, 1.0)
        @test sys isa ModelingToolkit.System
    end

    @testset "PDESystem without parameters" begin
        # Leaving out `ps` gives the system `NullParameters()`, which means no parameters.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)]
        @named pdesys = PDESystem([heat_eq], [dirichlet_all(Ω); ic], domains, [t, x, y], [u(t, x, y)])
        @test pdesys.ps isa SciMLBase.NullParameters
        sol = solve(discretize(pdesys, FEMDiscretization(Ω, t)), EXACT_ROWS; abstol = 1.0e-10, reltol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
    end

    @testset "P1 quadrilaterals" begin
        Ω = MeshDomain(rectangle_grid(Quadrilateral, (3, 4)))
        sol, _ = fem_solve([heat_eq], [dirichlet_all(Ω); ic], Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
    end

    @testset "P2 triangles" begin
        # q = 1 + x² + 2y² + xy has Δq = 6.
        q(x_, y_) = 1 + x_^2 + 2y_^2 + x_ * y_
        Ω = MeshDomain(rectangle_grid(Triangle, (3, 3)))
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + q(x, y) - 6 * (1 + t)
        bcs = [[u(t, Γ) ~ q(x, y) * (1 + t) for Γ in pieces(Ω)]; u(0, x, y) ~ q(x, y)]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> q(x_, y_) * (1 + t_)) <= tol
    end

    @testset "1D Line" begin
        Ω = MeshDomain(rectangle_grid(Line, (5,)))
        eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x))) + (1 + x)
        lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
        bcs = [u(t, lo) ~ 1 + t, NormalDerivative(hi)(u(t, hi)) ~ 1 + t, u(0, x) ~ 1 + x]
        sol, _ = fem_solve([eq], bcs, Ω, (x,), t, [u(t, x)])
        @test nodal_error(sol, u(t, x), (t_, x_) -> (1 + x_) * (1 + t_)) <= tol
    end

    @testset "Neumann" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        Dn(Γ) = NormalDerivative(Γ)
        bcs = [
            u(t, Γl) ~ (1 + 2y) * (1 + t),
            Dn(Γr)(u(t, Γr)) ~ 1 + t,
            Dn(Γb)(u(t, Γb)) ~ -2 * (1 + t),
            Dn(Γt)(u(t, Γt)) ~ 2 * (1 + t),
            ic,
        ]
        sol, _ = fem_solve([heat_eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
    end

    @testset "Robin with a conormal coefficient" begin
        # Equation flux k∇u; the Robin condition is written against the conormal k·Dn.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        @parameters k α
        eq = Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y))) + (1 + x + 2y)
        bcs = [
            u(t, Γl) ~ (1 + 2y) * (1 + t),
            u(t, Γb) ~ (1 + x) * (1 + t),
            u(t, Γt) ~ (3 + x) * (1 + t),
            k * NormalDerivative(Γr)(u(t, Γr)) + α * u(t, Γr) ~ (k + α * (2 + 2y)) * (1 + t),
            ic,
        ]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; ps = [k => 2.0, α => 3.0])
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
    end

    @testset "mixed Dirichlet, Neumann and Robin pieces on one mesh" begin
        Ω = MeshDomain(rectangle_grid(Quadrilateral, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        bcs = [
            u(t, Γl) ~ (1 + 2y) * (1 + t),
            NormalDerivative(Γb)(u(t, Γb)) ~ -2 * (1 + t),
            NormalDerivative(Γt)(u(t, Γt)) + u(t, Γt) ~ (5 + x) * (1 + t),
            NormalDerivative(Γr)(u(t, Γr)) ~ 1 + t,
            ic,
        ]
        sol, _ = fem_solve([heat_eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
    end

    @testset "Gmsh mesh and generated grid give the same exact solution" begin
        u_exact(t_, x_, y_) = (1 + x_) * (1 + t_)
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x)
        gmsh_Ω = MeshDomain(lshape_grid(; h = 0.5))
        Γi, Γw, Γo = boundarypiece.(gmsh_Ω, (:inlet, :wall, :outlet))
        bcs = [u(t, Γi) ~ 1 + t, u(t, Γw) ~ (1 + x) * (1 + t), NormalDerivative(Γo)(u(t, Γo)) ~ 1 + t, u(0, x, y) ~ 1 + x]
        sol, _ = fem_solve([eq], bcs, gmsh_Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), u_exact) <= tol
        box = MeshDomain(rectangle_grid(Triangle, (4, 4); hi = (2.0, 2.0)))
        Γl, Γr, Γb, Γt = pieces(box)
        bcs = [u(t, Γl) ~ 1 + t, u(t, Γb) ~ (1 + x) * (1 + t), u(t, Γt) ~ (1 + x) * (1 + t), NormalDerivative(Γr)(u(t, Γr)) ~ 1 + t, u(0, x, y) ~ 1 + x]
        sol, _ = fem_solve([eq], bcs, box, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), u_exact) <= tol
    end

    @testset "time-dependent source" begin
        # u* = (1 + x + 2y)(1 + t²): the source 2t(1 + x + 2y) is reassembled at every t.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + 2t * (1 + x + 2y)
        bcs = [[u(t, Γ) ~ (1 + x + 2y) * (1 + t^2) for Γ in pieces(Ω)]; ic]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> (1 + x_ + 2y_) * (1 + t_^2)) <= 1.0e-6
    end

    @testset "Dirichlet data defined only from the initial time on" begin
        # sqrt(t - 0.5) is undefined at t = 0, so the space must be initialised at t0 = 1.
        Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + 1 / (2 * sqrt(t - 0.5))
        bcs = [[u(t, Γ) ~ sqrt(t - 0.5) for Γ in pieces(Ω)]; u(1.0, x, y) ~ sqrt(0.5)]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; tspan = (1.0, 2.0))
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> sqrt(t_ - 0.5)) <= 1.0e-6
    end

    @testset "time-dependent load evaluations share no mutable state" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (24, 24)))
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + sin(3t + x) * (1 + y)
        @named pdesys = PDESystem([eq], [[u(t, Γ) ~ 0 for Γ in pieces(Ω)]; u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        prob = discretize(pdesys, FEMDiscretization(Ω, t))
        load = SciMLBase.SymbolicIndexingInterface.getp(prob.f.sys, prob.f.sys.Fop)(prob.p)
        n = length(prob.u0)
        times = range(0.0, 1.0; length = 32)
        serial = [FiniteElementMethod.load!(zeros(n), load, τ) for τ in times]
        concurrent = fetch.([Threads.@spawn FiniteElementMethod.load!(zeros(n), load, τ) for τ in times])
        @test concurrent == serial
        Threads.nthreads() > 1 || @info "load evaluations ran on one thread, so this test cannot see a race; run with JULIA_NUM_THREADS > 1"
    end

    @testset "algebraic row coupled to a parabolic field, listed $(algebraic_first ? "first" : "second")" for algebraic_first in (false, true)
        # u_t = Δu + v with the algebraic row 0 = Δv; u* = (1 + x + 2y)(1 + t), v* = 1 + x + 2y.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        parabolic = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + v(t, x, y)
        algebraic = 0 ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))
        bcs = [dirichlet_all(Ω); [v(t, Γ) ~ 1 + x + 2y for Γ in pieces(Ω)]; ic]
        eqs = algebraic_first ? [algebraic, parabolic] : [parabolic, algebraic]
        sol, prob = fem_solve(
            eqs, bcs, Ω, (x, y), t, [u(t, x, y), v(t, x, y)]; pairing = Dict(algebraic => v)
        )
        @test SciMLBase.successful_retcode(sol)
        space = sol.disc_data.space
        @test !any(prob.differential_vars[FiniteElementMethod.field_range(space, :v)])
        # v is algebraic but appears in its own equation (index 1): no abstol is set.
        @test !haskey(prob.kwargs, :abstol)
        @test nodal_error(sol, u(t, x, y), uexact) <= tol
        @test nodal_error(sol, v(t, x, y), (t_, x_, y_) -> 1 + x_ + 2y_) <= tol
    end

    @testset "two fields keep their identity; the Dirichlet condition listed last wins at a corner" begin
        # Declared [u, w], equations listed [w, u]; u* = (1 + x + 2y)(1 + t), w* = (2 - x + y)(1 + t).
        we(t_, x_, y_) = (2 - x_ + y_) * (1 + t_)
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        w_eq = Dt(w(t, x, y)) ~ Dx(Dx(w(t, x, y))) + Dy(Dy(w(t, x, y))) + (2 - x + y)
        u_eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x + 2y)
        # The bottom data for u differ from u* only at the corner (0, 0) it shares with the left piece.
        bump = ifelse(x < 1.0e-9, 5.0, 0.0)
        left = u(t, Γl) ~ (1 + 2y) * (1 + t)
        bottom = u(t, Γb) ~ (1 + x) * (1 + t) + bump
        all_bcs(u_bcs) = [
            u_bcs; u(t, Γr) ~ (2 + 2y) * (1 + t); u(t, Γt) ~ (3 + x) * (1 + t);
            [w(t, Γ) ~ (2 - x + y) * (1 + t) for Γ in pieces(Ω)];
            ic; w(0, x, y) ~ 2 - x + y
        ]
        # (u conditions in input order, corner bump expected)
        for (u_bcs, corner_bump) in [([left, bottom], 5), ([bottom, left], 0)]
            sol, _ = fem_solve([w_eq, u_eq], all_bcs(u_bcs), Ω, (x, y), t, [u(t, x, y), w(t, x, y)])
            @test nodal_error(sol, w(t, x, y), we) <= tol
            corner = findfirst(i -> sol[x][i] == 0 && sol[y][i] == 0, eachindex(sol[x]))
            @test sol[u(t, x, y)][:, corner] ≈ (1 .+ sol.t) .+ corner_bump atol = tol
        end
        # The same piece and field listed twice is a user error, reported by PDEBase.
        @test_throws ErrorException("boundary conditions $left on Γ_left and $left on Γ_left both fix field 1") fem_solve(
            [w_eq, u_eq], all_bcs([left, bottom, left]), Ω, (x, y), t, [u(t, x, y), w(t, x, y)]
        )
    end

    @testset "registered coefficient function in divergence form" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        # ∇·(a∇u) with a(x) = 1 + x², registered with its derivative; u* = x(1 + t) gives the source
        # x - a'(x)(1 + t). The flux is isotropic, so Dn is a conormal condition on every face. Quadrature order 2 integrates a∇u·∇v and the source exactly on P1.
        eq = Dt(u(t, x, y)) ~ Dx(smooth_coef(x) * Dx(u(t, x, y))) + Dy(smooth_coef(x) * Dy(u(t, x, y))) + x - 2x * (1 + t)
        bcs = [
            u(t, Γl) ~ 0, u(t, Γr) ~ 1 + t,
            NormalDerivative(Γb)(u(t, Γb)) ~ 0, NormalDerivative(Γt)(u(t, Γt)) ~ 0, u(0, x, y) ~ x,
        ]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> x_ * (1 + t_)) <= tol
    end
end
