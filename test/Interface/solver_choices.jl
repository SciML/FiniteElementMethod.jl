# One problem solved with each DAE solver that the test environment offers. `discretize` returns a
# DAEProblem, so the choices are the implicit-form BDF methods of OrdinaryDiffEqBDF (DFBDF, DABDF2,
# DImplicitEuler); ODE solvers are refused by `solve`. Rodas5P and IDA are not in the test environment.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x u(..)
Dt, Dx = Differential(t), Differential(x)

# u_t = u_xx on 16 line cells in P2 elements, t in (0, 0.1), with Dirichlet data `g(t, x)` at both ends
# and initial condition `initial(x)`.
function heat_problem(g, initial)
    Ω = MeshDomain(generate_grid(Line, (16,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    eqs = [Dt(u(t, x)) ~ Dx(Dx(u(t, x)))]
    bcs = [u(t, left) ~ g(t, 0), u(t, right) ~ g(t, 1), u(0, x) ~ initial(x)]
    @named pdesys = PDESystem(eqs, bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x], [u(t, x)], [])
    return discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, 2}())))
end
eigenmode_problem() = heat_problem((t_, x_) -> 1, x_ -> 1 + sin(pi * x_))
eigenmode(x_) = 1 + exp(-pi^2 * 0.1) * sin(pi * x_)
parabola_problem() = heat_problem((t_, x_) -> x_^2 + 2t_, x_ -> x_^2)

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), 16 line cells
# BCs:    u(t, 0) = u(t, 1) = 1 (constant Dirichlet); IC u(0, x) = 1 + sin(pi x)
# Exact:  u = 1 + exp(-pi^2 t) sin(pi x)
# Shows:  every DAE solver reaches the same spatial error (set by the mesh, not the solver) from one problem.
@testset "the DAE solver $name solves the heat equation to the spatial error of the mesh" for (name, alg) in (
        ("DFBDF", DFBDF()), ("DABDF2", DABDF2()), ("DImplicitEuler", DImplicitEuler()),
        ("DFBDF of order 2", DFBDF(max_order = Val(2))),
    )
    sol = solve(eigenmode_problem(), alg; abstol = 1.0e-10, reltol = 1.0e-10)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test l2_error(sol, u(t, x), eigenmode, 0.1, sample_points(100)) < 2.0e-4
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), 16 line cells
# BCs:    u(t, 0) = u(t, 1) = 1 (constant Dirichlet); IC u(0, x) = 1 + sin(pi x)
# Exact:  u = 1 + exp(-pi^2 t) sin(pi x)
# Shows:  tolerances are ordinary solve keywords: tightening them lowers the error down to the spatial error.
@testset "looser solver tolerances give a larger error than tight ones" begin
    error_at(tol) = l2_error(solve(eigenmode_problem(), DFBDF(); abstol = tol, reltol = tol), u(t, x), eigenmode, 0.1, sample_points(100))
    @test error_at(1.0e-3) > error_at(1.0e-10)
    @test error_at(1.0e-10) < 2.0e-4
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), 16 line cells
# BCs:    u(t, 0) = 2t, u(t, 1) = 1 + 2t (time-dependent Dirichlet); IC u(0, x) = x^2
# Exact:  u = x^2 + 2t
# Shows:  `saveat` chooses the output times and the solution holds the Dirichlet data at both ends at each of them.
@testset "saveat controls the stored times and the boundary value is held at each" begin
    times = collect(0.0:0.025:0.1)
    sol = solve(parabola_problem(), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = times)
    @test sol.t ≈ times
    @test all(abs(sol(tk, 0.0; dv = u(t, x)) - 2tk) < 1.0e-6 for tk in times)
    @test all(abs(sol(tk, 1.0; dv = u(t, x)) - (1 + 2tk)) < 1.0e-6 for tk in times)
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), 16 line cells
# BCs:    u(t, 0) = 2t, u(t, 1) = 1 + 2t (time-dependent Dirichlet)
# Exact:  none; an ODE solver is not valid for the DAEProblem
# Shows:  an ODE solver such as FBDF is refused with SciMLBase's pairing message, not a wrong answer.
@testset "an ODE solver is refused for the DAE problem" begin
    @test_throws "Incompatible problem+solver pairing" solve(parabola_problem(), FBDF())
end
