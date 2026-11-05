# The heat equation with a manufactured source term f(t, x, ...): the source is chosen so that a
# non-trivial exact solution (oscillating in time, non-zero on the boundary) solves the equation, and the
# exact solution is recovered from the PDESystem through the full interface.
# docs/src/equation_classes.md, Tier 1: second-order parabolic equations, with time-dependent Dirichlet data
# (VERIFIED); sources that depend on (t, x) only are plain right-hand-side terms, not state-dependent ones.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# PDE:    u_t = u_xx + f
# Domain: x in (0, 1), t in (0, 1), meshes of 4, 8 and 16 line cells
# BCs:    u = g(t, x) at both ends (time-dependent Dirichlet, from the exact solution); IC u(0, x) = g(0, x)
# Exact:  u = sin(3t) cos(pi x), so f = 3 cos(3t) cos(pi x) + pi^2 sin(3t) cos(pi x)
# Shows:  the solution is not zero at the boundary and changes sign in time, yet P1 and P2 recover it at rates 2 and 3.
@testset "the 1D heat equation with a manufactured source recovers the exact solution at the optimal rate for $name" for (name, order, rate) in (
        ("P1", 1, 1.9), ("P2", 2, 2.8),
    )
    ns = (4, 8, 16)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
        left, right = boundarypiece.(Ω, (:left, :right))
        source = 3cos(3t) * cos(pi * x) + pi^2 * sin(3t) * cos(pi * x)
        bcs = [u(t, left) ~ sin(3t), u(t, right) ~ -sin(3t), u(0, x) ~ 0]
        @named pdesys = PDESystem([Dt(u(t, x)) ~ Dx(Dx(u(t, x))) + source], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, order}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
        l2_error(sol, u(t, x), x_ -> sin(3) * cos(pi * x_), 1.0, sample_points(200))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> rate)
end

# PDE:    u_t = u_xx + u_yy + f
# Domain: the unit square, t in (0, 1), meshes of 2, 4 and 8 cells per side
# BCs:    u = g(t, x, y) on all four sides (time-dependent Dirichlet, from the exact solution); IC u(0, x, y) = g(0, x, y)
# Exact:  u = cos(t) exp(x + y), so f = -sin(t) exp(x + y) - 2 cos(t) exp(x + y)
# Shows:  P2 elements recover a solution that is in no element space at the optimal rate 3, on both cell shapes.
@testset "the 2D heat equation with a manufactured source converges at rate 3 on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    ns = (2, 4, 8)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
        source = (-sin(t) - 2cos(t)) * exp(x + y)
        bcs = [[u(t, Γ) ~ cos(t) * exp(x + y) for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ exp(x + y)]
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + source
        @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, 2}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
        l2_error(sol, u(t, x, y), (x_, y_) -> cos(1.0) * exp(x_ + y_), 1.0, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.7)
end

# PDE:    u_t = u_xx + u_yy + f
# Domain: the unit square, t in (0, 1), 4 x 4 triangle cells
# BCs:    u = t^2 (1 + x^2 + y^2) on all four sides; IC u(0, x, y) = 0
# Exact:  u = t^2 (1 + x^2 + y^2), so f = 2t (1 + x^2 + y^2) - 4 t^2, in the P2 space for every t
# Shows:  a source that is quadratic in time and space is reproduced to the solver tolerance.
@testset "a polynomial manufactured solution is reproduced exactly by P2 elements" begin
    Ω = MeshDomain(generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    source = 2t * (1 + x^2 + y^2) - 4t^2
    bcs = [[u(t, Γ) ~ t^2 * (1 + x^2 + y^2) for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ 0]
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + source
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test l2_error(sol, u(t, x, y), (x_, y_) -> 1 + x_^2 + y_^2, 1.0, sample_points(20), sample_points(20)) < 1.0e-7
end
