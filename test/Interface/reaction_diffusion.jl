# Reaction-diffusion through the full interface: a linear reaction term, a linearly coupled two-field
# system, and the documented rejection of a nonlinear reaction.
# docs/src/equation_classes.md lists second-order parabolic equations "scalar or coupled multi-field" as
# Tier 1 (VERIFIED, two-field reaction-diffusion to 1.9e-16). It also lists nonlinear sources `f(u)` as
# Tier 1 in the design, but states that this release rejects state-dependent coefficients with an error
# naming milestone M3, so the nonlinear case below asserts that rejection.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..) v(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

# Solves `eq` for u on the unit square with zero Dirichlet data and the initial condition u0.
function solve_square(eq, Ω, ref, order, u0, tend)
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ u0]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, order}()))
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = u_xx + u_yy - k u + f, with k = 4 (a decay reaction)
# Domain: the unit square, t in (0, 0.1), 16 x 16 cells
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-t) sin(pi x) sin(pi y), so f = (2 pi^2 + k - 1) u
# Shows:  a linear reaction term and a source term together, on triangles and quadrilaterals.
@testset "linear reaction-diffusion with a manufactured source matches the exact solution on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    k = 4.0
    exact(t_, x_, y_) = exp(-t_) * sin(pi * x_) * sin(pi * y_)
    source = (2pi^2 + k - 1) * exp(-t) * sin(pi * x) * sin(pi * y)
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) - k * u(t, x, y) + source
    sol = solve_square(eq, unit_square(celltype, 16), ref, 2, sin(pi * x) * sin(pi * y), 0.1)
    @test l2_error(sol, u(t, x, y), (x_, y_) -> exact(0.1, x_, y_), 0.1, sample_points(40), sample_points(40)) < 1.0e-4
end

# PDE:    u_t = u_xx - k u, with k = 3 (no source)
# Domain: x in (0, 1), t in (0, 0.2), meshes of 2, 4, 8 and 16 line cells
# BCs:    u(t, 0) = u(t, 1) = 0; IC u(0, x) = sin(pi x)
# Exact:  u = exp(-(pi^2 + k) t) sin(pi x): the reaction only speeds up the decay of the heat eigenmode
# Shows:  P2 elements converge at rate 3 and the reaction coefficient enters the decay rate.
@testset "P2 elements on a line converge at rate 3 for the decay-reaction heat equation" begin
    k, tend = 3.0, 0.2
    ns = (2, 4, 8, 16)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
        left, right = boundarypiece.(Ω, (:left, :right))
        bcs = [u(t, left) ~ 0, u(t, right) ~ 0, u(0, x) ~ sin(pi * x)]
        @named pdesys = PDESystem([Dt(u(t, x)) ~ Dx(Dx(u(t, x))) - k * u(t, x)], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x], [u(t, x)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, 2}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
        l2_error(sol, u(t, x), x_ -> exp(-(pi^2 + k) * tend) * sin(pi * x_), tend, sample_points(200))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end

# PDE:    u_t = u_xx + a (v - u) + f_u,   v_t = D v_xx - a (v - u) + f_v, with a = 2, D = 0.5
#         (two fields coupled linearly through the reaction)
# Domain: x in (0, 1), t in (0, 1), 4 line cells
# BCs:    u = v = g(t, x) at both ends (time-dependent Dirichlet); IC from the exact solution
# Exact:  u = x^2 + t, v = 1 + x^2 - t x^2, both in the P2 space, so f_u and f_v follow from the equations
# Shows:  coupled multi-field reaction-diffusion with a source in each field is solved exactly.
@testset "two linearly coupled fields are reproduced exactly by P2 elements" begin
    a, D = 2.0, 0.5
    u_exact(t_, x_) = x_^2 + t_
    v_exact(t_, x_) = 1 + x_^2 - t_ * x_^2
    Ω = MeshDomain(generate_grid(Line, (4,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    ue, ve = x^2 + t, 1 + x^2 - t * x^2
    f_u = 1 - 2 - a * (ve - ue)
    f_v = -x^2 - D * (2 - 2t) + a * (ve - ue)
    eqs = [
        Dt(u(t, x)) ~ Dx(Dx(u(t, x))) + a * (v(t, x) - u(t, x)) + f_u,
        Dt(v(t, x)) ~ D * Dx(Dx(v(t, x))) - a * (v(t, x) - u(t, x)) + f_v,
    ]
    bcs = [
        u(t, left) ~ u_exact(t, 0), u(t, right) ~ u_exact(t, 1), u(0, x) ~ u_exact(0, x),
        v(t, left) ~ v_exact(t, 0), v(t, right) ~ v_exact(t, 1), v(0, x) ~ v_exact(0, x),
    ]
    @named pdesys = PDESystem(eqs, bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x], [u(t, x), v(t, x)], [])
    P2 = Lagrange{RefLine, 2}()
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => P2, v => P2))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test l2_error(sol, u(t, x), x_ -> u_exact(1.0, x_), 1.0, sample_points(40)) < 1.0e-7
    @test l2_error(sol, v(t, x), x_ -> v_exact(1.0, x_), 1.0, sample_points(40)) < 1.0e-7
end

# PDE:    u_t = u_xx + u_yy + u (1 - u) (a nonlinear, logistic reaction)
# Shows:  state-dependent sources are named in the error rather than silently linearised (M3).
@testset "a nonlinear reaction term is rejected naming milestone M3" begin
    Ω = unit_square(Triangle, 2)
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + u(t, x, y) * (1 - u(t, x, y))
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ 0]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    @test_throws "nonlinear terms land in M3" discretize(pdesys, FEMDiscretization(Ω, t))
end
