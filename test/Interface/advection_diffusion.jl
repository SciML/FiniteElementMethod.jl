# Advection-diffusion with a constant velocity through the full interface, in 1D and 2D with P2 elements,
# against manufactured solutions: a source term f is chosen so that a known u solves the equation.
# docs/src/equation_classes.md lists convection-diffusion at moderate cell Peclet number as a Tier 1 class
# (supported, VERIFIED); only pure advection and advection-dominated problems are Tier 2 (they need SUPG, see `supg.jl`),
# so every case here keeps the cell Peclet number below one.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

line(n) = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))

# Solves the 1D problem eq on the line Ω with Dirichlet data g(t, x) at both ends and the initial condition g(0, x).
function solve_1d(eq, Ω, g, tend)
    left, right = boundarypiece.(Ω, (:left, :right))
    bcs = [u(t, left) ~ g(t, 0), u(t, right) ~ g(t, 1), u(0, x) ~ g(0, x)]
    @named pdesys = PDESystem([eq], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x], [u(t, x)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, 2}()))
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# Same in 2D on the unit square with Dirichlet data g(t, x, y) on all four sides.
function solve_2d(eq, Ω, ref, g, tend)
    bcs = [[u(t, Γ) ~ g(t, x, y) for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ g(0, x, y)]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, 2}()))
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t + c u_x = D u_xx + f, with c = 2, D = 0.5
# Domain: x in (0, 1), t in (0, 1), 4 line cells
# BCs:    u = g(t, x) at both ends (time-dependent Dirichlet); IC u(0, x) = g(0, x)
# Exact:  u = x^2 + 3x t + t, so f = u_t + c u_x - D u_xx = 3x + 1 + 2 (2x + 3t) - 1, which lies in the P2 space
# Shows:  the advection term has the right sign and size: the P2 solution is exact to the solver tolerance.
@testset "constant-velocity advection-diffusion is reproduced exactly by P2 elements on a line" begin
    c, D = 2.0, 0.5
    exact(t_, x_) = x_^2 + 3x_ * t_ + t_
    source = 3x + 1 + c * (2x + 3t) - D * 2
    sol = solve_1d(Dt(u(t, x)) + c * Dx(u(t, x)) ~ D * Dx(Dx(u(t, x))) + source, line(4), exact, 1.0)
    @test l2_error(sol, u(t, x), x_ -> exact(1.0, x_), 1.0, sample_points(40)) < 1.0e-7
end

# PDE:    u_t + c u_x = D u_xx + f, with c = 5, D = 1 (cell Peclet number c h / 2D is 0.63 on the coarsest mesh and 0.16 on the finest)
# Domain: x in (0, 1), t in (0, 0.2), meshes of 4, 8 and 16 line cells
# BCs:    u = 0 at both ends; IC u(0, x) = sin(pi x)
# Exact:  u = exp(-t) sin(pi x), so f = (D pi^2 - 1) u + c pi exp(-t) cos(pi x)
# Shows:  P2 elements converge at the optimal L2 rate 3 with a source term and a first-order term.
@testset "P2 elements on a line converge at rate 3 for advection-diffusion with a manufactured source" begin
    c, D = 5.0, 1.0
    source = exp(-t) * ((D * pi^2 - 1) * sin(pi * x) + c * pi * cos(pi * x))
    ns = (4, 8, 16)
    errors = map(ns) do n
        sol = solve_1d(Dt(u(t, x)) + c * Dx(u(t, x)) ~ D * Dx(Dx(u(t, x))) + source, line(n), (t_, x_) -> exp(-t_) * sin(pi * x_), 0.2)
        l2_error(sol, u(t, x), x_ -> exp(-0.2) * sin(pi * x_), 0.2, sample_points(200))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end

# PDE:    u_t + c . grad u = D (u_xx + u_yy) + f, with c = (1, -2), D = 0.5
# Domain: the unit square, t in (0, 1), 4 x 4 cells
# BCs:    u = g(t, x, y) on all four sides (time-dependent Dirichlet); IC u(0, x, y) = g(0, x, y)
# Exact:  u = x^2 + x y + 2 y^2 + t (1 + x - y), so f = u_t + c . grad u - D Laplace u, in the P2 space
# Shows:  a constant velocity vector acts in both directions, on triangles and quadrilaterals.
@testset "constant-velocity advection-diffusion is reproduced exactly by P2 elements on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    cx, cy, D = 1.0, -2.0, 0.5
    exact(t_, x_, y_) = x_^2 + x_ * y_ + 2y_^2 + t_ * (1 + x_ - y_)
    u_x, u_y = 2x + y + t, x + 4y - t
    source = (1 + x - y) + cx * u_x + cy * u_y - D * (2 + 4)
    eq = Dt(u(t, x, y)) + cx * Dx(u(t, x, y)) + cy * Dy(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) + source
    sol = solve_2d(eq, unit_square(celltype, 4), ref, exact, 1.0)
    @test l2_error(sol, u(t, x, y), (x_, y_) -> exact(1.0, x_, y_), 1.0, sample_points(20), sample_points(20)) < 1.0e-7
end

# PDE:    u_t + c . grad u = D (u_xx + u_yy) + f, with c = (3, 2), D = 1
# Domain: the unit square, t in (0, 0.1), meshes of 2, 4 and 8 per side
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-t) sin(pi x) sin(pi y), so f = (2 D pi^2 - 1) u + c_x pi exp(-t) cos(pi x) sin(pi y) + c_y pi exp(-t) sin(pi x) cos(pi y)
# Shows:  P2 elements converge at the optimal L2 rate 3 in 2D with advection and a manufactured source.
@testset "P2 elements on triangles converge at rate 3 for 2D advection-diffusion with a manufactured source" begin
    cx, cy, D = 3.0, 2.0, 1.0
    s = exp(-t)
    source = s * ((2D * pi^2 - 1) * sin(pi * x) * sin(pi * y) + cx * pi * cos(pi * x) * sin(pi * y) + cy * pi * sin(pi * x) * cos(pi * y))
    eq = Dt(u(t, x, y)) + cx * Dx(u(t, x, y)) + cy * Dy(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) + source
    ns = (2, 4, 8)
    errors = map(ns) do n
        sol = solve_2d(eq, unit_square(Triangle, n), RefTriangle, (t_, x_, y_) -> exp(-t_) * sin(pi * x_) * sin(pi * y_), 0.1)
        l2_error(sol, u(t, x, y), (x_, y_) -> exp(-0.1) * sin(pi * x_) * sin(pi * y_), 0.1, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.7)
end
