# The 2D heat equation through the full interface on a quadrilateral mesh and on a triangle mesh,
# in P1 and P2 elements, with constant and time-dependent Dirichlet data, compared with exact
# solutions. Every case builds the PDESystem, the FEMDiscretization and the problem, solves it, and
# measures the L2 distance to the exact solution at the final time.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))

unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

# Solves `heat` on Ω with the given element space and boundary and initial conditions to `tend`.
function solve_heat(Ω, space, bcs, tend)
    domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)]
    @named pdesys = PDESystem([heat], bcs, domains, [t, x, y], [u(t, x, y)], [])
    prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => space)))
    return solve(prob, DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), 16 x 16 cells
# BCs:    u = 0 on all four sides (constant Dirichlet); IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-2 pi^2 t) sin(pi x) sin(pi y)
# Shows:  the decaying eigenmode is reproduced by P1 and P2 elements on triangles and quadrilaterals.
@testset "the heat equation with zero Dirichlet data decays as its eigenmode on $name meshes" for (name, celltype, ref, order, tol) in (
        ("P1 triangle", Triangle, RefTriangle, 1, 5.0e-3), ("P2 triangle", Triangle, RefTriangle, 2, 1.0e-4),
        ("P1 quadrilateral", Quadrilateral, RefQuadrilateral, 1, 5.0e-3), ("P2 quadrilateral", Quadrilateral, RefQuadrilateral, 2, 1.0e-4),
    )
    Ω = unit_square(celltype, 16)
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ sin(pi * x) * sin(pi * y)]
    sol = solve_heat(Ω, Lagrange{ref, order}(), bcs, 0.05)
    exact(x_, y_) = exp(-2pi^2 * 0.05) * sin(pi * x_) * sin(pi * y_)
    @test l2_error(sol, u(t, x, y), exact, 0.05, sample_points(50), sample_points(50)) < tol
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 1), 4 x 4 cells
# BCs:    u = x^2 + y^2 + 4t on all four sides (time-dependent Dirichlet); IC u(0, x, y) = x^2 + y^2
# Exact:  u = x^2 + y^2 + 4t, which lies in the P2 space, so the error is the solver tolerance
# Shows:  time-dependent Dirichlet data is imposed at every time, on both cell shapes.
@testset "time-dependent Dirichlet data is reproduced exactly by P2 elements on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    Ω = unit_square(celltype, 4)
    bcs = [[u(t, Γ) ~ x^2 + y^2 + 4t for Γ in sides(Ω)]; u(0, x, y) ~ x^2 + y^2]
    sol = solve_heat(Ω, Lagrange{ref, 2}(), bcs, 1.0)
    @test l2_error(sol, u(t, x, y), (x_, y_) -> x_^2 + y_^2 + 4, 1.0, sample_points(20), sample_points(20)) < 1.0e-7
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.5), meshes of 4, 8 and 16 triangle pairs per side
# BCs:    u = exp(2t + x + y) on all four sides (time-dependent, not in the element space)
# Exact:  u = exp(2t + x + y)
# Shows:  P1 elements converge at the optimal L2 rate 2 under time-dependent Dirichlet data.
@testset "P1 elements converge at rate 2 under time-dependent Dirichlet data on triangles" begin
    ns = (4, 8, 16)
    errors = map(ns) do n
        Ω = unit_square(Triangle, n)
        bcs = [[u(t, Γ) ~ exp(2t + x + y) for Γ in sides(Ω)]; u(0, x, y) ~ exp(x + y)]
        sol = solve_heat(Ω, Lagrange{RefTriangle, 1}(), bcs, 0.5)
        l2_error(sol, u(t, x, y), (x_, y_) -> exp(1 + x_ + y_), 0.5, sample_points(60), sample_points(60))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 1.9)
end
