# The same smooth problem solved with elements of increasing polynomial order, showing the observed
# L2 convergence rate equal to order + 1 on a refinement sequence. P1 and P2 are available in 1D and
# 2D; P3 is available on triangles and quadrilaterals (Ferrite has no cubic Lagrange line).
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# L2 error at t = 0.1 of the 1D eigenmode on `n` line cells.
function eigenmode_error_1d(n, order)
    Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    bcs = [u(t, left) ~ 0, u(t, right) ~ 0, u(0, x) ~ sin(pi * x)]
    @named pdesys = PDESystem([Dt(u(t, x)) ~ Dx(Dx(u(t, x)))], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x], [u(t, x)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, order}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    return l2_error(sol, u(t, x), x_ -> exp(-pi^2 * 0.1) * sin(pi * x_), 0.1, sample_points(200))
end

# L2 error at t = 0.05 of the 2D eigenmode on an `n` x `n` mesh of `celltype`.
function eigenmode_error_2d(celltype, ref, n, order)
    Ω = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    bcs = [[u(t, Γ) ~ 0 for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ sin(pi * x) * sin(pi * y)]
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, order}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    exact(x_, y_) = exp(-2pi^2 * 0.05) * sin(pi * x_) * sin(pi * y_)
    return l2_error(sol, u(t, x, y), exact, 0.05, sample_points(40), sample_points(40))
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), meshes of 2, 4, 8 and 16 line cells
# BCs:    u(t, 0) = u(t, 1) = 0 (constant Dirichlet); IC u(0, x) = sin(pi x)
# Exact:  u = exp(-pi^2 t) sin(pi x)
# Shows:  on a line the L2 rate is order + 1: about 2 for P1 and about 3 for P2.
@testset "$name elements on a line converge at rate $(order + 1)" for (name, order) in (("P1", 1), ("P2", 2))
    ns = (2, 4, 8, 16)
    errors = [eigenmode_error_1d(n, order) for n in ns]
    @test issorted(errors; rev = true)
    @test all(observed_rate(errors, 1 ./ ns)[2:end] .> order + 0.85)
end

shapes = (("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral))
orders = (("P1", 1), ("P2", 2), ("P3", 3))

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), meshes of 2, 4 and 8 cells per side
# BCs:    u = 0 on all four sides (constant Dirichlet); IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-2 pi^2 t) sin(pi x) sin(pi y)
# Shows:  on triangles and quadrilaterals the L2 rate is order + 1 for P1, P2 and P3; the finest
#         pair of meshes is in the asymptotic range, so only its rate is checked.
@testset "$name elements converge at rate $(order + 1) on $shape meshes" for (shape, celltype, ref) in shapes, (name, order) in orders
    ns = order == 1 ? (4, 8, 16) : (2, 4, 8)
    errors = [eigenmode_error_2d(celltype, ref, n, order) for n in ns]
    @test issorted(errors; rev = true)
    @test last(observed_rate(errors, 1 ./ ns)) > order + 0.85
end
