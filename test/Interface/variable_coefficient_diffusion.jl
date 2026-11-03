# Diffusion with a space-dependent diffusivity a(x), in the two forms the front end accepts: divergence
# form Dx(a Dx(u)) and non-divergence form a Dxx(u), with manufactured solutions and sources.
# docs/src/equation_classes.md, Tier 1: "Second-order parabolic equations in divergence or non-divergence
# form" (VERIFIED, non-divergence `a(x,y) Δu` at rates 1.96 to 2.01). Coefficients that depend on time or on
# the solution are rejected (M3), and discontinuous coefficients across interfaces are "not evidenced",
# so a(x) here is smooth and depends on space only.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# The warning the front end gives when integration by parts of a(x) u_xx produces product-rule terms.
const warning = r"product-rule correction terms"

# Solves `eq` for u on a line of n cells with u = 0 at both ends and IC sin(pi x); returns the error at tend.
function error_1d(eq, n, tend)
    Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    bcs = [u(t, left) ~ 0, u(t, right) ~ 0, u(0, x) ~ sin(pi * x)]
    @named pdesys = PDESystem([eq], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x], [u(t, x)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, 2}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
    return l2_error(sol, u(t, x), x_ -> exp(-tend) * sin(pi * x_), tend, sample_points(200))
end

# Same on the unit square for the exact solution exp(-t) sin(pi x) sin(pi y).
function error_2d(eq, celltype, ref, n, tend)
    Ω = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    bcs = [[u(t, Γ) ~ 0 for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ sin(pi * x) * sin(pi * y)]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, 2}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    return l2_error(sol, u(t, x, y), (x_, y_) -> exp(-tend) * sin(pi * x_) * sin(pi * y_), tend, sample_points(40), sample_points(40))
end

# PDE:    u_t = (a u_x)_x + f, with a(x) = 1 + x (divergence form)
# Domain: x in (0, 1), t in (0, 0.2), meshes of 4, 8 and 16 line cells
# BCs:    u = 0 at both ends; IC u(0, x) = sin(pi x)
# Exact:  u = exp(-t) sin(pi x), so f = u_t - a_x u_x - a u_xx = exp(-t) (-sin(pi x) - pi cos(pi x) + (1 + x) pi^2 sin(pi x))
# Shows:  P2 elements converge at rate 3; the product rule on a(x) u_x is applied by the front end.
@testset "divergence-form diffusion with a(x) = 1 + x converges at rate 3 on a line" begin
    source = exp(-t) * (-sin(pi * x) - pi * cos(pi * x) + (1 + x) * pi^2 * sin(pi * x))
    ns = (4, 8, 16)
    errors = [error_1d(Dt(u(t, x)) ~ Dx((1 + x) * Dx(u(t, x))) + source, n, 0.2) for n in ns]
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end

# PDE:    u_t = a u_xx + f, with a(x) = 1 + x (non-divergence form)
# Domain: x in (0, 1), t in (0, 0.2), meshes of 4, 8 and 16 line cells
# BCs:    u = 0 at both ends; IC u(0, x) = sin(pi x)
# Exact:  u = exp(-t) sin(pi x), so f = u_t - a u_xx = exp(-t) (-1 + (1 + x) pi^2) sin(pi x)
# Shows:  P2 elements converge at rate 3; the same exact solution needs a different source than in divergence form,
#         and the front end warns that the product-rule terms make the stiffness matrix non-symmetric.
@testset "non-divergence-form diffusion with a(x) = 1 + x converges at rate 3 on a line" begin
    source = exp(-t) * (-1 + (1 + x) * pi^2) * sin(pi * x)
    ns = (4, 8, 16)
    errors = @test_logs (:warn, warning) match_mode = :any [error_1d(Dt(u(t, x)) ~ (1 + x) * Dx(Dx(u(t, x))) + source, n, 0.2) for n in ns]
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end

# PDE:    u_t = (a u_x)_x + (a u_y)_y + f, with a(x, y) = 1 + x + y (divergence form)
# Domain: the unit square, t in (0, 0.1), meshes of 2, 4 and 8 cells per side
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-t) sin(pi x) sin(pi y); with a_x = a_y = 1, f = u_t - a Laplace(u) - u_x - u_y
# Shows:  P2 elements converge at rate 3 on triangles and quadrilaterals.
@testset "divergence-form diffusion with a(x, y) = 1 + x + y converges at rate 3 on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    a = 1 + x + y
    s = exp(-t)
    source = s * (-sin(pi * x) * sin(pi * y) + 2pi^2 * a * sin(pi * x) * sin(pi * y) - pi * cos(pi * x) * sin(pi * y) - pi * sin(pi * x) * cos(pi * y))
    eq = Dt(u(t, x, y)) ~ Dx(a * Dx(u(t, x, y))) + Dy(a * Dy(u(t, x, y))) + source
    ns = (2, 4, 8)
    errors = [error_2d(eq, celltype, ref, n, 0.1) for n in ns]
    @test all(observed_rate(errors, 1 ./ ns) .> 2.7)
end

# PDE:    u_t = a (u_xx + u_yy) + f, with a(x, y) = 1 + x + y (non-divergence form)
# Domain: the unit square, t in (0, 0.1), meshes of 2, 4 and 8 cells per side
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-t) sin(pi x) sin(pi y), so f = (-1 + 2 pi^2 a) u
# Shows:  P2 elements converge at rate 3 on triangles and quadrilaterals, with the same product-rule warning.
@testset "non-divergence-form diffusion with a(x, y) = 1 + x + y converges at rate 3 on $name meshes" for (name, celltype, ref) in (
        ("triangle", Triangle, RefTriangle), ("quadrilateral", Quadrilateral, RefQuadrilateral),
    )
    a = 1 + x + y
    source = exp(-t) * (-1 + 2pi^2 * a) * sin(pi * x) * sin(pi * y)
    eq = Dt(u(t, x, y)) ~ a * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) + source
    ns = (2, 4, 8)
    errors = @test_logs (:warn, warning) match_mode = :any [error_2d(eq, celltype, ref, n, 0.1) for n in ns]
    @test all(observed_rate(errors, 1 ./ ns) .> 2.7)
end
