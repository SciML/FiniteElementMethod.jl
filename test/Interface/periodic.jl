# Periodic boundary conditions through the full interface: an equation between the traces of one field on two
# boundary pieces, `u(t, Γa) ~ u(t, Γb)` (or `u(Γa) ~ u(Γb)` without time), identifies the field on the two pieces when
# they are translates of each other. Each case builds the PDESystem as a user would, solves it and compares with an
# exact solution; the last cases are the inputs that are refused.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, NonlinearSolveFirstOrder, Test
using Ferrite: Grid, Node, generate_grid, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..) v(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))

unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

# Solves `eq` on Ω with the conditions `bcs` to `tend`; the keywords go to `FEMDiscretization`.
function solve_heat(eq, Ω, bcs, tend; kw...)
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], [])
    return solve(discretize(pdesys, FEMDiscretization(Ω, t; kw...)), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), meshes of 8, 16 and 32 cells per side
# BCs:    u(t, left) ~ u(t, right) and u(t, bottom) ~ u(t, top): periodic in both directions; IC sin(2 pi (x + y))
# Exact:  u = exp(-8 pi^2 t) sin(2 pi (x + y)), which does not vanish on the boundary
# Shows:  a fully periodic box, whose corners belong to several pairs, converges at rate 2 (P1) and 3 (P2), on both cell
#         shapes, and the two sides of the seam carry the same values.
@testset "heat equation on a fully periodic box follows its Fourier mode at the optimal rate for $name" for (name, celltype, ref, order) in (
        ("P1 triangles", Triangle, RefTriangle, 1), ("P2 quadrilaterals", Quadrilateral, RefQuadrilateral, 2),
    )
    exact(x_, y_) = exp(-8pi^2 * 0.05) * sin(2pi * (x_ + y_))
    ns = (8, 16, 32)
    errors = map(ns) do n
        Ω = unit_square(celltype, n)
        left, right, bottom, top = sides(Ω)
        bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ sin(2pi * (x + y))]
        sol = solve_heat(heat, Ω, bcs, 0.05; spaces = Dict(u => Lagrange{ref, order}()))
        edge = collect(range(0.0, 1.0; length = 7))
        @test sol(0.05, [0.0], edge; dv = u(t, x, y)) ≈ sol(0.05, [1.0], edge; dv = u(t, x, y)) atol = 1.0e-12
        @test sol(0.05, edge, [0.0]; dv = u(t, x, y)) ≈ sol(0.05, edge, [1.0]; dv = u(t, x, y)) atol = 1.0e-12
        l2_error(sol, u(t, x, y), exact, 0.05, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> order + 0.8)
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), meshes of 8, 16 and 32 triangle pairs per side
# BCs:    periodic left and right; u = 0 on bottom and top (Dirichlet); IC sin(2 pi x) sin(pi y)
# Exact:  u = exp(-5 pi^2 t) sin(2 pi x) sin(pi y)
# Shows:  a pair of periodic pieces whose corners meet Dirichlet pieces: the Dirichlet value holds at the shared
#         corners and P1 elements converge at rate 2.
@testset "heat equation periodic in x and Dirichlet in y follows its Fourier mode at rate 2" begin
    exact(x_, y_) = exp(-5pi^2 * 0.05) * sin(2pi * x_) * sin(pi * y_)
    ns = (8, 16, 32)
    errors = map(ns) do n
        Ω = unit_square(Triangle, n)
        left, right, bottom, top = sides(Ω)
        bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ sin(2pi * x) * sin(pi * y)]
        sol = solve_heat(heat, Ω, bcs, 0.05)
        @test sol(0.05, [0.0, 1.0], [0.0, 1.0]; dv = u(t, x, y)) ≈ zeros(2, 2) atol = 1.0e-12
        l2_error(sol, u(t, x, y), exact, 0.05, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 1.8)
end

# PDE:    -ν (u_xx + u_yy) + b.grad u + c u = f with ν = 0.1, b = (1, 0.5), c = 1 (steady advection-diffusion-reaction)
# Domain: the unit square, meshes of 8, 16 and 32 cells per side, P1 triangles
# BCs:    u(left) ~ u(right) and u(bottom) ~ u(top), with no time variable
# Exact:  u = sin(2 pi x) cos(2 pi y) + 0.5, so f follows by differentiation
# Shows:  the steady path imposes periodic conditions too and converges at rate 2; SUPG with a small τ stays within
#         a factor 1.5 of the Galerkin error and converges at a rate above 1.5.
@testset "steady advection-diffusion on a periodic box converges at rate 2, with Galerkin and with SUPG" begin
    ν, bx, by, c = 0.1, 1.0, 0.5, 1.0
    ex = sin(2pi * x) * cos(2pi * y) + 0.5
    exact(x_, y_) = sin(2pi * x_) * cos(2pi * y_) + 0.5
    D(f, v) = ModelingToolkit.expand_derivatives(Differential(v)(f))
    source = -ν * (D(D(ex, x), x) + D(D(ex, y), y)) + bx * D(ex, x) + by * D(ex, y) + c * ex
    U = u(x, y)
    eq = -ν * (Dx(Dx(U)) + Dy(Dy(U))) + bx * Dx(U) + by * Dy(U) + c * U ~ source
    function solve_periodic(n, test_space)
        Ω = unit_square(Triangle, n)
        left, right, bottom, top = sides(Ω)
        @named pdesys = PDESystem([eq], [u(left) ~ u(right), u(bottom) ~ u(top)], [(x, y) ∈ Ω], [x, y], [U], [])
        return solve(discretize(pdesys, FEMDiscretization(Ω; test_space)), NewtonRaphson(); abstol = 1.0e-12)
    end
    ns = (8, 16, 32)
    errors = map(n -> l2_error(solve_periodic(n, Galerkin()), U, exact, sample_points(40), sample_points(40)), ns)
    @test all(observed_rate(errors, 1 ./ ns) .> 1.8)
    supg = SUPG((hK, b) -> hK / (20 * hypot(b...)))
    errors_supg = map(n -> l2_error(solve_periodic(n, supg), U, exact, sample_points(40), sample_points(40)), ns)
    @test all(observed_rate(errors_supg, 1 ./ ns) .> 1.5)
    @test all(errors_supg .< 1.5 .* errors)
end

# PDE:    u_t = u_xx + u_yy and v_t = v_xx + v_yy + 3 u, two coupled fields with P2 and P1 spaces
# Domain: the unit square, t in (0, 0.05), 16 x 16 triangle pairs
# BCs:    u periodic in x and y; v periodic in x only and zero on bottom and top
# Exact:  u = exp(-4 pi^2 t) sin(2 pi x), v = exp(-5 pi^2 t) sin(2 pi x) sin(pi y) + the particular solution of the coupling
# Shows:  the periodic pairs are per field: one field may be periodic where another has Dirichlet pieces.
@testset "periodic conditions apply per field in a coupled system with different spaces" begin
    Ω = unit_square(Triangle, 16)
    left, right, bottom, top = sides(Ω)
    eqs = [heat, Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y))) + 3u(t, x, y)]
    bcs = [
        u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), v(t, left) ~ v(t, right), v(t, bottom) ~ 0, v(t, top) ~ 0,
        u(0, x, y) ~ sin(2pi * x), v(0, x, y) ~ 0,
    ]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y), v(t, x, y)], [])
    spaces = Dict(u => Lagrange{RefTriangle, 2}(), v => Lagrange{RefTriangle, 1}())
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces)), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test l2_error(sol, u(t, x, y), (x_, y_) -> exp(-4pi^2 * 0.05) * sin(2pi * x_), 0.05, sample_points(40), sample_points(40)) < 1.0e-4
    edge = collect(range(0.0, 1.0; length = 7))
    @test sol(0.05, [0.0], edge; dv = v(t, x, y)) ≈ sol(0.05, [1.0], edge; dv = v(t, x, y)) atol = 1.0e-12
    @test sol(0.05, edge, [0.0, 1.0]; dv = v(t, x, y)) ≈ zeros(7, 2) atol = 1.0e-12
end

# PDE:    the 2D heat equation on the unit square
# BCs:    malformed periodic pairs, pieces that are not translates of each other, and a periodic piece with another
#         condition on the same field
# Shows:  each mistake is refused when the problem is built, and the message names the pieces involved.
@testset "periodic conditions that cannot be imposed are rejected with the pieces named" begin
    ic = u(0, x, y) ~ 0
    Ω = unit_square(Triangle, 4)
    left, right, bottom, top = sides(Ω)
    lower(bcs; mesh = Ω, kw...) = begin
        @named pdesys = PDESystem([heat], bcs, [(x, y) ∈ mesh, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        discretize(pdesys, FEMDiscretization(mesh, t; kw...))
    end
    dirichlet = [u(t, bottom) ~ 0, u(t, top) ~ 0]
    @test lower([u(t, left) ~ u(t, right); dirichlet; ic]) isa SciMLBase.DAEProblem
    @test_throws "periodic pieces Γ_left and Γ_bottom of u do not match under a translation" lower([u(t, left) ~ u(t, bottom), u(t, right) ~ 0, u(t, top) ~ 0, ic])
    # A mesh stretched by 1 + x / 2 in y: the sides have the same number of nodes, but the right one is longer.
    grid = generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0))
    stretched = Grid(grid.cells, [Node(Vec(n.x[1], n.x[2] * (1 + n.x[1] / 2))) for n in grid.nodes]; facetsets = grid.facetsets)
    Ωs = MeshDomain(stretched)
    left_s, right_s, bottom_s, top_s = sides(Ωs)
    @test_throws "periodic pieces Γ_left and Γ_right of u do not match under a translation" lower(
        [u(t, left_s) ~ u(t, right_s), u(t, bottom_s) ~ 0, u(t, top_s) ~ 0, ic]; mesh = Ωs
    )
    @test_throws "has a periodic condition on Γ_left and a Dirichlet condition on the same piece" lower([u(t, left) ~ u(t, right), u(t, left) ~ 0, dirichlet..., ic])
    @test_throws "has a periodic condition on Γ_right and a Neumann or Robin condition on the same piece" lower(
        [u(t, left) ~ u(t, right), NormalDerivative(right)(u(t, right)) ~ 0, dirichlet..., ic]
    )
    @test_throws "must be a periodic pair of the form u(t, Γa) ~ u(t, Γb)" lower([u(t, left) ~ u(t, right) + 1, dirichlet..., ic])
    @test_throws "dirichlet = :eliminate does not support periodic conditions" lower([u(t, left) ~ u(t, right); dirichlet; ic]; dirichlet = :eliminate)
end
