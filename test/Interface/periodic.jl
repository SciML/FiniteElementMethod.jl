# Periodic boundary conditions through the full interface: an equation between the traces of one field on two
# boundary pieces, `u(t, Γa) ~ u(t, Γb)` (or `u(Γa) ~ u(Γb)` without time), identifies the field on the two pieces when
# they are translates of each other. Each case builds the PDESystem as a user would, solves it and compares with an
# exact solution; the last cases are the inputs that are refused.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, NonlinearSolveFirstOrder, Test
using Ferrite: Ferrite, Grid, Node, generate_grid, Line, Triangle, Quadrilateral, Hexahedron, Vec
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

# BCs:    a periodic pair for a field whose space is not a scalar Lagrange space (an edge element)
# Shows:  the unsupported combination is refused with an error naming the space, rather than giving a wrong answer.
@testset "periodic conditions on a non-Lagrange space are rejected" begin
    Ω = unit_square(Triangle, 4)
    left, right, bottom, top = sides(Ω)
    bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ 0]
    @named pdesys = PDESystem([heat], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Ferrite.Nedelec{RefTriangle, 1}()))
    @test_throws "is not a nodal Lagrange space" discretize(pdesys, disc)
end

# PDE:    ((1 + u^2) u_x)_x + ((1 + u^2) u_y)_y - u = f, a steady nonlinear diffusion solved by Newton's method
# Domain: the unit square, meshes of 8, 16 and 32 cells per side, P1 triangles
# BCs:    periodic left and right, periodic bottom and top, no time variable
# Exact:  u = 0.5 sin(2 pi x) cos(2 pi y) + 1, so f follows by differentiation
# Shows:  the Newton residual and Jacobian carry the periodic ties: the nonlinear steady problem converges at rate 2.
@testset "steady nonlinear diffusion on a periodic box converges at rate 2" begin
    ex = 0.5 * sin(2pi * x) * cos(2pi * y) + 1
    exact(x_, y_) = 0.5 * sin(2pi * x_) * cos(2pi * y_) + 1
    D(f, v) = ModelingToolkit.expand_derivatives(Differential(v)(f))
    source = D((1 + ex^2) * D(ex, x), x) + D((1 + ex^2) * D(ex, y), y) - ex
    U = u(x, y)
    eq = Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) - U ~ source
    ns = (8, 16, 32)
    errors = map(ns) do n
        Ω = unit_square(Triangle, n)
        left, right, bottom, top = sides(Ω)
        @named pdesys = PDESystem([eq], [u(left) ~ u(right), u(bottom) ~ u(top)], [(x, y) ∈ Ω], [x, y], [U], [])
        sol = solve(discretize(pdesys, FEMDiscretization(Ω)), NewtonRaphson(); abstol = 1.0e-11)
        @test SciMLBase.successful_retcode(sol)
        l2_error(sol, U, exact, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 1.8)
end

# PDE:    u_t + u u_x + u u_y = ν (u_xx + u_yy) + f with ν = 0.1 (viscous Burgers), a transient nonlinear problem
# Domain: the unit square, t in (0, 0.1), meshes of 8, 16 and 32 cells per side, P1 triangles
# BCs:    periodic in both directions; IC sin(2 pi (x + y))
# Exact:  u = exp(-t) sin(2 pi (x + y)), so f follows by differentiation
# Shows:  the nonlinear DAE path carries the periodic ties and converges at rate 2.
@testset "viscous Burgers on a periodic box converges at rate 2 in the transient path" begin
    ν = 0.1
    ex = exp(-t) * sin(2pi * (x + y))
    exact(x_, y_) = exp(-0.1) * sin(2pi * (x_ + y_))
    D(f, v) = ModelingToolkit.expand_derivatives(Differential(v)(f))
    source = D(ex, t) + ex * D(ex, x) + ex * D(ex, y) - ν * (D(D(ex, x), x) + D(D(ex, y), y))
    U = u(t, x, y)
    eq = Dt(U) + U * Dx(U) + U * Dy(U) ~ ν * (Dx(Dx(U)) + Dy(Dy(U))) + source
    ns = (8, 16, 32)
    errors = map(ns) do n
        Ω = unit_square(Triangle, n)
        left, right, bottom, top = sides(Ω)
        bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ sin(2pi * (x + y))]
        sol = solve_heat(eq, Ω, bcs, 0.1)
        @test SciMLBase.successful_retcode(sol)
        l2_error(sol, U, exact, 0.1, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 1.8)
end

# PDE:    u_t + c u_x = ε u_xx with c = 1, ε = 0.05, a travelling decaying wave on a ring
# Domain: x in (0, 1), t in (0, 0.2), meshes of 16, 32 and 64 line cells, P1; Galerkin and SUPG
# BCs:    u(t, left) ~ u(t, right); IC sin(2 pi x)
# Exact:  u = exp(-4 pi^2 ε t) sin(2 pi (x - c t))
# Shows:  periodic conditions on a line mesh, in the transient path, converge at rate 2 with Galerkin testing and at rate 1
#         with the first-order SUPG stabilisation.
@testset "advection-diffusion on a periodic line converges at rate $rate with $name testing" for (name, test_space, rate) in (
        ("Galerkin", Galerkin(), 2), ("SUPG", SUPG((hK, b) -> hK / (20 * abs(b[1]))), 1),
    )
    c, ε = 1.0, 0.05
    eq = Dt(u(t, x)) + c * Dx(u(t, x)) ~ ε * Dx(Dx(u(t, x)))
    exact(x_) = exp(-4pi^2 * ε * 0.2) * sin(2pi * (x_ - c * 0.2))
    ns = (16, 32, 64)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
        left, right = boundarypiece.(Ω, (:left, :right))
        bcs = [u(t, left) ~ u(t, right), u(0, x) ~ sin(2pi * x)]
        @named pdesys = PDESystem([eq], bcs, [(x,) ∈ Ω, t ∈ Interval(0.0, 0.2)], [t, x], [u(t, x)], [])
        sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; test_space)), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
        @test sol(0.2, [0.0]; dv = u(t, x)) ≈ sol(0.2, [1.0]; dv = u(t, x)) atol = 1.0e-12
        l2_error(sol, u(t, x), exact, 0.2, sample_points(200))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> rate - 0.2)
end

# PDE:    -(u_xx + u_yy + u_zz) + u = f
# Domain: the unit cube, meshes of 4 and 8 hexahedra per side, P1
# BCs:    periodic in x, y and z (three pairs; the 8 corners form one class)
# Exact:  u = sin(2 pi x) sin(2 pi y) cos(2 pi z) + 1
# Shows:  periodic conditions in 3D, with edges and corners shared by several pairs, converge at rate 2.
@testset "steady reaction-diffusion on a fully periodic cube converges at rate 2" begin
    @variables z
    Dz = Differential(z)
    ex = sin(2pi * x) * sin(2pi * y) * cos(2pi * z) + 1
    exact(x_, y_, z_) = sin(2pi * x_) * sin(2pi * y_) * cos(2pi * z_) + 1
    D(f, v) = ModelingToolkit.expand_derivatives(Differential(v)(f))
    source = -(D(D(ex, x), x) + D(D(ex, y), y) + D(D(ex, z), z)) + ex
    U = u(x, y, z)
    eq = -(Dx(Dx(U)) + Dy(Dy(U)) + Dz(Dz(U))) + U ~ source
    ns = (4, 8)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Hexahedron, (n, n, n), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0)))
        left, right, front, back, bottom, top = boundarypiece.(Ω, (:left, :right, :front, :back, :bottom, :top))
        @named pdesys = PDESystem([eq], [u(left) ~ u(right), u(front) ~ u(back), u(bottom) ~ u(top)], [(x, y, z) ∈ Ω], [x, y, z], [U], [])
        sol = solve(discretize(pdesys, FEMDiscretization(Ω)), NewtonRaphson(); abstol = 1.0e-11)
        l2_error(sol, U, exact, sample_points(12), sample_points(12), sample_points(12))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 1.7)
end

# PDE:    the 2D heat equation on the unit square, t in (0, 0.2), 4 x 4 triangle pairs
# BCs:    periodic left and right; Dirichlet data on bottom and top whose corners are shared through the periodic map
# Shows:  Dirichlet data that agree at the two ends of a periodic class are imposed and the solution is periodic; data that
#         differ (at any time of the interval) are an error naming the pieces, the nodes and the two values.
@testset "Dirichlet data on the corners of a periodic class must agree" begin
    Ω = unit_square(Triangle, 4)
    left, right, bottom, top = sides(Ω)
    ic = u(0, x, y) ~ 1
    build(bcs; tend = 0.2) = begin
        @named pdesys = PDESystem([heat], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], [])
        discretize(pdesys, FEMDiscretization(Ω, t))
    end
    sol = solve(build([u(t, left) ~ u(t, right), u(t, bottom) ~ 1, u(t, top) ~ 1, ic]), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    edge = collect(range(0.0, 1.0; length = 5))
    @test sol(0.2, [0.0], edge; dv = u(t, x, y)) ≈ sol(0.2, [1.0], edge; dv = u(t, x, y)) atol = 1.0e-12
    @test sol(0.2, [0.0, 1.0], [0.0, 1.0]; dv = u(t, x, y)) ≈ ones(2, 2) atol = 1.0e-12
    # Time-dependent data that agree: the same function of t on both corners.
    @test build([u(t, left) ~ u(t, right), u(t, bottom) ~ 1 + t, u(t, top) ~ 1 + t, ic]) isa SciMLBase.DAEProblem
    @test_throws "the Dirichlet data of periodic pieces (Γ_left ~ Γ_right) disagree" build([u(t, left) ~ u(t, right), u(t, bottom) ~ x, u(t, top) ~ 0, ic])
    @test_throws "Γ_bottom prescribes 0.0 and Γ_bottom prescribes 1.0" build([u(t, left) ~ u(t, right), u(t, bottom) ~ x, u(t, top) ~ 0, ic])
    # Data that agree at all of the 17 sample times of the interval but differ in between are caught when the solver
    # evaluates them: x sin(16 pi t) agrees on the two bottom corners at t = 0, 1/16, ... 1 and differs by 1 at t = 1/32.
    sneaky = build([u(t, left) ~ u(t, right), u(t, bottom) ~ x * sin(16pi * t), u(t, top) ~ 0, ic]; tend = 1.0)
    @test sneaky isa SciMLBase.DAEProblem
    @test_throws "the Dirichlet data of periodic pieces (Γ_left ~ Γ_right) disagree" sneaky.f(zeros(length(sneaky.u0)), zeros(length(sneaky.u0)), sneaky.u0, sneaky.p, 1 / 32)
    # Data that agree at t = 0 and differ later.
    @test_throws "disagree" build([u(t, left) ~ u(t, right), u(t, bottom) ~ t * x, u(t, top) ~ 0, ic])
end

# PDE:    the 2D heat equation, t in (0, 0.1)
# BCs:    periodic left and right on meshes whose sides match one to one (a normal and a graded mesh) and on meshes whose sides
#         have the same number of nodes without a one-to-one correspondence
# Shows:  the nodes of the two pieces are matched one to one; two nodes closer than the tolerance to the same partner, or a
#         node moved by more than the tolerance, are an error naming the pieces and the nodes.
@testset "periodic pieces need a one-to-one correspondence of their nodes" begin
    function problem(grid)
        Ω = MeshDomain(grid)
        left, right, bottom, top = sides(Ω)
        bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ 0]
        @named pdesys = PDESystem([heat], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [u(t, x, y)], [])
        return discretize(pdesys, FEMDiscretization(Ω, t))
    end
    base = generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0))
    remesh(f) = Grid(base.cells, [Node(f(n.x)) for n in base.nodes]; facetsets = base.facetsets)
    @test problem(base) isa SciMLBase.DAEProblem
    @test problem(remesh(X -> Vec(X[1], X[2]^2))) isa SciMLBase.DAEProblem
    moved = remesh(X -> X[1] > 0.99 && 0.4 < X[2] < 0.6 ? Vec(X[1], X[2] + 1.0e-3) : X)
    @test_throws "periodic pieces Γ_left and Γ_right of u do not match under a translation" problem(moved)
    # Two nodes on each side lie within the tolerance of one node of the other side.
    nodes = [Node(Vec(0.0, 0.0)), Node(Vec(1.0, 0.0)), Node(Vec(1.0, 1.0e-7)), Node(Vec(0.0, 1.0e-9)), Node(Vec(1.0, 1.0)), Node(Vec(0.0, 1.0))]
    close_grid = Grid(
        [Quadrilateral((1, 2, 3, 4)), Quadrilateral((4, 3, 5, 6))], nodes;
        facetsets = Dict(
            "left" => Set([Ferrite.FacetIndex(1, 4), Ferrite.FacetIndex(2, 4)]), "right" => Set([Ferrite.FacetIndex(1, 2), Ferrite.FacetIndex(2, 2)]),
            "bottom" => Set([Ferrite.FacetIndex(1, 1)]), "top" => Set([Ferrite.FacetIndex(2, 3)]),
        )
    )
    @test_throws "both match the node" problem(close_grid)
end
