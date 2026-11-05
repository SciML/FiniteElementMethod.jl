# Steady problems through the full interface: a PDESystem with no time variable and a FEMDiscretization built
# without one lower to a NonlinearProblem, and `solve` returns a solution called without a time, `sol(x, y; dv)`.
using FiniteElementMethod, ModelingToolkit, DomainSets, NonlinearSolveFirstOrder, Symbolics, Test
using Ferrite: generate_grid, Line, Triangle, Vec
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dx, Dy = Differential(x), Differential(y)

# Solves the steady `eq` for u on the unit square meshed by `n` x `n` triangle pairs with the given order.
function solve_steady_square(eq, bcs, n, order)
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    @named pdesys = PDESystem([eq], bcs(Ω), [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    disc = FEMDiscretization(Ω; spaces = Dict(u => Lagrange{RefTriangle, order}()))
    return solve(discretize(pdesys, disc), NewtonRaphson(); abstol = 1.0e-12)
end

# PDE:    u_xx = -pi^2 sin(pi x)
# Domain: x in (0, 1), meshes of 4, 8, 16 and 32 line cells (P1) or 2, 4, 8 and 16 (P2)
# BCs:    u(0) = 0, u(1) = 1 (Dirichlet); no time variable, so no initial condition
# Exact:  u = sin(pi x) + x
# Shows:  a steady Poisson problem converges at the optimal L2 rates 2 (P1) and 3 (P2); the linear problem takes one Newton step.
@testset "steady Poisson on a line converges at rate $(order + 1) for P$order elements" for (order, ns) in ((1, (4, 8, 16, 32)), (2, (2, 4, 8, 16)))
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
        left, right = boundarypiece.(Ω, (:left, :right))
        @named pdesys = PDESystem([Dx(Dx(u(x))) ~ -pi^2 * sin(pi * x)], [u(left) ~ 0, u(right) ~ 1], [(x,) ∈ Ω], [x], [u(x)], [])
        disc = FEMDiscretization(Ω; spaces = Dict(u => Lagrange{RefLine, order}()))
        sol = solve(discretize(pdesys, disc), NewtonRaphson(); abstol = 1.0e-12)
        @test sol.stats.nsteps == 1
        l2_error(sol, u(x), x_ -> sin(pi * x_) + x_, sample_points(400))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> order + 0.8)
end

# PDE:    u_xx + u_yy = (1 - pi^2) sin(pi x) exp(y)
# Domain: the unit square, meshes of 4 x 4, 8 x 8 and 16 x 16 triangle pairs
# BCs:    Dirichlet u = sin(pi x) exp(y) on left, right and bottom; Neumann du/dn = e sin(pi x) on top
# Exact:  u = sin(pi x) exp(y)
# Shows:  a steady problem with Dirichlet and Neumann pieces converges at rate 2 (P1) and 3 (P2).
@testset "steady Poisson with Dirichlet and Neumann sides converges at rate $(order + 1) for P$order elements" for order in (1, 2)
    exact(x_, y_) = sin(pi * x_) * exp(y_)
    bcs(Ω) = begin
        left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        [u(left) ~ exact(x, y), u(right) ~ exact(x, y), u(bottom) ~ exact(x, y), NormalDerivative(top)(u(top)) ~ ℯ * sin(pi * x)]
    end
    eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ (1 - pi^2) * exact(x, y)
    ns = (4, 8, 16)
    errors = [l2_error(solve_steady_square(eq, bcs, n, order), u(x, y), exact, sample_points(50), sample_points(50)) for n in ns]
    @test all(observed_rate(errors, 1 ./ ns) .> order + 0.8)
end

# PDE:    (k(u) u_x)_x + (k(u) u_y)_y = f with k(u) = 1 + u^2 (nonlinear diffusion)
# Domain: the unit square, meshes of 4 x 4, 8 x 8 and 16 x 16 triangle pairs, P2
# BCs:    u = 0 on all four sides
# Exact:  u = sin(pi x) sin(pi y); the source f is the left-hand side evaluated at the exact solution
# Shows:  a nonlinear steady problem is solved by Newton's method (more than one step) at rate 3.
@testset "steady nonlinear diffusion converges at rate 3 with P2 elements" begin
    exact = sin(pi * x) * sin(pi * y)
    f = expand_derivatives(Dx((1 + exact^2) * Dx(exact)) + Dy((1 + exact^2) * Dy(exact)))
    eq = Dx((1 + u(x, y)^2) * Dx(u(x, y))) + Dy((1 + u(x, y)^2) * Dy(u(x, y))) ~ f
    bcs(Ω) = [u(Γ) ~ 0 for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]
    ns = (4, 8, 16)
    sols = [solve_steady_square(eq, bcs, n, 2) for n in ns]
    @test all(sol -> sol.stats.nsteps > 1, sols)
    errors = [l2_error(sol, u(x, y), (x_, y_) -> sin(pi * x_) * sin(pi * y_), sample_points(50), sample_points(50)) for sol in sols]
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end
