# `dirichlet = :eliminate`: the prescribed dofs are removed from the unknowns instead of kept as algebraic rows, so the
# DAE is smaller and has no Dirichlet rows. It needs Dirichlet data that are constant in time (they may vary in space).
# The solution interface is unchanged: `sol[u]` and `sol(t, x, y; dv)` hold the full field.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, NonlinearSolveFirstOrder, Test
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

unit_square(n) = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.1), 8 x 8 triangle pairs, P2
# BCs:    u = 1 + x + 2y on all sides (constant in time, linear in space); IC u = 1 + x + 2y + sin(pi x) sin(pi y)
# Exact:  u = 1 + x + 2y + exp(-2 pi^2 t) sin(pi x) sin(pi y)
# Shows:  the eliminated problem has fewer unknowns, the same solution as the algebraic one, and the boundary values.
@testset "dirichlet = :eliminate solves the heat equation with fewer unknowns and the same solution" begin
    Ω = unit_square(8)
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U = u(t, x, y)
    bcs = [[u(t, Γ) ~ 1 + x + 2y for Γ in sides]; u(0, x, y) ~ 1 + x + 2y + sin(pi * x) * sin(pi * y)]
    @named pdesys = PDESystem([Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [U], [])
    solve_with(dirichlet) = begin
        prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()), dirichlet, saveat = 0.025))
        prob, solve(prob, DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    end
    (algebraic_problem, algebraic), (eliminated_problem, eliminated) = solve_with(:algebraic), solve_with(:eliminate)
    @test length(eliminated_problem.u0) == length(algebraic_problem.u0) - 64 # 32 vertices and 32 mid-edge nodes on the boundary
    @test eliminated[U] ≈ algebraic[U] atol = 1.0e-8
    exact(x_, y_) = 1 + x_ + 2y_ + exp(-2pi^2 * 0.1) * sin(pi * x_) * sin(pi * y_)
    @test l2_error(eliminated, U, exact, 0.1, sample_points(40), sample_points(40)) < 1.0e-3
    left = first(sides)
    @test eliminated[u(t, left)] ≈ repeat(permutedims(1 .+ 2 .* eliminated[y][piece_nodes(eliminated, left)]), length(eliminated.t)) atol = 1.0e-12
end

# PDE:    u_xx + u_yy = 0 (steady)
# Domain: the unit square, 4 x 4 triangle pairs, P1
# BCs:    u = 1 + x + 2y on all sides
# Exact:  u = 1 + x + 2y
# Shows:  a steady problem takes `dirichlet = :eliminate` too: fewer unknowns, the full field in the solution.
@testset "dirichlet = :eliminate on a steady problem leaves only the interior nodes unknown" begin
    Ω = unit_square(4)
    bcs = [u(Γ) ~ 1 + x + 2y for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]
    @named pdesys = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], bcs, [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    prob = discretize(pdesys, FEMDiscretization(Ω; dirichlet = :eliminate))
    sol = solve(prob, NewtonRaphson(); abstol = 1.0e-12)
    @test length(prob.u0) == 9 # the 3 x 3 interior nodes of the 5 x 5 vertex lattice
    @test sol[u(x, y)] ≈ 1 .+ sol[x] .+ 2 .* sol[y] atol = 1.0e-10
end
