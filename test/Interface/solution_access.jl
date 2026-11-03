# Reading a solution: `sol.t`, `sol[u(t, x, y)]`, `sol[x]`, `sol(t, x, y; dv)`, traces, derivatives and
# the parameters of the PDESystem, on the problem u_t = D (u_xx + u_yy) on the unit square with the
# manufactured solution u = x^2 + y^2 + 4 D t, which a P2 space on triangles holds exactly.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@parameters D
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# Builds, discretizes and solves the problem with diffusivity `diffusivity`; returns the solution and the left piece.
function solve_diffusion(diffusivity)
    Ω = MeshDomain(generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    exact = x^2 + y^2 + 4D * t
    eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
    bcs = [[u(t, Γ) ~ exact for Γ in (left, right, bottom, top)]; u(0, x, y) ~ x^2 + y^2]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [D => diffusivity])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()), saveat = 0.25)
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10), left
end

# PDE:    u_t = D (u_xx + u_yy) with D = 2
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2; saved every 0.25
# BCs:    u = x^2 + y^2 + 4 D t on all sides; IC u(0, x, y) = x^2 + y^2
# Exact:  u = x^2 + y^2 + 4 D t
# Shows:  the shapes of sol.t, sol[u], sol[x] and sol[y], and that column i of sol[u] sits at (sol[x][i], sol[y][i]).
@testset "sol[u(t, x, y)] holds one row per saved time and one column per mesh node" begin
    sol, _ = solve_diffusion(2.0)
    @test sol.t == [0.0, 0.25, 0.5, 0.75, 1.0]
    values, xs, ys = sol[u(t, x, y)], sol[x], sol[y]
    @test size(values) == (5, length(xs)) && length(xs) == length(ys) == 25
    @test values ≈ [x_^2 + y_^2 + 8t_ for t_ in sol.t, (x_, y_) in zip(xs, ys)] atol = 1.0e-7
end

# PDE, Domain, BCs, Exact: as in the first case, with D = 2
# Shows:  sol(t, x, y; dv) evaluates the finite-element field anywhere in the mesh, interpolating in time.
@testset "sol(t, x, y; dv) evaluates the field at points and on tensor grids at any time" begin
    sol, _ = solve_diffusion(2.0)
    exact(t_, x_, y_) = x_^2 + y_^2 + 8t_
    @test sol(0.6, 0.3, 0.7; dv = u(t, x, y)) ≈ exact(0.6, 0.3, 0.7) atol = 1.0e-5
    grid = sol(0.5, [0.1, 0.5, 0.9], [0.2, 0.8]; dv = u(t, x, y))
    @test size(grid) == (3, 2)
    @test grid ≈ [exact(0.5, x_, y_) for x_ in (0.1, 0.5, 0.9), y_ in (0.2, 0.8)] atol = 1.0e-7
    @test isnan(sol(0.5, 1.5, 0.5; dv = u(t, x, y)))   # outside the mesh
    @test_throws DomainError sol(0.5, 1.5, 0.5; dv = u(t, x, y), outside = :error)
    @test_throws DomainError sol(2.0, 0.5, 0.5; dv = u(t, x, y))   # outside the saved time span
end

# PDE, Domain, BCs, Exact: as in the first case, with D = 2
# Shows:  derivatives and traces are read with the same indexing: sol[Dx(u)] is du/dx = 2x,
#         and sol[u(t, Γ)] is the field on the nodes of the boundary piece Γ.
@testset "sol[Dx(u)] gives the gradient and sol[u(t, Γ)] the trace on a boundary piece" begin
    sol, left = solve_diffusion(2.0)
    xs = sol[x]
    @test sol[Dx(u(t, x, y))][end, :] ≈ 2 .* xs atol = 1.0e-5
    trace = sol[u(t, left)]
    @test size(trace) == (5, length(piece_nodes(sol, left)))
    @test trace ≈ [y_^2 + 8t_ for t_ in sol.t, y_ in sol[y][piece_nodes(sol, left)]] atol = 1.0e-7
end

# PDE:    u_t = D (u_xx + u_yy), solved for D = 1, 2 and 4
# Domain: the unit square, t in (0, 1), P2; BCs and IC as in the first case
# Exact:  u = x^2 + y^2 + 4 D t
# Shows:  the numeric value given to the parameter D in the PDESystem's parameter list sets the diffusivity,
#         so the value at the centre of the square, 0.5 + 4 D t, grows with D.
@testset "the parameter values of the PDESystem set the diffusivity" begin
    for diffusivity in (1.0, 2.0, 4.0)
        sol, _ = solve_diffusion(diffusivity)
        @test sol(1.0, 0.5, 0.5; dv = u(t, x, y)) ≈ 0.5 + 4diffusivity atol = 1.0e-7
    end
end
