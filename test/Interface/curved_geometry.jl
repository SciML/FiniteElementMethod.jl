# Curved geometry: a quarter annulus meshed with quadratic triangles (Ferrite's `QuadraticTriangle`), whose
# edges on the circles are arcs. The grid is built with Ferrite's public constructors; the discretization maps
# each cell isoparametrically, so the element spaces keep their optimal rates on the curved boundary.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: Ferrite, Grid, Node, QuadraticTriangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# The quarter annulus 1 <= r <= 2, 0 <= θ <= π/2 made of 2 n^2 quadratic triangles: an n x n grid of (r, θ)
# rectangles, each cut into two triangles. Every node, mid-side nodes included, sits at its (r, θ) position.
# Facet sets: "inner" (r = 1), "outer" (r = 2), "bottom" (y = 0) and "left" (x = 0).
function quarter_annulus(n)
    m = 2n + 1
    node_index(i, j) = i + 1 + j * m
    position(i, j) = Vec((1 + i / 2n) * cos(pi / 2 * j / 2n), (1 + i / 2n) * sin(pi / 2 * j / 2n))
    nodes = [Node(position(i, j)) for j in 0:(2n) for i in 0:(2n)]
    cells = QuadraticTriangle[]
    for a in 0:(n - 1), b in 0:(n - 1)
        i, j = 2a, 2b
        ids = (di, dj) -> node_index(i + di, j + dj)
        push!(cells, QuadraticTriangle((ids(0, 0), ids(2, 0), ids(2, 2), ids(1, 0), ids(2, 1), ids(1, 1))))
        push!(cells, QuadraticTriangle((ids(0, 0), ids(2, 2), ids(0, 2), ids(1, 1), ids(1, 2), ids(0, 1))))
    end
    grid = Grid(cells, nodes)
    radius(p) = sqrt(p[1]^2 + p[2]^2)
    Ferrite.addfacetset!(grid, "inner", p -> radius(p) ≈ 1)
    Ferrite.addfacetset!(grid, "outer", p -> radius(p) ≈ 2)
    Ferrite.addfacetset!(grid, "bottom", p -> abs(p[2]) < 1.0e-12)
    Ferrite.addfacetset!(grid, "left", p -> abs(p[1]) < 1.0e-12)
    return grid
end

# L2 error of the field `dv` at time `t_end` over the annulus: the midpoint rule in (r, θ) with weight r dr dθ.
function annulus_l2_error(sol, dv, exact, t_end, n)
    radii, angles = sample_points(n; lo = 1.0, hi = 2.0), sample_points(n; lo = 0.0, hi = pi / 2)
    points = [Vec(r * cos(θ), r * sin(θ)) for r in radii for θ in angles]
    weights = [r * step(radii) * step(angles) for r in radii for θ in angles]
    return sqrt(sum(weights .* abs2.(sol(t_end, points; dv) .- exact.(points))))
end

# Solves u_t = Δu + f on the quarter annulus of `n` cells per direction in order-`order` elements.
function solve_annulus(n, order)
    Ω = MeshDomain(quarter_annulus(n))
    inner, outer, bottom, left = boundarypiece.(Ω, (:inner, :outer, :bottom, :left))
    exact = (1 + t) * sin(x) * cos(y)
    bcs = [
        u(t, inner) ~ exact, u(t, outer) ~ exact, u(t, left) ~ exact,
        NormalDerivative(bottom)(u(t, bottom)) ~ 0, u(0, x, y) ~ sin(x) * cos(y),
    ]
    eqs = [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (3 + 2t) * sin(x) * cos(y)]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, order}()))
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = Δu + (3 + 2t) sin(x) cos(y)
# Domain: the quarter annulus 1 <= r <= 2, 0 <= θ <= π/2, t in (0, 0.5), 2 n^2 quadratic triangles,
#         n = 4, 8, 16, with P1 or P2 elements; the arcs r = 1 and r = 2 are exact in the geometry
# BCs:    u = (1 + t) sin(x) cos(y) on the arcs and on x = 0; ∂ₙu = 0 on y = 0 (which the exact solution satisfies)
# Exact:  u = (1 + t) sin(x) cos(y), not in the element space
# Shows:  a curved boundary, given by mid-side nodes, keeps the optimal L2 rates p + 1 (P1: 2, P2: 3),
#         so the boundary is not approximated by chords.
@testset "$name elements keep the optimal rate on a quarter annulus with curved boundaries" for (name, order, rate) in (("P1", 1, 2), ("P2", 2, 3))
    ns = (4, 8, 16)
    errors = map(ns) do n
        sol = solve_annulus(n, order)
        annulus_l2_error(sol, u(t, x, y), p -> 1.5 * sin(p[1]) * cos(p[2]), 0.5, 60)
    end
    @test issorted(errors; rev = true)
    @test all(observed_rate(errors, 1 ./ ns) .> rate - 0.2)
end
