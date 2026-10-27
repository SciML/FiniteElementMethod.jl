# Claims of the user documentation that the code must keep true.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@testset verbose = true "documentation" begin
    @testset "the domain pairing names the mesh coordinates" begin
        @variables x y u(..)
        Dt = Differential(t)
        grid = rectangle_grid(Triangle, (4, 2); hi = (2.0, 1.0))
        Ω = MeshDomain(grid)
        bcs(u) = [u(t, Γ) ~ 0 for Γ in boundarypieces(Ω)]
        coordinate(d) = [Ferrite.get_node_coordinate(n)[d] for n in Ferrite.getnodes(grid)]

        # `(y, x) ∈ Ω` with `u(t, y, x)`: y is mesh coordinate 1.
        eq = Dt(u(t, y, x)) ~ Differential(x)(Differential(x)(u(t, y, x))) + Differential(y)(Differential(y)(u(t, y, x)))
        @named sys = PDESystem(
            [eq], [bcs(u); u(0, y, x) ~ y], [(y, x) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, y, x], [u(t, y, x)], []
        )
        sol = solve(discretize(sys, FEMDiscretization(Ω, t)), EXACT_ROWS)
        @test sol[y] == coordinate(1)
        @test sol[x] == coordinate(2)
        interior = [i for i in eachindex(sol[y]) if 0 < sol[y][i] < 2 && 0 < sol[x][i] < 1]
        @test sol[u(t, y, x)][1, interior] ≈ sol[y][interior]
        @test sol(0.0, 1.5, 0.5; dv = u(t, y, x)) ≈ 1.5

        # `(y, x) ∈ Ω` with `u(t, x, y)`: the pairing, not the argument order, numbers the coordinates.
        # On the 2 × 1 mesh the same physical solution, 1 + p₁ + 2p₂ in mesh coordinates p, is
        # written once with x = p₁ and once with y = p₁; the two solves must agree node by node.
        Dx, Dy = Differential(x), Differential(y)
        heat(f) = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + f
        dirichlet(f) = [[u(t, Γ) ~ f * (1 + t) for Γ in boundarypieces(Ω)]; u(0, x, y) ~ f]
        straight, _ = fem_solve([heat(1 + x + 2y)], dirichlet(1 + x + 2y), Ω, (x, y), t, [u(t, x, y)])
        swapped, _ = fem_solve([heat(1 + y + 2x)], dirichlet(1 + y + 2x), Ω, (y, x), t, [u(t, x, y)])
        @test SciMLBase.successful_retcode(straight) && SciMLBase.successful_retcode(swapped)
        @test straight[x] == swapped[y] == coordinate(1)
        @test straight[y] == swapped[x] == coordinate(2)
        @test straight.t == swapped.t
        @test maximum(abs, swapped[u(t, x, y)] - straight[u(t, x, y)]) <= 1.0e-12
        exact = [(1 + swapped[y][i] + 2swapped[x][i]) * (1 + tk) for tk in swapped.t, i in eachindex(swapped[x])]
        @test maximum(abs, swapped[u(t, x, y)] - exact) <= 1.0e-8
    end
end
