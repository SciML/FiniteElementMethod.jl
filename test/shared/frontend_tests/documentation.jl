# Claims of the user documentation that the code must keep true.
using Test
using FiniteElementMethod: FiniteElementMethod, PDEBase, rectangle_grid
using Ferrite: Ferrite, Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@testset verbose = true "documentation" begin
    @testset "supported_conditions matches docs/src/boundary_conditions.md" begin
        if isdefined(PDEBase, :supported_conditions)
            disc = FEMDiscretization(MeshDomain(rectangle_grid(Triangle, (1, 1))))
            rendered = repr(MIME"text/markdown"(), PDEBase.supported_conditions(disc))
            page = read(joinpath(pkgdir(FiniteElementMethod), "docs", "src", "boundary_conditions.md"), String)
            @test occursin(rendered, page)
        else
            @test_skip "the installed PDEBase has no supported_conditions"
        end
    end

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

        # A pairing in another order than the dependent variable's arguments is an error.
        eq = Dt(u(t, x, y)) ~ Differential(x)(Differential(x)(u(t, x, y)))
        @named swapped = PDESystem(
            [eq], [bcs(u); u(0, x, y) ~ 0], [(y, x) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
        )
        @test_throws ArgumentError(
            "the domain pairing names the coordinates in the order (y, x) but the dependent variables take them in the order (x, y); write both in the same order"
        ) discretize(swapped, FEMDiscretization(Ω, t))
    end

    @testset "a time-dependent Robin coefficient is rejected" begin
        @variables x y u(..)
        Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))
        Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        eq = Differential(t)(u(t, x, y)) ~ Differential(x)(Differential(x)(u(t, x, y))) + Differential(y)(Differential(y)(u(t, x, y)))
        bcs = [
            u(t, Γl) ~ 0, u(t, Γb) ~ 0, u(t, Γt) ~ 0,
            NormalDerivative(Γr)(u(t, Γr)) + (1 + t) * u(t, Γr) ~ 0, u(0, x, y) ~ 0,
        ]
        @named sys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        @test_throws ArgumentError(
            "time-dependent coefficients in the mass or stiffness operator land in M3: equation 1 has U₁*V + T*U₁*V"
        ) discretize(sys, FEMDiscretization(Ω, t))
    end
end
