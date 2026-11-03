# `symbolic_discretize(pdesys, disc)` returns the `(sys, tspan)` of the discretization `discretize(pdesys, disc)` builds.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t, ProblemTypeCtx

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

@testset verbose = true "symbolic_discretize" begin
    Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    bcs = [[u(t, Γ) ~ 0 for Γ in sides]; u(0, x, y) ~ x]
    @named pdesys = PDESystem([heat], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t)

    @testset "the system and time span of discretize" begin
        sys, tspan = SciMLBase.symbolic_discretize(pdesys, disc)
        prob = discretize(pdesys, disc)
        @test sys isa ModelingToolkit.System
        @test tspan == prob.tspan == (0.0, 0.5)
        @test isequal(unknowns(sys), unknowns(prob.f.sys))
        @test length(equations(sys)) == length(equations(prob.f.sys))
        @test isequal(ModelingToolkit.parameters(sys), ModelingToolkit.parameters(prob.f.sys))
        md = ModelingToolkit.getmetadata(sys, ProblemTypeCtx, nothing)
        @test ModelingToolkit.getmetadata(prob.f.sys, ProblemTypeCtx, nothing) === prob.problem_type
        @test md isa FEMMetadata
        @test md.metadata[] === sys
        @test sprint(show, md) == sprint(show, prob.problem_type) == "FEMMetadata(fields = [u], ndofs = 9, hasTime = true)"
        @test isequal(md.x̄, prob.problem_type.x̄)
        @test isequal([f.dv for f in md.fields], [f.dv for f in prob.problem_type.fields])
        @test md.u0 == prob.problem_type.u0
        @test md.pieces == prob.problem_type.pieces
    end

    @testset "keywords reach the problem construction of discretize" begin
        sys, tspan = SciMLBase.symbolic_discretize(pdesys, disc; saveat = 0.25)
        @test tspan == (0.0, 0.5)
        @test isequal(unknowns(sys), unknowns(discretize(pdesys, disc; saveat = 0.25).f.sys))
        reserved = "keyword `du0` is set by the FEM discretization and cannot be passed; the reserved keywords are iv, unknowns, du0, coordinate, name, parameters"
        @test_throws ArgumentError(reserved) discretize(pdesys, disc; du0 = zeros(9))
        @test_throws ArgumentError(reserved) SciMLBase.symbolic_discretize(pdesys, disc; du0 = zeros(9))
    end

    @testset "no further positional arguments" begin
        @test_throws ArgumentError(
            "symbolic_discretize(pdesys, disc; kwargs...) takes no further positional arguments; got 1"
        ) SciMLBase.symbolic_discretize(pdesys, disc, :extra)
        @test_throws ArgumentError(
            "symbolic_discretize(pdesys, disc; kwargs...) takes no further positional arguments; got 2"
        ) SciMLBase.symbolic_discretize(pdesys, disc, 1, 2)
    end

    # The steady path builds its NonlinearProblem from a bare NonlinearFunction, with no System to return.
    @testset "steady discretization" begin
        @named steady = PDESystem(
            [Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], [u(Γ) ~ 0 for Γ in sides], [(x, y) ∈ Ω], [x, y], [u(x, y)], []
        )
        @test NonlinearProblem(steady, FEMDiscretization(Ω)).f.sys === nothing
        @test_throws ArgumentError(
            "a steady FEMDiscretization builds no symbolic System, so it has no symbolic_discretize; " *
                "call discretize(pdesys, disc) for its NonlinearProblem"
        ) SciMLBase.symbolic_discretize(steady, FEMDiscretization(Ω))
    end
end
