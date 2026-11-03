# Exact messages of front-end errors not covered in errors.jl.
using Test
using Ferrite: Ferrite, Triangle, Quadrilateral, Grid, Node, Vec, Lagrange, RefTriangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

@testset "front-end messages" begin
    @testset "mixed cell types" begin
        nodes = [Node(Vec(0.0, 0.0)), Node(Vec(1.0, 0.0)), Node(Vec(0.0, 1.0)), Node(Vec(2.0, 0.0)), Node(Vec(1.0, 1.0))]
        grid = Grid(Ferrite.AbstractCell[Triangle((1, 2, 3)), Quadrilateral((2, 4, 5, 3))], nodes)
        Ω = MeshDomain(grid)
        heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
        @named pdesys = PDESystem([heat], [u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 1}()))
        @test_throws ArgumentError("meshes with mixed cell types are not supported; use a mesh with a single cell type") discretize(
            pdesys, disc
        )
    end
end
