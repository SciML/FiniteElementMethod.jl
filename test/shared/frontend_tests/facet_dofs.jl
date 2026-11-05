# Index data on the FEMMetadata of a discretized problem: `facet_dofs`, `dof_coordinates` and `cell_dofs`.
using Test
using FiniteElementMethod: rectangle_grid, facet_dofs, dof_coordinates, cell_dofs
using Ferrite: Triangle, Quadrilateral, addcellset!, getcellset, getfacetset, getfacetsets, addfacetset!

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..) w(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))

sides(Ω) = Symbol.(keys(getfacetsets(Ω.mesh)))

# The metadata of the heat problem on `Ω` with no boundary conditions (natural zero flux on every side).
function heat_metadata(Ω; kwargs...)
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Δ(u(t, x, y))], [u(0, x, y) ~ x], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
    )
    return discretize(pdesys, FEMDiscretization(Ω, t; natural = sides(Ω), kwargs...)).problem_type
end

@testset verbose = true "facet_dofs, dof_coordinates, cell_dofs" begin
    @testset "P1 and P2 triangles" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 3)))
        left, top = boundarypiece.(Ω, (:left, :top))
        for (order, nleft, ntop) in ((1, 4, 5), (2, 7, 9))
            md = heat_metadata(Ω; spaces = Dict(u => Lagrange{RefTriangle, order}()))
            ndofs = size(dof_coordinates(md), 2)
            X = dof_coordinates(md)
            @test size(X) == (2, (order == 1 ? 5 * 4 : 9 * 7))
            dl = facet_dofs(md, left, u)
            @test length(dl) == nleft && issorted(dl)
            @test all(iszero, X[1, dl])
            @test facet_dofs(md, "top", :u) == facet_dofs(md, top, u)
            @test length(facet_dofs(md, top, u(t, x, y))) == ntop
            @test all(≈(1.0), X[2, facet_dofs(md, :top, u)])
            C = cell_dofs(md, u)
            @test size(C) == (order == 1 ? 3 : 6, 4 * 3 * 2)
            @test sort!(unique(vec(C))) == 1:ndofs
        end
    end

    @testset "a mesh of one cell still gives a matrix" begin
        md = heat_metadata(MeshDomain(rectangle_grid(Quadrilateral, (1, 1))))
        @test cell_dofs(md, :u) isa Matrix{Int}
        @test size(cell_dofs(md, :u)) == (4, 1)
    end

    @testset "a field that does not reach the facet set has no dofs there" begin
        grid = rectangle_grid(Triangle, (4, 4))
        addcellset!(grid, "left_half", X -> X[1] <= 0.5 + 1.0e-12)
        region = getcellset(grid, "left_half")
        for side in ("bottom", "top")
            set = getfacetset(grid, side)
            delete!(grid.facetsets, side)
            addfacetset!(grid, side * "_left", Set(fi for fi in set if fi[1] in region))
            addfacetset!(grid, side * "_right", Set(fi for fi in set if !(fi[1] in region)))
        end
        Ω = MeshDomain(grid)
        Γ = Dict(n => boundarypiece(Ω, n) for n in (:left, :right, :bottom_left, :bottom_right, :top_left, :top_right))
        U, W = u(t, x, y), w(t, x, y)
        bcs = [
            [u(t, Γ[n]) ~ 0 for n in keys(Γ)];
            [w(t, Γ[n]) ~ 0 for n in (:left, :bottom_left, :top_left)];
            u(0, x, y) ~ x; w(0, x, y) ~ y
        ]
        @named pdesys = PDESystem(
            [Dt(U) ~ Δ(U), Dt(W) ~ Δ(W) + U], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U, W], []
        )
        md = discretize(pdesys, FEMDiscretization(Ω, t; natural = sides(Ω), regions = Dict(w => Region(Ω, :left_half)))).problem_type
        @test isempty(facet_dofs(md, :right, w))
        @test !isempty(facet_dofs(md, :right, u))
        @test length(facet_dofs(md, :left, w)) == 5
        @test occursin("regions", errmsg(() -> cell_dofs(md, w)))
    end

    @testset "cells that hold no field at all" begin
        grid = rectangle_grid(Triangle, (4, 4))
        addcellset!(grid, "left_half", X -> X[1] <= 0.5 + 1.0e-12)
        Ω = MeshDomain(grid)
        Γ = Dict(n => boundarypiece(Ω, n) for n in (:left, :right))
        U = u(t, x, y)
        @named pdesys = PDESystem(
            [Dt(U) ~ Δ(U)], [u(t, Γ[:left]) ~ 0; u(0, x, y) ~ x], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], []
        )
        md = discretize(pdesys, FEMDiscretization(Ω, t; natural = sides(Ω), regions = Dict(u => Region(Ω, :left_half)))).problem_type
        @test isempty(facet_dofs(md, :right, u))
        @test length(facet_dofs(md, :left, u)) == 5
    end

    @testset "cell_dofs rejects a single region that covers only part of the mesh" begin
        grid = rectangle_grid(Triangle, (4, 4))
        addcellset!(grid, "left_half", X -> X[1] <= 0.5 + 1.0e-12)
        md = heat_metadata(MeshDomain(grid); regions = Dict(u => Region(MeshDomain(grid), :left_half)))
        @test occursin("regions", errmsg(() -> cell_dofs(md, u)))
    end

    @testset "rejections" begin
        md = heat_metadata(MeshDomain(rectangle_grid(Quadrilateral, (2, 2))))
        @test occursin("not a facet set", errmsg(() -> facet_dofs(md, "nope", u)))
        @test occursin("not a field", errmsg(() -> facet_dofs(md, "left", :v)))
        @test occursin("not a field", errmsg(() -> cell_dofs(md, :v)))
    end
end
