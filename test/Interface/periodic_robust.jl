# Robustness of the periodic matching: pieces whose nodes match but whose facets do not, meshes far from the origin or thin
# along the periodic axis, and `operators` when every periodic node is prescribed.
using FiniteElementMethod, ModelingToolkit, DomainSets, NonlinearSolveFirstOrder, Test
using FiniteElementMethod: FEMSpace
using ModelingToolkit: t_nounits as t
using Ferrite: Ferrite, Lagrange, RefQuadrilateral, Grid, Node, Tetrahedron, Hexahedron, Quadrilateral, generate_grid, addfacetset!, Vec
using LinearAlgebra: det

@variables x y z u(..)
Dx, Dy, Dz = Differential(x), Differential(y), Differential(z)

# The unit cube as 5 tetrahedra, whose opposite faces are split along opposite diagonals, with the six facet sets of
# `generate_grid`.
function five_tetrahedra_cube()
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    nodes = [Node(Vec(Float64.(c)...)) for c in corners]
    cells = Ferrite.Tetrahedron[]
    for t in ((2, 4, 5, 7), (1, 2, 4, 5), (3, 2, 4, 7), (6, 2, 5, 7), (8, 4, 5, 7))
        p = [collect(Float64, corners[i]) for i in t]
        positive = det(hcat(p[2] - p[1], p[3] - p[1], p[4] - p[1])) > 0
        push!(cells, Ferrite.Tetrahedron(positive ? t : (t[1], t[3], t[2], t[4])))
    end
    grid = Grid(cells, nodes)
    for (name, axis, value) in (("left", 1, 0.0), ("right", 1, 1.0), ("front", 2, 0.0), ("back", 2, 1.0), ("bottom", 3, 0.0), ("top", 3, 1.0))
        addfacetset!(grid, name, X -> X[axis] == value)
    end
    return grid
end

@testset "periodic faces triangulated along different diagonals are rejected" begin
    function steady(grid)
        Ω = MeshDomain(grid)
        left, right, front, back, bottom, top = boundarypiece.(Ω, (:left, :right, :front, :back, :bottom, :top))
        bcs = [u(left) ~ u(right), [u(Γ) ~ y * z for Γ in (front, back, bottom, top)]...]
        @named pdesys = PDESystem([Dx(Dx(u(x, y, z))) + Dy(Dy(u(x, y, z))) + Dz(Dz(u(x, y, z))) ~ 0], bcs, [(x, y, z) ∈ Ω], [x, y, z], [u(x, y, z)], [])
        return discretize(pdesys, FEMDiscretization(Ω))
    end
    @test_throws "periodic pieces Γ_left and Γ_right of u do not match under a translation: the facet of Γ_left" steady(five_tetrahedra_cube())
    @test_throws "triangulated differently" steady(five_tetrahedra_cube())
    @test steady(generate_grid(Tetrahedron, (2, 2, 2), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))) isa SciMLBase.NonlinearProblem
    @test steady(generate_grid(Hexahedron, (2, 2, 2), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))) isa SciMLBase.NonlinearProblem
end

periodic_square(corner, size; n = (1, 1)) = generate_grid(Quadrilateral, n, Vec(corner...), Vec((corner .+ size)...))

# Integer keys of absolute coordinates would overflow at x0 = 1e14, so the matching uses coordinates relative to the mesh.
@testset "periodic matching does not depend on the distance of the mesh from the origin" begin
    ties(corner) = FEMSpace(periodic_square(corner, (1.0, 1.0); n = (3, 3)), [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "left", "right")]).ties
    @test !isempty(ties((0.0, 0.0)))
    @test ties((1.0e14, 0.0)) == ties((0.0, 0.0))
    function solution(x0)
        Ω = MeshDomain(periodic_square((x0, 0.0), (1.0, 1.0); n = (2, 2)))
        left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        @named pdesys = PDESystem([Dy(Dy(u(x, y))) - u(x, y) ~ sin(pi * y) * (-pi^2 - 1)], [u(left) ~ u(right), u(bottom) ~ 0, u(top) ~ 0], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
        return solve(discretize(pdesys, FEMDiscretization(Ω)), NewtonRaphson(); abstol = 1.0e-12)[u(x, y)]
    end
    @test solution(1.0e14) ≈ solution(0.0) atol = 1.0e-8
end

# The tolerance follows each axis, so an exact translation of 1e-7 along a thin axis is kept.
@testset "periodic matching on a thin domain, and coincident pieces" begin
    grid = periodic_square((0.0, 0.0), (1.0e-7, 1.0); n = (1, 4))
    space = FEMSpace(grid, [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "left", "right")])
    @test length(space.ties) == 5
    # Two disjoint pieces with the same centroid: the lower left with the upper right, and the upper left with the lower right.
    square = periodic_square((0.0, 0.0), (1.0, 1.0); n = (2, 2))
    addfacetset!(square, "diagonal_a", X -> (X[1] == 0.0 && X[2] <= 0.5) || (X[1] == 1.0 && X[2] >= 0.5))
    addfacetset!(square, "diagonal_b", X -> (X[1] == 0.0 && X[2] >= 0.5) || (X[1] == 1.0 && X[2] <= 0.5))
    @test_throws "periodic pieces Γ_diagonal_a and Γ_diagonal_b of u coincide" FEMSpace(square, [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "diagonal_a", "diagonal_b")])
end

function periodic_pair(grid)
    Ω = MeshDomain(grid)
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named steady = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], [u(left) ~ u(right), u(bottom) ~ 0, u(top) ~ 0], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    @named heat = PDESystem(
        [Differential(t)(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
    )
    return () -> discretize(steady, FEMDiscretization(Ω)), () -> FiniteElementMethod.operators(heat, FEMDiscretization(Ω, t))
end

const SIDES = Dict("bottom" => [Ferrite.FacetIndex(1, 1)], "right" => [Ferrite.FacetIndex(1, 2)], "top" => [Ferrite.FacetIndex(1, 3)], "left" => [Ferrite.FacetIndex(1, 4)])

# One Q1 cell of height 1 with its left side at x0 and its right vertices at x0 + bottom and x0 + top.
quadrilateral(x0, bottom, top) = Grid([Quadrilateral((1, 2, 3, 4))], Node.([Vec(x0, 0.0), Vec(x0 + bottom, 0.0), Vec(x0 + top, 1.0), Vec(x0, 1.0)]); facetsets = SIDES)

const MISMATCH = "periodic pieces Γ_left and Γ_right of u do not match under a translation"

# The tolerance follows the extent of the mesh, not the size of its coordinates: a cell four ulps wide is matched, a skew of two ulps refused.
@testset "periodic matching of a mesh a few ulps wide far from the origin" begin
    x0 = 1.0e12
    w = 4 * eps(x0)
    ties(grid) = FEMSpace(grid, [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "left", "right")]).ties
    @test length(ties(quadrilateral(0.0, w, w))) == 2
    @test ties(quadrilateral(x0, w, w)) == ties(quadrilateral(0.0, w, w))
    for corner in (0.0, x0)
        steady, ops = periodic_pair(quadrilateral(corner, w, w))
        @test steady() isa SciMLBase.NonlinearProblem
        @test size(ops().P) == (4, 2)
    end
    steady, ops = periodic_pair(quadrilateral(x0, 2w, 3w))
    @test_throws MISMATCH steady()
    @test_throws MISMATCH ops()
    # A slanted cell with a Q2 field: the midside dofs are interpolated from the vertices, which far from the origin would round
    # them to whole ulps and break the exact translation of 5 ulps.
    e = eps(x0)
    slanted(x) = Grid([Quadrilateral((1, 2, 3, 4))], Node.([Vec(x, 0.0), Vec(x + 5e, 0.0), Vec(x + 6e, 100.0), Vec(x + e, 100.0)]); facetsets = SIDES)
    q2ties(grid) = FEMSpace(grid, [:u => Lagrange{RefQuadrilateral, 2}()]; periodic = [(:u, "left", "right")]).ties
    @test length(q2ties(slanted(0.0))) == 3
    @test q2ties(slanted(x0)) == q2ties(slanted(0.0))
    # A field restricted to a cell 1e-14 wide at x = 1, in a mesh that reaches x = 0: positions are measured from the corner of
    # the pair's cells, whose extent gives the tolerance, not from the corner of the mesh.
    d = 1.0e-14
    narrow = Grid(
        [Quadrilateral((1, 2, 5, 4)), Quadrilateral((2, 3, 6, 5))], Node.([Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(1.0 + d, 0.0), Vec(0.0, 1.0), Vec(1.0, 1.0), Vec(1.0 + d, 1.0)]);
        cellsets = Dict("narrow" => Set([2])), facetsets = Dict("bottom" => [Ferrite.FacetIndex(2, 1)], "top" => [Ferrite.FacetIndex(2, 3)])
    )
    @test length(FEMSpace(narrow, [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "bottom", "top")], regions = Dict(:u => "narrow")).ties) == 2
end

# A cell at -2^54 rounds 0.5 away in any frame that contains it; the pair is matched in the frame of its own cells.
@testset "a distant cell does not erase the separation of a periodic pair's nodes" begin
    @variables v(..)
    function distant(x0)
        nodes = Node.([Vec(0.0, 0.0), Vec(0.5, 0.0), Vec(0.5, 2.0^40), Vec(0.0, 2.0^40), Vec(x0, 0.0), Vec(x0 + 16, 0.0), Vec(x0 + 16, 1.0), Vec(x0, 1.0)])
        return Grid(
            [Quadrilateral((1, 2, 3, 4)), Quadrilateral((5, 6, 7, 8))], nodes;
            cellsets = Dict("rectangle" => Set([1]), "far" => Set([2])), facetsets = Dict(k => f for (k, f) in SIDES if k in ("bottom", "top"))
        )
    end
    ip = Lagrange{RefQuadrilateral, 1}()
    layouts = [([:u => ip], Dict()), ([:u => ip, :v => ip], Dict(:u => "rectangle", :v => "far")), ([:u => ip], Dict(:u => "rectangle"))]
    for x0 in (-32.0, -2.0^54), (fields, regions) in layouts
        @test FEMSpace(distant(x0), fields; periodic = [(:u, "bottom", "top")], regions).ties == [3 => 2, 4 => 1]
    end
    for x0 in (-32.0, -2.0^54), shared in (true, false)
        Ω = MeshDomain(distant(x0))
        bottom, top = boundarypiece.(Ω, (:bottom, :top))
        Dt = Differential(t)
        equations = [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))]
        bcs = [u(t, bottom) ~ u(t, top), u(0, x, y) ~ 0]
        shared || (push!(equations, Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y)))); push!(bcs, v(0, x, y) ~ 0))
        regions = shared ? Dict() : Dict(u => Region(Ω, :rectangle), v => Region(Ω, :far))
        @named heat = PDESystem(equations, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], shared ? [u(t, x, y)] : [u(t, x, y), v(t, x, y)], [])
        disc = FEMDiscretization(Ω, t; regions, boundary_check = :off)
        @test discretize(heat, disc) isa SciMLBase.DAEProblem
        ops = FiniteElementMethod.operators(heat, disc)
        coords = FiniteElementMethod._dof_coordinates(ops.metadata.space)
        @test all((ops.P[d, :] == ops.P[e, :]) == (coords[d][1] == coords[e][1]) for d in 1:4, e in 1:4)
    end
end

# Facets of a piece outside the field's region carry none of its dofs, so their geometry is not compared.
@testset "periodic facets are compared only inside the field's region" begin
    grid = Grid(
        [Quadrilateral((1, 2, 5, 4)), Quadrilateral((2, 3, 6, 5))], Node.([Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(2.0, 0.0), Vec(0.0, 1.0), Vec(1.0, 1.0), Vec(2.3, 1.0)]);
        cellsets = Dict("left" => Set([1])), facetsets = Dict("bottom" => [Ferrite.FacetIndex(1, 1), Ferrite.FacetIndex(2, 1)], "top" => [Ferrite.FacetIndex(1, 3), Ferrite.FacetIndex(2, 3)])
    )
    space(regions) = FEMSpace(grid, [:u => Lagrange{RefQuadrilateral, 1}()]; periodic = [(:u, "bottom", "top")], regions)
    @test space(Dict(:u => "left")).ties == [3 => 2, 4 => 1]
    @test_throws "periodic pieces Γ_bottom and Γ_top of u do not match under a translation" space(Dict())
end

@testset "periodic pieces must be translates in the mesh geometry, not only at the field's dofs" begin
    straight = generate_grid(Ferrite.QuadraticQuadrilateral, (1, 1), Vec(0.0, 0.0), Vec(1.0, 1.0))
    nodes = map(n -> n.x == Vec(0.0, 0.5) ? Node(Vec(0.15, 0.5)) : n, straight.nodes)
    steady, ops = periodic_pair(straight)
    @test steady() isa SciMLBase.NonlinearProblem
    @test size(ops().P) == (4, 2)
    steady, ops = periodic_pair(Grid(straight.cells, nodes; facetsets = straight.facetsets))
    @test_throws "$MISMATCH: the facet of Γ_left" steady()
    @test_throws "$MISMATCH: the facet of Γ_left" ops()
end

@testset "the periodic Jacobian scatter is type-stable and allocation-free" begin
    out, vals = zeros(6), collect(1.0:6.0)
    from, to = [1, 3, 5], [2, 4, 6]
    @inferred FiniteElementMethod.scatter_ties!(out, vals, from, to)
    @test out == [0.0, 1.0, 0.0, 3.0, 0.0, 5.0]
    FiniteElementMethod.scatter_ties!(out, vals, from, to)
    @test (@allocated FiniteElementMethod.scatter_ties!(out, vals, from, to)) == 0
    # A Jacobian with another value type (here Float32) is accepted too: inferred, allocation-free, the same scatter.
    out32, vals32 = zeros(Float32, 6), Float32.(vals)
    @inferred FiniteElementMethod.scatter_ties!(out32, vals32, from, to)
    @test out32 == Float32[0, 1, 0, 3, 0, 5]
    FiniteElementMethod.scatter_ties!(out32, vals32, from, to)
    @test (@allocated FiniteElementMethod.scatter_ties!(out32, vals32, from, to)) == 0
    # Mixed value types: the sum is formed in the wider type and converted on assignment, so a small difference survives.
    mixed = Float32[1.0, 0.0]
    FiniteElementMethod.scatter_ties!(mixed, [-1.0 + 2.0^-25], [1], [1])
    @test mixed[1] == Float32(2.0^-25)
    @inferred FiniteElementMethod.scatter_ties!(zeros(Float32, 6), vals, from, to)
    @test (@allocated FiniteElementMethod.scatter_ties!(out32, vals, from, to)) == 0
    # A complex destination (a complex Jacobian with real state values).
    outc = zeros(ComplexF64, 6)
    @inferred FiniteElementMethod.scatter_ties!(outc, vals, from, to)
    @test outc == ComplexF64[0, 1, 0, 3, 0, 5]
    @test (@allocated FiniteElementMethod.scatter_ties!(outc, vals, from, to)) == 0
end

@testset "operators with every periodic node prescribed" begin
    Ω = MeshDomain(periodic_square((0.0, 0.0), (1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    Dt = Differential(t)
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
    )
    ops = FiniteElementMethod.operators(pdesys, FEMDiscretization(Ω, t))
    @test sort(ops.prescribed) == 1:4
    @test size(ops.P) == (4, 2)
    @test length(ops.tied) == 2 && issorted(ops.tied)
    @test all(sum(ops.P; dims = 2) .== 1)
    @test !any(d -> d in ops.tied, [argmax(Vector(ops.P[:, c])) for c in 1:2])
end
