using FiniteElementMethod
using FiniteElementMethod: FEMSpace, rectangle_grid, field_range, prescribed_dofs, dirichlet_values!,
    check_facetsets, _meshdim, _facetset_names, _cellset_names, _contains_point, _none_shared
using Ferrite
using PDEBase: PDEBase
using Test

include(joinpath(@__DIR__, "..", "shared", "lshape.jl"))

errmsg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

facet_coordinates(grid, fi) = [get_node_coordinate(grid, n) for n in Ferrite.facets(getcells(grid, fi[1]))[fi[2]]]

# A triangle on its own reference shape that gives its edges and facets per cell, with no `reference_edges`.
struct RefCustomTriangle <: Ferrite.AbstractRefShape{2} end
struct CustomTriangle <: Ferrite.AbstractCell{RefCustomTriangle}
    nodes::NTuple{3, Int}
end
Ferrite.vertices(c::CustomTriangle) = c.nodes
Ferrite.edges(c::CustomTriangle) = ((c.nodes[1], c.nodes[2]), (c.nodes[2], c.nodes[3]), (c.nodes[3], c.nodes[1]))
Ferrite.faces(c::CustomTriangle) = (c.nodes,)
Ferrite.reference_facets(::CustomTriangle) = ((1, 2), (2, 3), (3, 1))

# A cell with one pentagonal facet, and a pentagonal submesh cell.
struct PentagonParent <: Ferrite.AbstractCell{RefPyramid}
    nodes::NTuple{5, Int}
end
Ferrite.vertices(c::PentagonParent) = c.nodes
Ferrite.reference_facets(::PentagonParent) = ((4, 3, 2, 1, 5),)
Ferrite.nfacets(::PentagonParent) = 1
struct Pentagon <: Ferrite.AbstractCell{RefQuadrilateral}
    nodes::NTuple{5, Int}
end
Ferrite.vertices(c::Pentagon) = c.nodes

const P1 = Lagrange{RefTriangle, 1}()
const P2 = Lagrange{RefTriangle, 2}()

@testset "rectangle_grid" begin
    grid = rectangle_grid(Quadrilateral, (4, 3))
    @test _meshdim(grid) == 2
    @test _facetset_names(grid) == [:bottom, :left, :right, :top]
    xs = [n.x for n in grid.nodes]
    @test extrema(first.(xs)) == (0.0, 1.0)
    @test extrema(last.(xs)) == (0.0, 1.0)
    faces = (bottom = (2, 0.0), right = (1, 1.0), top = (2, 1.0), left = (1, 0.0))
    for (name, (axis, value)) in pairs(faces)
        set = getfacetset(grid, String(name))
        @test length(set) == (axis == 1 ? 3 : 4)
        for fi in set
            @test all(x[axis] ≈ value for x in facet_coordinates(grid, fi))
        end
    end

    tri = rectangle_grid(Triangle, (2, 2); lo = (1, 2), hi = (3, 5))
    xs = [n.x for n in tri.nodes]
    @test extrema(first.(xs)) == (1.0, 3.0)
    @test extrema(last.(xs)) == (2.0, 5.0)

    line = rectangle_grid(Line, (5,))
    @test _meshdim(line) == 1
    @test _facetset_names(line) == [:x1_hi, :x1_lo]
    @test only(facet_coordinates(line, only(getfacetset(line, "x1_hi"))))[1] == 1.0

    hex = rectangle_grid(Hexahedron, (2, 2, 2))
    @test _meshdim(hex) == 3
    @test _facetset_names(hex) == [:x1_hi, :x1_lo, :x2_hi, :x2_lo, :x3_hi, :x3_lo]
    for (name, axis, value) in ((:x1_lo, 1, 0.0), (:x2_hi, 2, 1.0), (:x3_lo, 3, 0.0))
        for fi in getfacetset(hex, String(name))
            @test all(x[axis] ≈ value for x in facet_coordinates(hex, fi))
        end
    end

    @test _contains_point(grid, (0.5, 0.5))
    @test !_contains_point(grid, (1.5, 0.5))
    @test !_contains_point(grid, (NaN, 0.5))
    @test _cellset_names(grid) == Symbol[]

    # The 3 nearest nodes of (4.9, 4.9) all belong to cells other than the one containing it.
    graded = Grid(
        Triangle.([(1, 2, 3), (2, 4, 3), (2, 5, 4), (4, 5, 6), (3, 4, 6)]),
        Node.([Vec(0.0, 0.0), Vec(10.0, 0.0), Vec(0.0, 10.0), Vec(5.1, 5.1), Vec(5.2, 5.1), Vec(5.1, 5.2)])
    )
    @test _contains_point(graded, (4.9, 4.9))
    @test !_contains_point(graded, (6.0, 6.0))
    @test !_contains_point(graded, (11.0, 1.0))
end

@testset "FieldWise blocks in declaration order" begin
    grid = rectangle_grid(Triangle, (4, 4))
    nn = getnnodes(grid)
    space = FEMSpace(grid, [:u => P1, :v => P2])
    @test space isa PDEBase.AbstractDiscreteSpace
    @test space.grid === grid
    @test space.dh isa DofHandler
    @test space.fields == [:u, :v]
    @test space.pieces == Dict(n => String(n) for n in (:bottom, :left, :right, :top))
    @test field_range(space, :u) == 1:nn
    @test field_range(space, :v) == (nn + 1):ndofs(space)
    for cell in 1:getncells(grid)
        dofs = celldofs(space.dh, cell)
        @test all(in(field_range(space, :u)), dofs[dof_range(space.dh, :u)])
        @test all(in(field_range(space, :v)), dofs[dof_range(space.dh, :v)])
    end

    reversed = FEMSpace(grid, [:v => P2, :u => P1])
    nv = length(field_range(space, :v))
    @test field_range(reversed, :v) == 1:nv
    @test field_range(reversed, :u) == (nv + 1):(nv + nn)
    @test_throws ArgumentError field_range(space, :w)

    @test getnquadpoints(space.cellvalues.u) == getnquadpoints(space.cellvalues.v) ==
        length(QuadratureRule{RefTriangle}(4).weights)
    @test getnquadpoints(space.facetvalues.v) == getnquadpoints(FacetQuadratureRule{RefTriangle}(4), 1)
    coarse = FEMSpace(grid, [:u => P1]; quadrature = 1)
    @test getnquadpoints(coarse.cellvalues.u) == 1
end

@testset "fields on regions and Hessians" begin
    grid = rectangle_grid(Triangle, (4, 4))
    Ferrite.addcellset!(grid, "left_half", x -> x[1] <= 0.5 + 1.0e-12)
    grid.cellsets["nothing"] = empty(grid.cellsets["left_half"])   # addcellset! warns on an empty set
    space = FEMSpace(grid, [:u => P1, :w => P2]; regions = Dict(:w => "left_half"), hessians = true)
    @test length(space.dh.subdofhandlers) == 2
    @test field_range(space, :u) == 1:getnnodes(grid)
    @test length(field_range(space, :w)) == 5 * 9                     # P2 nodes of the 2 × 4 left cells
    @test last(field_range(space, :w)) == ndofs(space)
    @test Ferrite.shape_hessian(space.cellvalues.w, 1, 1) isa Ferrite.Tensor{2, 2}
    @test_throws ArgumentError("region given for unknown field :v; fields are [:u, :w]") FEMSpace(
        grid, [:u => P1, :w => P2]; regions = Dict(:v => "left_half")
    )
    @test_throws ArgumentError(
        "unknown cell set :right_half for the region of field w; the mesh has cell sets :left_half, :nothing"
    ) FEMSpace(grid, [:u => P1, :w => P2]; regions = Dict(:w => "right_half"))
    @test_throws ArgumentError("the region :nothing of field w has no cells") FEMSpace(
        grid, [:u => P1, :w => P2]; regions = Dict(:w => "nothing")
    )
end

@testset "fields on disjoint regions keep declaration order" begin
    # u on the left half, v on the right half, w everywhere: the sub-handlers hold [u, w] and [v, w], so Ferrite
    # lists the fields as [u, w, v], yet the blocks must follow the declaration order u, v, w.
    grid = rectangle_grid(Triangle, (2, 2))
    Ferrite.addcellset!(grid, "left_half", x -> x[1] <= 0.5 + 1.0e-12)
    Ferrite.addcellset!(grid, "right_half", x -> x[1] >= 0.5 - 1.0e-12)
    space = FEMSpace(grid, [:u => P1, :v => P1, :w => P1]; regions = Dict(:u => "left_half", :v => "right_half"))
    @test field_range(space, :u) == 1:6
    @test field_range(space, :v) == 7:12
    @test field_range(space, :w) == 13:21
    for sdh in space.dh.subdofhandlers, cell in sdh.cellset, f in sdh.field_names
        @test all(in(field_range(space, f)), celldofs(space.dh, cell)[dof_range(sdh, f)])
    end
end

@testset "shared sparsity pattern" begin
    grid = rectangle_grid(Triangle, (4, 4))
    space = FEMSpace(grid, [:u => P1, :v => P2]; dirichlet = [(:u, :left, (x, t) -> 0.0)])
    K, M = copy(space.pattern), copy(space.pattern)
    Kasm, Masm = start_assemble(K), start_assemble(M)
    nd = ndofs_per_cell(space.dh)
    Ke, Me = zeros(nd, nd), zeros(nd, nd)
    for cell in CellIterator(space.dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        for field in (:u, :v)
            cv = space.cellvalues[field]
            r = dof_range(space.dh, field)
            Ferrite.reinit!(cv, cell)
            for q in 1:getnquadpoints(cv), (i, I) in enumerate(r), (j, J) in enumerate(r)
                dΩ = getdetJdV(cv, q)
                Ke[I, J] += shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j) * dΩ
                Me[I, J] += shape_value(cv, q, i) * shape_value(cv, q, j) * dΩ
            end
        end
        assemble!(Kasm, celldofs(cell), Ke)
        assemble!(Masm, celldofs(cell), Me)
    end
    @test M.rowval == K.rowval && M.colptr == K.colptr
    for field in (:u, :v)
        r = field_range(space, field)
        @test sum(M[r, r]) ≈ 1
        @test maximum(abs, K[r, r] * ones(length(r))) < 1.0e-12
    end
    pd = prescribed_dofs(space)
    @test !isempty(pd)
    @test all(d -> d in K.rowval[K.colptr[d]:(K.colptr[d + 1] - 1)], pd)
end

@testset "Dirichlet values" begin
    grid = rectangle_grid(Triangle, (4, 4))
    gu = (x, t) -> 2.0
    gv = (x, t) -> sin(t) * (1 + x[1])
    space = FEMSpace(grid, [:u => P1, :v => P2]; dirichlet = [(:u, :left, gu), (:v, "bottom", gv)])
    pd = prescribed_dofs(space)
    @test count(in(field_range(space, :u)), pd) == 5
    @test count(in(field_range(space, :v)), pd) == 9
    @test length(pd) == 14
    vals = zeros(length(pd))
    for t in (0.0, 0.7, 2.0)
        expected = zeros(ndofs(space))
        apply_analytical!(expected, space.dh, :u, x -> gu(x, t))
        apply_analytical!(expected, space.dh, :v, x -> gv(x, t))
        @test dirichlet_values!(vals, space, t) === vals
        @test vals ≈ expected[pd]
    end

    # Data defined only from the initial time on: the constructor must not evaluate it at t = 0.
    late = [(:u, :left, (x, t) -> sqrt(t - 0.5))]
    @test_throws DomainError FEMSpace(grid, [:u => P1]; dirichlet = late)
    space = FEMSpace(grid, [:u => P1]; dirichlet = late, t0 = 1)
    @test dirichlet_values!(zeros(5), space, 1.0) ≈ fill(sqrt(0.5), 5)
    @test dirichlet_values!(zeros(5), space, 2.0) ≈ fill(sqrt(1.5), 5)

    # Time-independent one-argument data, with and without t0.
    for t0 in (0.0, 1)
        space = FEMSpace(grid, [:u => P1]; dirichlet = [(:u, :left, x -> 1.0 + x[2])], t0)
        expected = zeros(ndofs(space))
        apply_analytical!(expected, space.dh, :u, x -> 1.0 + x[2])
        @test dirichlet_values!(zeros(5), space, 3.0) ≈ expected[prescribed_dofs(space)]
    end
end

@testset "facet-set validation" begin
    grid = rectangle_grid(Triangle, (4, 4))
    msg = errmsg(() -> FEMSpace(grid, [:u => P1]; dirichlet = [(:u, :inlet, (x, t) -> 0.0)]))
    @test occursin(":inlet", msg)
    @test occursin(":bottom, :left, :right, :top", msg)
    @test occursin(":w", errmsg(() -> FEMSpace(grid, [:u => P1]; dirichlet = [(:w, :left, (x, t) -> 0.0)])))

    addfacetset!(grid, "lower_left", x -> x[1] ≈ 0 && x[2] <= 0.5)
    msg = errmsg(() -> check_facetsets(grid, [:left, :bottom, :lower_left]))
    @test occursin("2 facets", msg)
    @test occursin(":left", msg) && occursin(":lower_left", msg)
    @test_throws ArgumentError FEMSpace(
        grid, [:u => P1, :v => P1];
        dirichlet = [(:u, :left, (x, t) -> 0.0), (:v, :lower_left, (x, t) -> 0.0)]
    )
    @test check_facetsets(grid, [:left, :left, :bottom]) === nothing

    addfacetset!(grid, "middle", x -> x[1] ≈ 0.5)
    @test_throws ArgumentError check_facetsets(grid, [:middle])
    @test occursin("not on the boundary", errmsg(() -> check_facetsets(grid, [:middle])))
    @test_logs check_facetsets(grid, [:left, :top])
end

@testset "interior facets are those of ExclusiveTopology" begin
    box(CT, dim) = generate_grid(CT, ntuple(Returns(2), dim))
    nodes = [Node(x) for x in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)]]
    allfacets(grid) = [FacetIndex(c, f) for c in 1:getncells(grid) for f in 1:nfacets(getcells(grid, c))]
    # Whether `check_facetsets` accepts each facet on its own, against the neighbours that `ExclusiveTopology` finds.
    function agrees(grid)
        topology = ExclusiveTopology(grid)
        return all(allfacets(grid)) do fi
            addfacetset!(grid, "one", [fi])
            msg = errmsg(() -> check_facetsets(grid, [:one]))
            delete!(grid.facetsets, "one")
            isempty(getneighborhood(topology, grid, fi)) ? msg == "" : occursin("not on the boundary", msg)
        end
    end
    function onepass(grid, facets)
        addfacetset!(grid, "probe", facets)
        proved = _none_shared(grid, [:probe])
        delete!(grid.facetsets, "probe")
        return proved
    end

    # Cell types whose boundary facets are proved unshared in one pass.
    for (CT, dim) in (
            (Line, 1), (QuadraticLine, 1), (Triangle, 2), (QuadraticTriangle, 2), (Quadrilateral, 2),
            (QuadraticQuadrilateral, 2), (Ferrite.SerendipityQuadraticQuadrilateral, 2), (Tetrahedron, 3),
        )
        grid = CT === Ferrite.SerendipityQuadraticQuadrilateral ?
            Grid([CT((1, 2, 3, 4, 5, 6, 7, 8))], [Node((0.0, 0.0)) for _ in 1:8]) : box(CT, dim)
        @test agrees(grid)
        topology = ExclusiveTopology(grid)
        boundary = filter(fi -> isempty(getneighborhood(topology, grid, fi)), allfacets(grid))
        @test onepass(grid, boundary)
    end
    # Every other cell type is left to `ExclusiveTopology`.
    for grid in (
            box(Hexahedron, 3), box(Wedge, 3), box(Pyramid, 3),
            Grid(Ferrite.AbstractCell[Triangle((1, 2, 3)), Triangle((1, 3, 4)), Quadrilateral((2, 5, 6, 3))], nodes),
            Grid(Ferrite.AbstractCell[Triangle((1, 2, 3)), Triangle((1, 3, 4)), Line((1, 3)), Line((1, 2))], nodes),
        )
        @test agrees(grid)
        @test !onepass(grid, [FacetIndex(1, 1)])
    end

    grid = box(Tetrahedron, 3)
    addfacetset!(grid, "inside", [FacetIndex(1, 3)])
    @test errmsg(() -> check_facetsets(grid, [:left, :inside])) ==
        "ArgumentError: facet set :inside is not on the boundary: it contains 1 interior facets " *
        "(shared by two cells; first: cell 1, local facet 3)"
    # An invalid facet in a later set does not hide the interior facet of an earlier one.
    addfacetset!(grid, "invalid", [FacetIndex(getncells(grid) + 1, 1)])
    @test_throws "not on the boundary" check_facetsets(grid, [:inside, :invalid])
    @test_throws BoundsError check_facetsets(grid, [:left, :invalid])

    # A tetrahedron under a hexahedron, its face covering half of the hexahedron's base: the face is not on the
    # boundary, although the two cells have no facet with the same corners.
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, -1)]
    cells = Ferrite.AbstractCell[Hexahedron((1, 2, 3, 4, 5, 6, 7, 8)), Tetrahedron((1, 2, 3, 9))]
    hanging = Grid(cells, [Node(Float64.(x)) for x in corners])
    addfacetset!(hanging, "half", [FacetIndex(2, 1)])
    @test_throws "not on the boundary" check_facetsets(hanging, [:half])

    # The same between two pyramids: a triangular side of the second covers half of the base of the first.
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (0, 0, -1), (1, 0, -1)]
    pyramids = Grid([Pyramid((1, 2, 3, 4, 5)), Pyramid((1, 2, 6, 7, 3))], [Node(Float64.(x)) for x in corners])
    addfacetset!(pyramids, "half", [FacetIndex(2, 2)])
    @test_throws "not on the boundary" check_facetsets(pyramids, [:half])

    # Two hexahedra whose faces share three corners but not the fourth.
    corners = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        (0, 2, 0), (0, 0, -1), (1, 0, -1), (1, 1, -1), (0, 1, -1),
    ]
    hexes = Grid([Hexahedron((1, 2, 3, 4, 5, 6, 7, 8)), Hexahedron((10, 11, 12, 13, 1, 2, 3, 9))], [Node(Float64.(x)) for x in corners])
    addfacetset!(hexes, "base", [FacetIndex(1, 1)])
    @test_throws "not on the boundary" check_facetsets(hexes, [:base])

    # Broken meshes of the cell types counted in one pass: duplicated cells, and cells sharing two edges.
    for broken in (
            Grid([Triangle((1, 2, 3)), Triangle((1, 2, 3))], nodes),
            Grid([Quadrilateral((1, 2, 3, 4)), Quadrilateral((1, 2, 3, 5))], nodes),
            Grid([Tetrahedron((1, 2, 3, 4)), Tetrahedron((1, 2, 3, 4))], box(Tetrahedron, 3).nodes),
            Grid([Line((1, 1)), Line((2, 3))], box(Line, 1).nodes),
        )
        @test agrees(broken)
    end

    # An embedded grid is rejected, as by `ExclusiveTopology`.
    surface = Grid([Triangle((1, 2, 3))], [Node((n.x[1], n.x[2], 0.0)) for n in nodes])
    addfacetset!(surface, "edge", [FacetIndex(1, 1)])
    @test_throws "Purely embedded" check_facetsets(surface, [:edge])
end

@testset "custom cell types" begin
    # No cell-type metadata beyond what `ExclusiveTopology` needs is asked of a custom cell.
    grid = Grid([CustomTriangle((1, 2, 3))], [Node((0.0, 0.0)), Node((1.0, 0.0)), Node((0.0, 1.0))])
    addfacetset!(grid, "edge", [FacetIndex(1, 1)])
    @test check_facetsets(grid, [:edge]) === nothing

    # A facet is identified by all of its corners, not four of them.
    nodes = [Node((cos(2pi * k / 5), sin(2pi * k / 5), 0.0)) for k in 0:4]
    grid = Grid([PentagonParent((1, 2, 3, 4, 5))], nodes)
    @test_throws "match no facet" add_boundary_pieces!(grid, :four => Grid([Quadrilateral((1, 2, 3, 4))], nodes[1:4]))
    @test !haskey(grid.facetsets, "four")
    add_boundary_pieces!(grid, :five => Grid([Pentagon((1, 5, 4, 3, 2))], nodes))
    @test collect(getfacetset(grid, "five")) == [FacetIndex(1, 1)]
end

@testset "Gmsh L-shape" begin
    grid = lshape_grid()
    @test _meshdim(grid) == 2
    @test _facetset_names(grid) == [:inlet, :outlet, :wall]
    @test :domain in _cellset_names(grid)
    @test !haskey(grid.vertexsets, "corner")
    @test !haskey(grid.nodesets, "corner")
    @test _contains_point(grid, (0.5, 0.5))
    @test _contains_point(grid, (1.5, 0.5))
    @test _contains_point(grid, (0.5, 1.5))
    @test !_contains_point(grid, (1.5, 1.5))
    @test !_contains_point(grid, (2.5, 0.5))
    @test check_facetsets(grid, [:inlet, :wall, :outlet]) === nothing

    space = FEMSpace(grid, [:u => P1]; dirichlet = [(:u, :inlet, (x, t) -> 1.0 + x[2])])
    pd = prescribed_dofs(space)
    vals = dirichlet_values!(zeros(length(pd)), space, 0.0)
    expected = zeros(ndofs(space))
    apply_analytical!(expected, space.dh, :u, x -> 1.0 + x[2])
    @test vals ≈ expected[pd]
    @test extrema(vals) == (1.0, 3.0)
end
