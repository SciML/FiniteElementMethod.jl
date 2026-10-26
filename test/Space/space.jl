using FiniteElementMethod
using FiniteElementMethod: FEMSpace, rectangle_grid, field_range, prescribed_dofs, dirichlet_values!,
    check_facetsets, _meshdim, _facetset_names, _cellset_names, _contains_point
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
    @test_logs (:warn, r"middle.*interior") check_facetsets(grid, [:middle])
    @test_logs check_facetsets(grid, [:left, :top])
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
