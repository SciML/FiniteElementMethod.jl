# Boundary pieces given as submeshes (add_boundary_pieces!) and the completeness rules of the boundary
# conditions (natural, boundary_check), each end to end: PDESystem -> DAEProblem -> solve -> exact solution.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, add_boundary_pieces!
using Ferrite: Ferrite, Triangle, Quadrilateral, Hexahedron, Line, Grid, Node, Vec, addfacetset!, getfacetset

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@variables x y z u(..) v(..)
const Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)
const tol = 1.0e-8
Dn(Γ) = NormalDerivative(Γ)

# A box grid without its facet sets, so that every boundary piece comes from a submesh.
function bare_box(celltype, cells)
    grid = rectangle_grid(celltype, cells)
    empty!(grid.facetsets)
    return grid
end

# The segment from `a` to `b` in `n` Line cells; every other cell is reversed and the nodes are numbered backwards,
# so that neither node ids nor cell orientation agree with the parent's.
function segment(a, b, n)
    nodes = [Node(Vec(Tuple(a .+ (b .- a) .* (k / n)))) for k in n:-1:0]
    cells = [Line(isodd(i) ? (n + 2 - i, n + 1 - i) : (n + 1 - i, n + 2 - i)) for i in 1:n]
    return Grid(cells, nodes)
end

# The unit square [0, 1]² in `m × m` quadrilaterals, placed on the plane `x_axis = value` of the unit cube.
function face(axis, value, m)
    sq = Ferrite.generate_grid(Quadrilateral, (m, m), Vec(0.0, 0.0), Vec(1.0, 1.0))
    place(p) = Vec(Tuple(insert_at(axis, value, p)))
    return Grid(Ferrite.getcells(sq), [Node(place(n.x)) for n in Ferrite.getnodes(sq)])
end
insert_at(axis, value, p) = axis == 1 ? (value, p[1], p[2]) : axis == 2 ? (p[1], value, p[2]) : (p[1], p[2], value)

heat2d = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
ic2d = u(0, x, y) ~ 1 + x
uexact2d(t_, x_, y_) = (1 + x_) * (1 + t_)

function box2d(n = 4)
    grid = bare_box(Triangle, (n, n))
    add_boundary_pieces!(
        grid, :bottom => segment((0.0, 0.0), (1.0, 0.0), n), :right => segment((1.0, 0.0), (1.0, 1.0), n),
        :top => segment((0.0, 1.0), (1.0, 1.0), n), :left => segment((0.0, 0.0), (0.0, 1.0), n)
    )
    return MeshDomain(grid)
end

@testset verbose = true "boundary pieces from submeshes" begin
    @testset "2D box with its four sides as 1D submeshes" begin
        Ω = box2d()
        Γb, Γr, Γt, Γl = boundarypiece.(Ω, (:bottom, :right, :top, :left))
        @test length(getfacetset(Ω.mesh, "bottom")) == 4 && all(fi -> fi[1] ≤ 32, getfacetset(Ω.mesh, "left"))
        eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x)
        # Dirichlet on three sides, Neumann (du/dn = 1 + t) on the right side.
        bcs = [
            u(t, Γl) ~ 1 + t, u(t, Γb) ~ (1 + x) * (1 + t), u(t, Γt) ~ (1 + x) * (1 + t),
            Dn(Γr)(u(t, Γr)) ~ 1 + t, ic2d,
        ]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
    end

    @testset "3D box with its six faces as 2D submeshes" begin
        grid = bare_box(Hexahedron, (2, 2, 2))
        faces = [
            :x1_lo => face(1, 0.0, 2), :x1_hi => face(1, 1.0, 2), :x2_lo => face(2, 0.0, 2),
            :x2_hi => face(2, 1.0, 2), :x3_lo => face(3, 0.0, 2), :x3_hi => face(3, 1.0, 2),
        ]
        add_boundary_pieces!(grid, Dict(faces))
        Ω = MeshDomain(grid)
        @test all(p -> length(getfacetset(grid, String(first(p)))) == 4, faces)
        eq = Dt(u(t, x, y, z)) ~ Dx(Dx(u(t, x, y, z))) + Dy(Dy(u(t, x, y, z))) + Dz(Dz(u(t, x, y, z))) + (1 + x + 2y + 3z)
        bcs = [
            [u(t, boundarypiece(Ω, first(p))) ~ (1 + x + 2y + 3z) * (1 + t) for p in faces];
            u(0, x, y, z) ~ 1 + x + 2y + 3z
        ]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y, z), t, [u(t, x, y, z)]; spaces = Dict(u => Lagrange{RefHexahedron, 1}()))
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, u(t, x, y, z), (t_, x_, y_, z_) -> (1 + x_ + 2y_ + 3z_) * (1 + t_)) <= tol
    end

    @testset "submesh errors" begin
        grid() = bare_box(Triangle, (4, 4))
        # A segment shifted off the boundary: its nodes are not nodes of the parent.
        msg = errmsg(() -> add_boundary_pieces!(grid(), :bottom => segment((0.0, 0.1), (1.0, 0.1), 4)))
        @test occursin("boundary piece :bottom", msg) && occursin("submesh nodes are not within atol", msg)
        @test occursin("first: submesh node", msg)
        # Nodes of the parent, but one cell spans two parent facets.
        msg = errmsg(() -> add_boundary_pieces!(grid(), :bottom => segment((0.0, 0.0), (1.0, 0.0), 2)))
        @test occursin("2 of 2 submesh cells match no facet of the parent mesh", msg)
        @test occursin("first: submesh cell 1 centred at (", msg)
        # The mid-line y = 0.5 lies on mesh edges of the parent, but they are interior facets.
        g = grid()
        msg = errmsg(() -> add_boundary_pieces!(g, :middle => segment((0.0, 0.5), (1.0, 0.5), 4)))
        @test occursin("4 of 4 submesh cells match an interior facet of the parent mesh", msg)
        @test !haskey(g.facetsets, "middle")
        # One bad piece adds none of the pieces, and a piece cannot take the name of a facet set.
        g = grid()
        @test_throws ArgumentError add_boundary_pieces!(
            g, :bottom => segment((0.0, 0.0), (1.0, 0.0), 4), :middle => segment((0.0, 0.5), (1.0, 0.5), 4)
        )
        @test isempty(g.facetsets)
        add_boundary_pieces!(g, :bottom => segment((0.0, 0.0), (1.0, 0.0), 4))
        @test occursin("already has a facet set :bottom", errmsg(() -> add_boundary_pieces!(g, :bottom => segment((0.0, 0.0), (1.0, 0.0), 4))))
        @test_throws ArgumentError("boundary piece names must be unique; got [:a, :a]") add_boundary_pieces!(
            g, :a => segment((0.0, 0.0), (1.0, 0.0), 4), :a => segment((0.0, 1.0), (1.0, 1.0), 4)
        )
        # A 2D cell is not a facet of a 2D mesh; a 3D submesh does not live in the parent's space.
        msg = errmsg(() -> add_boundary_pieces!(grid(), :all => rectangle_grid(Triangle, (1, 1))))
        @test occursin("must have dimension 1", msg)
        msg = errmsg(() -> add_boundary_pieces!(grid(), :face => face(1, 0.0, 2)))
        @test occursin("has 3-dimensional coordinates", msg)
        # A looser tolerance admits a segment that is off by less than it.
        g = grid()
        add_boundary_pieces!(g, :bottom => segment((0.0, 1.0e-4), (1.0, 1.0e-4), 4); atol = 1.0e-3)
        @test length(getfacetset(g, "bottom")) == 4
    end

    @testset "a piece that is not on the boundary is an error" begin
        grid = rectangle_grid(Triangle, (4, 4))
        addfacetset!(grid, "middle", p -> p[2] ≈ 0.5)
        Ω = MeshDomain(grid)
        Γ, Γl, Γr, Γb = boundarypiece.(Ω, (:middle, :left, :right, :bottom))
        bcs = [u(t, Γ) ~ 0, u(t, Γl) ~ 0, u(t, Γr) ~ 0, u(t, Γb) ~ 0, ic2d]
        msg = errmsg(() -> fem_solve([heat2d], bcs, Ω, (x, y), t, [u(t, x, y)]))
        @test occursin("facet set :middle is not on the boundary", msg)
    end

    @testset "two conditions on one facet are an error" begin
        # The bottom side twice, once as a submesh and once as the grid's own facet set.
        grid = rectangle_grid(Triangle, (4, 4))
        add_boundary_pieces!(grid, :floor => segment((0.0, 0.0), (1.0, 0.0), 4))
        Ω = MeshDomain(grid)
        Γb, Γf = boundarypiece.(Ω, (:bottom, :floor))
        bcs = [u(t, Γb) ~ 0, u(t, Γf) ~ 1, ic2d]
        @test occursin("share 4 facets", errmsg(() -> fem_solve([heat2d], bcs, Ω, (x, y), t, [u(t, x, y)])))
    end
end

@testset verbose = true "boundary completeness" begin
    # u* = (1 + x)(1 + t) has no normal derivative on the top side, so leaving it out is the zero-flux condition.
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x)
    function open_top(; disc_kwargs...)
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [u(t, Γl) ~ 1 + t, u(t, Γr) ~ 2 * (1 + t), u(t, Γb) ~ (1 + x) * (1 + t), ic2d]
        return fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)]; disc_kwargs...), Γt
    end

    @testset "a boundary with no condition warns once, naming the field and the piece" begin
        (sol, _), _ = @test_logs (:warn, r"u: 4 of 16 boundary facets have no condition: Γ_top \(4 facets\)") open_top()
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
    end

    @testset "natural = [pieces] declares the zero-flux condition and silences the warning" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        (sol, _), _ = @test_logs min_level = Base.CoreLogging.Warn open_top(; natural = [:top])
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
        # A BoundaryPiece is accepted as well as a name.
        @test_logs min_level = Base.CoreLogging.Warn open_top(; natural = [boundarypiece(Ω, :top)])
        @test_throws ArgumentError("unknown facet set :nosuch; the mesh has facet sets :bottom, :left, :right, :top") FEMDiscretization(Ω, t; natural = [:nosuch])
        @test_throws ArgumentError FEMDiscretization(Ω, t; boundary_check = :loud)
    end

    @testset "an explicit zero Neumann condition is silent too" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [u(t, Γl) ~ 1 + t, u(t, Γr) ~ 2 * (1 + t), u(t, Γb) ~ (1 + x) * (1 + t), Dn(Γt)(u(t, Γt)) ~ 0, ic2d]
        sol, _ = @test_logs min_level = Base.CoreLogging.Warn fem_solve([eq], bcs, Ω, (x, y), t, [u(t, x, y)])
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
    end

    @testset "boundary_check = :error is strict" begin
        msg = errmsg(() -> open_top(; boundary_check = :error))
        @test occursin("u: 4 of 16 boundary facets have no condition: Γ_top (4 facets)", msg)
        @test occursin("natural = [pieces]", msg)
        (sol, _), _ = open_top(; boundary_check = :error, natural = [:top])
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
    end

    @testset "boundary_check = :off skips the check" begin
        (sol, _), _ = @test_logs min_level = Base.CoreLogging.Warn open_top(; boundary_check = :off)
        @test nodal_error(sol, u(t, x, y), uexact2d) <= tol
    end

    @testset "one warning per call covers every field" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        eqs = [
            Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))),
            Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y))),
        ]
        bcs = [u(t, Γl) ~ 0, v(t, Γl) ~ 0, v(t, Γt) ~ 0, u(0, x, y) ~ 0, v(0, x, y) ~ 0]
        @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y), v(t, x, y)], [])
        logs, _ = Test.collect_test_logs(() -> discretize(pdesys, FEMDiscretization(Ω, t)))
        warns = filter(l -> l.level == Base.CoreLogging.Warn, logs)
        @test length(warns) == 1
        @test occursin(r"u: 12 of 16 boundary facets.*Γ_bottom \(4 facets\), Γ_right \(4 facets\), Γ_top \(4 facets\)", warns[1].message)
        @test occursin(r"v: 8 of 16 boundary facets.*Γ_bottom \(4 facets\), Γ_right \(4 facets\)", warns[1].message)
    end

    @testset "a boundary outside every named piece is counted" begin
        grid = bare_box(Triangle, (4, 4))
        add_boundary_pieces!(grid, :left => segment((0.0, 0.0), (0.0, 1.0), 4))
        Ω = MeshDomain(grid)
        @named pdesys = PDESystem([heat2d], [u(t, boundarypiece(Ω, :left)) ~ 0, u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        @test_logs (:warn, r"12 of 16 boundary facets have no condition: 12 facets in no piece") discretize(pdesys, FEMDiscretization(Ω, t))
    end

    @testset "a mesh with no named pieces warns instead of failing" begin
        Ω = MeshDomain(bare_box(Triangle, (4, 4)))
        @named pdesys = PDESystem([heat2d], [u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        @test_logs (:warn, r"16 of 16 boundary facets have no condition: 16 facets in no piece") discretize(pdesys, FEMDiscretization(Ω, t))
    end
end
