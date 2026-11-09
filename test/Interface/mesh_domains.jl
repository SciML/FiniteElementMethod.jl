# Mesh domains with named boundary pieces. A `MeshDomain` takes its pieces from the facet sets of the Ferrite
# grid: `boundarypiece(Ω, :name)` is the facet set "name". The sets come from `generate_grid`, from
# `Ferrite.addfacetset!` on a grid you build, from the physical groups of a Gmsh mesh read with FerriteGmsh, or
# from submeshes added with `add_boundary_pieces!`. A Meshes.jl mesh becomes a grid with `ferrite_grid`.
# Each piece then carries its own boundary condition.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: Ferrite, generate_grid, Triangle, Vec
using LinearAlgebra: Diagonal
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh
using Meshes: Meshes
using ModelingToolkit: t_nounits as t

@parameters k α
@variables x y z u(..)
Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)
laplacian(f) = Dx(Dx(f)) + Dy(Dy(f))

# Writes a triangle mesh of the L-shape [0, 2]^2 minus (1, 2]^2 to `path` as a Gmsh .msh file, with the physical
# groups "inlet" (x = 0), "outlet" (x = 2), "wall" (the other four edges) and "domain".
function write_lshape(path; h = 0.5)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lshape")
        corners = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        points = [gmsh.model.geo.addPoint(px, py, 0, h) for (px, py) in corners]
        lines = [gmsh.model.geo.addLine(points[i], points[mod1(i + 1, 6)]) for i in 1:6]
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [lines[6]], -1, "inlet")
        gmsh.model.addPhysicalGroup(1, [lines[2]], -1, "outlet")
        gmsh.model.addPhysicalGroup(1, lines[[1, 3, 4, 5]], -1, "wall")
        gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        Gmsh.finalize()
    end
    return path
end

# PDE:    u_t = Δu
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2; the bottom side is split in two pieces
#         at x = 1/2 with `addfacetset!`, giving five pieces: left, right, top, bottom_left, bottom_right
# BCs:    left and bottom_right: Dirichlet u = x^2 + y^2 + 4t; bottom_left: ∂ₙu = 0; right: Robin ∂ₙu + 3u = 5 + 3y^2 + 12t;
#         top: ∂ₙu = 2
# Exact:  u = x^2 + y^2 + 4t, in the P2 space
# Shows:  pieces made by the user are first-class: one side of the square carries two different conditions.
@testset "a side split into two named pieces carries two different boundary conditions" begin
    grid = generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0))
    delete!(grid.facetsets, "bottom")
    Ferrite.addfacetset!(grid, "bottom_left", p -> p[2] < 1.0e-12 && p[1] < 0.5 + 1.0e-12)
    Ferrite.addfacetset!(grid, "bottom_right", p -> p[2] < 1.0e-12 && p[1] > 0.5 - 1.0e-12)
    Ω = MeshDomain(grid)
    left, right, top, bottom_left, bottom_right = boundarypiece.(Ω, (:left, :right, :top, :bottom_left, :bottom_right))
    @test length(boundarypieces(Ω)) == 5
    bcs = [
        u(t, left) ~ x^2 + y^2 + 4t, u(t, bottom_right) ~ x^2 + y^2 + 4t,
        NormalDerivative(bottom_left)(u(t, bottom_left)) ~ 0,
        NormalDerivative(right)(u(t, right)) + 3u(t, right) ~ 5 + 3y^2 + 12t,
        NormalDerivative(top)(u(t, top)) ~ 2,
        u(0, x, y) ~ x^2 + y^2,
    ]
    @named pdesys = PDESystem([Dt(u(t, x, y)) ~ laplacian(u(t, x, y))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test sol[u(t, x, y)][end, :] ≈ sol[x] .^ 2 .+ sol[y] .^ 2 .+ 4 atol = 1.0e-7
    @test sol[u(t, bottom_left)][end, :] ≈ sol[x][piece_nodes(sol, bottom_left)] .^ 2 .+ 4 atol = 1.0e-7
end

# PDE:    u_t = k Δu with k = 2, and a Robin condition with α = 5 in `ps`
# Domain: the L-shape [0, 2]^2 minus (1, 2]^2, t in (0, 1), a Gmsh triangle mesh (h = 0.5) written to a .msh
#         file and read back with `FerriteGmsh.togrid(path)`; P2; pieces inlet, wall, outlet from physical groups
# BCs:    inlet and wall: Dirichlet u = x^2 + y^2 + 4 k t; outlet (x = 2): k ∂ₙu + α u = 4k + α (4 + y^2 + 4 k t);
#         IC u(0, x, y) = x^2 + y^2
# Exact:  u = x^2 + y^2 + 4 k t, in the P2 space of the straight-sided triangles
# Shows:  a mesh stored in a file becomes a domain whose pieces are the Gmsh physical groups; the missing
#         quadrant reads NaN; and a name that is not a facet set of the mesh is rejected with the valid names.
@testset "an L-shaped Gmsh mesh read from a file is solved with inlet, wall and outlet pieces" begin
    grid = mktempdir() do dir
        FerriteGmsh.togrid(write_lshape(joinpath(dir, "lshape.msh")))
    end
    Ω = MeshDomain(grid)
    inlet, wall, outlet = boundarypiece.(Ω, (:inlet, :wall, :outlet))
    exact = x^2 + y^2 + 4k * t
    bcs = [
        u(t, inlet) ~ exact, u(t, wall) ~ exact,
        k * NormalDerivative(outlet)(u(t, outlet)) + α * u(t, outlet) ~ 4k + α * (4 + y^2 + 4k * t),
        u(0, x, y) ~ x^2 + y^2,
    ]
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ k * laplacian(u(t, x, y))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [k => 2.0, α => 5.0]
    )
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test sol[u(t, x, y)][end, :] ≈ sol[x] .^ 2 .+ sol[y] .^ 2 .+ 8 atol = 1.0e-7
    @test sol(1.0, 0.5, 1.5; dv = u(t, x, y)) ≈ 0.25 + 2.25 + 8 atol = 1.0e-7
    @test isnan(sol(1.0, 1.5, 1.5; dv = u(t, x, y)))
    @test_throws "has no facet set :outflow; valid names are :inlet, :outlet, :wall" boundarypiece(Ω, :outflow)
end

# PDE:    u_t = Δu
# Domain: the square [0, 4]^2 with the square hole [1, 3]^2, t in (0, 1), a Meshes.jl `Pgon` triangulated
#         with `Meshes.discretize` and converted with `ferrite_grid`; P2; the pieces outer and hole are the two
#         rings, each discretized into a mesh of segments and added with `add_boundary_pieces!`
# BCs:    outer and hole: Dirichlet u = x^2 + y^2 + 4t; IC u(0, x, y) = x^2 + y^2
# Exact:  u = x^2 + y^2 + 4t, in the P2 space
# Shows:  a Meshes.jl polygon becomes a domain whose pieces are its rings; the hole reads NaN.
@testset "a Meshes.jl polygon with a hole is solved with its two rings as pieces" begin
    outer = Meshes.Ring((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    hole = Meshes.Ring((1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0))
    grid = ferrite_grid(Meshes.discretize(Meshes.Pgon(outer, hole)))
    add_boundary_pieces!(grid, :outer => Meshes.discretize(outer), :hole => Meshes.discretize(hole))
    Ω = MeshDomain(grid)
    @test length(boundarypieces(Ω)) == 2
    bcs = [[u(t, Γ) ~ x^2 + y^2 + 4t for Γ in boundarypiece.(Ω, (:outer, :hole))]; u(0, x, y) ~ x^2 + y^2]
    @named pdesys = PDESystem([Dt(u(t, x, y)) ~ laplacian(u(t, x, y))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test sol[u(t, x, y)][end, :] ≈ sol[x] .^ 2 .+ sol[y] .^ 2 .+ 4 atol = 1.0e-7
    @test sol(1.0, 0.5, 2.0; dv = u(t, x, y)) ≈ 0.25 + 4 + 4 atol = 1.0e-7
    @test isnan(sol(1.0, 2.0, 2.0; dv = u(t, x, y)))
end

# Solves u_t = Δu on the grid of a Meshes.jl mesh in 1, 2 or 3 dimensions, with u = |x|^2 + 2 dim t on the whole
# boundary as one piece, and returns the largest nodal error at t = 1.
function meshes_heat_error(mesh, ref)
    grid = ferrite_grid(mesh)
    Ferrite.addboundaryfacetset!(grid, Ferrite.ExclusiveTopology(grid), "boundary", Returns(true))
    Ω = MeshDomain(grid)
    X = (x, y, z)[1:Meshes.embeddim(mesh)]
    exact = sum(abs2, X) + 2length(X) * t
    bcs = [u(t, boundarypiece(Ω, :boundary)) ~ exact, u(0, X...) ~ sum(abs2, X)]
    heat = Dt(u(t, X...)) ~ sum(ξ -> Differential(ξ)(Differential(ξ)(u(t, X...))), X)
    @named pdesys = PDESystem([heat], bcs, [X ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, X...], [u(t, X...)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, 2}()))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    return maximum(abs, sol[u(t, X...)][end, :] .- sum(ξ -> sol[ξ] .^ 2, X) .- 2length(X))
end

# The unit box in `dim` dimensions with 2 cells per side, and its mirror image x -> 1 - x, whose cells all have the
# other orientation.
box(dim) = Meshes.CartesianGrid(ntuple(Returns(0.0), dim), ntuple(Returns(1.0), dim); dims = ntuple(Returns(2), dim))
function mirror(mesh)
    dim = Meshes.embeddim(mesh)
    A, b = Diagonal([-1; ones(dim - 1)]), [1; zeros(dim - 1)]
    return Meshes.SimpleMesh(Meshes.vertices(mesh), Meshes.topology(mesh)) |> Meshes.Affine(A, b)
end

# PDE:    u_t = Δu
# Domain: the unit box, t in (0, 1), a Meshes.jl `CartesianGrid` with 2 cells per side, as segments,
#         quadrangles, triangles, hexahedra or tetrahedra (`Meshes.simplexify`), and mirrored with a
#         `Meshes.Affine` map; P2; one piece holds every boundary facet, added with `Ferrite.addboundaryfacetset!`
# BCs:    Dirichlet u = |x|^2 + 2 dim t; IC u(0, x) = |x|^2
# Exact:  u = |x|^2 + 2 dim t, in the P2 space
# Shows:  `ferrite_grid` gives every cell the orientation Ferrite needs, also when Meshes.jl has the other one.
@testset "a Meshes.jl grid of $name is solved exactly" for (name, mesh, ref) in (
        ("mirrored segments", mirror(box(1)), RefLine),
        ("quadrangles", box(2), RefQuadrilateral),
        ("mirrored quadrangles", mirror(box(2)), RefQuadrilateral),
        ("mirrored triangles", Meshes.simplexify(mirror(box(2))), RefTriangle),
        ("hexahedra", box(3), RefHexahedron),
        ("mirrored hexahedra", mirror(box(3)), RefHexahedron),
        ("tetrahedra", Meshes.simplexify(box(3)), RefTetrahedron),
        ("mirrored tetrahedra", Meshes.simplexify(mirror(box(3))), RefTetrahedron),
    )
    @test meshes_heat_error(mesh, ref) < 1.0e-7
end
