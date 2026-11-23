# Reading a Gmsh mesh file straight into a MeshDomain with `gmsh_domain`, as the one-liner for
# `MeshDomain(FerriteGmsh.togrid(path))`.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh
using ModelingToolkit: t_nounits as t

@parameters k α
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

function write_lshape(path; h = 0.5, groups = true, unnamed = false, order = 1)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lshape")
        corners = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        points = [gmsh.model.geo.addPoint(px, py, 0, h) for (px, py) in corners]
        lines = [gmsh.model.geo.addLine(points[i], points[mod1(i + 1, 6)]) for i in 1:6]
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        gmsh.model.geo.synchronize()
        if groups
            gmsh.model.addPhysicalGroup(1, [lines[6]], -1, unnamed ? "" : "inlet")
            gmsh.model.addPhysicalGroup(1, [lines[2]], -1, unnamed ? "" : "outlet")
            gmsh.model.addPhysicalGroup(1, lines[[1, 3, 4, 5]], -1, unnamed ? "" : "wall")
            gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")
        end
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        Gmsh.finalize()
    end
    return path
end

function lshape_solution(Ω)
    inlet, wall, outlet = boundarypiece.(Ω, (:inlet, :wall, :outlet))
    exact = x^2 + y^2 + 4k * t
    bcs = [
        u(t, inlet) ~ exact, u(t, wall) ~ exact,
        k * NormalDerivative(outlet)(u(t, outlet)) + α * u(t, outlet) ~ 4k + α * (4 + y^2 + 4k * t),
        u(0, x, y) ~ x^2 + y^2,
    ]
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ k * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [k => 2.0, α => 5.0]
    )
    return solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# Shows: `gmsh_domain(path)` gives the domain of the manual route, with the physical groups as named pieces,
#        and the same solution through `discretize`; a missing file or a file without physical groups is an error.
@testset "gmsh_domain reads a Gmsh file as the manual route does" begin
    mktempdir() do dir
        path = write_lshape(joinpath(dir, "lshape.msh"))
        manual = MeshDomain(FerriteGmsh.togrid(path))
        Ω = gmsh_domain(path)
        @test Ω isa MeshDomain
        @test Ω.mesh.cells == manual.mesh.cells
        @test [n.x for n in Ω.mesh.nodes] == [n.x for n in manual.mesh.nodes]
        @test Ω.mesh.facetsets == manual.mesh.facetsets
        @test sort(string.(boundarypieces(Ω))) == sort(string.(boundarypieces(manual)))
        @test string(boundarypiece(Ω, :inlet)) == "Γ_inlet"
        @test_throws "has no facet set :outflow; valid names are :inlet, :outlet, :wall" boundarypiece(Ω, :outflow)
        sol, sol_manual = lshape_solution(Ω), lshape_solution(manual)
        @test maximum(abs, sol[u(t, x, y)][end, :] .- sol_manual[u(t, x, y)][end, :]) < 1.0e-12

        @test_throws "no such file" gmsh_domain(joinpath(dir, "missing.msh"))
        bare = write_lshape(joinpath(dir, "bare.msh"); groups = false)
        @test_throws "no physical groups" gmsh_domain(bare)
        unnamed = write_lshape(joinpath(dir, "unnamed.msh"); unnamed = true)
        @test_throws "every physical group must be named" gmsh_domain(unnamed)

        padded = " " * path * " "
        @test boundarypieces(gmsh_domain(strip(padded))) |> length == 3
    end
end

# Shows: `gmsh_domain` leaves the Gmsh session as it found it: a session of the caller keeps its models, and a
#        session started by `gmsh_domain` is finalized also when the conversion throws (cubic triangles).
@testset "gmsh_domain leaves the Gmsh session as it found it" begin
    mktempdir() do dir
        path = write_lshape(joinpath(dir, "lshape.msh"))
        cubic = write_lshape(joinpath(dir, "cubic.msh"); order = 3)
        Gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("caller")
            models = gmsh.model.list()
            for _ in 1:2
                gmsh_domain(path)
                @test gmsh.model.list() == models
                @test gmsh.model.getCurrent() == "caller"
            end
            @test_throws Exception gmsh_domain(cubic)
            @test gmsh.model.list() == models
            @test gmsh.model.getCurrent() == "caller"
        finally
            Gmsh.finalize()
        end
        @test iszero(gmsh.isInitialized())
        @test_throws Exception gmsh_domain(cubic)
        @test iszero(gmsh.isInitialized())
        @test gmsh_domain(path) isa MeshDomain
        @test iszero(gmsh.isInitialized())
    end
end
