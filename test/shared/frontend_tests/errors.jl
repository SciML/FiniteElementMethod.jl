# The user-facing error paths of the front end. Each is raised by `discretize` (which is
# `discretize(pdesys, disc)`) before any solver runs. Documented user-facing messages are matched
# exactly; errors raised inside PDEBase's analysis are matched by substring.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle, Quadrilateral, Grid, Node, Vec, addfacetset!
using PDEBase: PDEBase

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, @register_symbolic

opaque(x) = 1 + x^2
@register_symbolic opaque(x)

@variables x y u(..) v(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
const ic = u(0, x, y) ~ 0

function lower(eqs, bcs, Ω; dvs = [u(t, x, y)], domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], kw...)
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], dvs, [])
    return discretize(pdesys, FEMDiscretization(Ω, t; kw...))
end

box() = MeshDomain(rectangle_grid(Triangle, (2, 2)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
dirichlet(Ω) = [u(t, Γ) ~ 0 for Γ in sides(Ω)]

@testset verbose = true "front-end errors" begin
    @testset "unknown piece name" begin
        Ω = box()
        @test occursin("valid names are", errmsg(() -> boundarypiece(Ω, :nosuch)))
        Γ = PDEBase.BoundaryPiece(Ω, :nosuch, nothing)
        @test_throws ArgumentError("unknown facet set :nosuch; the mesh has facet sets :bottom, :left, :right, :top") lower(
            [heat], [u(t, Γ) ~ 0, ic], Ω
        )
    end

    @testset "overlapping pieces" begin
        grid = rectangle_grid(Triangle, (2, 2))
        addfacetset!(grid, "inlet", x -> x[1] ≈ 0)
        Ω = MeshDomain(grid)
        Γ, Γl = boundarypiece.(Ω, (:inlet, :left))
        @test_throws ArgumentError(
            "facet sets :inlet and :left share 2 facets (first: cell 1, local facet 3); boundary pieces must be disjoint"
        ) lower([heat], [u(t, Γ) ~ 0, u(t, Γl) ~ 1, ic], Ω)
    end

    @testset "trace without its time argument" begin
        Ω = box()
        Γ = first(sides(Ω))
        @test_throws ArgumentError("u(Γ_left): trace is missing the time argument; write `u(t, Γ_left)`") lower(
            [heat], [u(Γ) ~ 0, ic], Ω
        )
    end

    @testset "normal derivative on another piece" begin
        Ω = box()
        Γl, Γr = sides(Ω)
        @test_throws ArgumentError(
            "Dn_left(u(t, Γ_right)) takes the normal derivative on Γ_left of an expression located at " *
                "Any[PDEBase.Piece(Γ_right)], which is not on Γ_left; write the trace on the operator's piece"
        ) lower([heat], [NormalDerivative(Γl)(u(t, Γr)) ~ 0, ic], Ω)
    end

    @testset "endpoint form on a mesh" begin
        # A mesh domain has no coordinate intervals, so u(t, 0, y) cannot name a face even when
        # the mesh is a box; the condition must be a trace on a boundary piece.
        Ω = box()
        msg = errmsg(() -> lower([heat], [u(t, 0, y) ~ 0, ic], Ω))
        @test occursin("write a trace on a boundary piece instead", msg)
    end

    @testset "periodic conditions" begin
        Ω = box()
        Γl, Γr = sides(Ω)
        @test_throws ArgumentError(
            "periodic boundary conditions are not supported: u(t, Γ_left) ~ u(t, Γ_right)"
        ) lower([heat], [u(t, Γl) ~ u(t, Γr), ic], Ω)
    end

    @testset "reserved options" begin
        Ω = box()
        @test_throws ArgumentError("codegen = :mtk is not supported; use the default codegen = :numeric") FEMDiscretization(
            Ω, t; codegen = :mtk
        )
        @test_throws ArgumentError("dirichlet must be :algebraic or :eliminate, got :condense") FEMDiscretization(
            Ω, t; dirichlet = :condense
        )
        for (value, shown) in ((0, "0"), (-1.0e8, "-1.0e8"), (NaN, "NaN"), ("1e8", "\"1e8\""), (:on, ":on"), ([1.0e8], "[1.0e8]"))
            @test_throws ArgumentError("algebraic_abstol must be :auto, nothing or a positive number, got $shown") FEMDiscretization(
                Ω, t; algebraic_abstol = value
            )
        end
    end

    @testset "steady and time-dependent entry points" begin
        Ω = box()
        @named steady = PDESystem(
            [Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], [u(Γ) ~ 0 for Γ in sides(Ω)], [(x, y) ∈ Ω], [x, y], [u(x, y)], []
        )
        @named transient = PDESystem([heat], [dirichlet(Ω); ic], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
        @test_throws ArgumentError(
            "the PDESystem pairs t with an interval, but the FEMDiscretization has no time variable; " *
                "for a time-dependent problem write FEMDiscretization(Ω, t)"
        ) discretize(transient, FEMDiscretization(Ω))
        @test_throws ArgumentError(
            "keyword `name` is set by the FEM discretization and cannot be passed; the reserved keywords are iv, unknowns, du0, coordinate, name, parameters"
        ) discretize(steady, FEMDiscretization(Ω); name = :other)
        @test_throws ArgumentError(
            "algebraic_abstol applies to time-dependent problems (the DAE path); a steady FEMDiscretization has no time stepper"
        ) FEMDiscretization(Ω; algebraic_abstol = 1.0e8)
        # :auto (the default) and nothing are accepted and set no abstol on the NonlinearProblem.
        for value in (:auto, nothing)
            @test !haskey(discretize(steady, FEMDiscretization(Ω; algebraic_abstol = value)).kwargs, :abstol)
        end
    end

    @testset "time-dependent operators" begin
        Ω = box()
        td = Dt(u(t, x, y)) ~ Dx((1 + t) * Dx(u(t, x, y))) + Dy((1 + t) * Dy(u(t, x, y)))
        @test_throws ArgumentError(
            "time-dependent coefficients in the mass or stiffness operator are not supported: equation 1 has " *
                "G₁*W₁ + G₂*W₂ + G₁*T*W₁ + G₂*T*W₂"
        ) lower([td], [dirichlet(Ω); ic], Ω)
    end

    @testset "non-finite initial condition names the node" begin
        Ω = box()
        r = sqrt(x^2 + y^2)
        msg = errmsg(() -> lower([heat], [dirichlet(Ω); u(0, x, y) ~ sin(r) / r], Ω))
        @test occursin("is not finite at the node coordinate (0.0, 0.0)", msg)
    end

    @testset "essential condition coupling two fields" begin
        Ω = box()
        eqs = [heat, Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))]
        Γl = first(sides(Ω))
        bcs = [u(t, Γl) + v(t, Γl) ~ 1, ic, v(0, x, y) ~ 0]
        msg = errmsg(() -> lower(eqs, bcs, Ω; dvs = [u(t, x, y), v(t, x, y)]))
        @test occursin("Dirichlet data depends on another field", msg)
    end

    @testset "no MeshDomain" begin
        Ω = box()
        domains = [(x, y) ∈ Rectangle([0.0, 0.0], [1.0, 1.0]), t ∈ Interval(0.0, 1.0)]
        @test_throws ArgumentError(
            "the PDESystem has no spatial domain of the form (x, y) ∈ MeshDomain(mesh); FEMDiscretization discretizes on a mesh"
        ) lower([heat], [ic], Ω; domains)
    end

    @testset "time must be an interval" begin
        Ω = box()
        domains = [(x, y) ∈ Ω, t ∈ UnionDomain(Interval(0.0, 1.0), Interval(2.0, 3.0))]
        @test_throws ArgumentError("time t must be paired with an interval, as in t ∈ Interval(0.0, 1.0)") lower(
            [heat], [ic], Ω; domains
        )
    end

    @testset "time must be the first or the last argument" begin
        Ω = box()
        eq = Dt(u(x, t, y)) ~ Dx(Dx(u(x, t, y))) + Dy(Dy(u(x, t, y)))
        bcs = [u(x, 0, y) ~ 0]
        @test_throws ArgumentError("time must be the first or the last argument of u(x, t, y), as in u(t, x, y) or u(x, y, t)") lower(
            [eq], bcs, Ω; dvs = [u(x, t, y)]
        )
    end

    @testset "unknown spaces key" begin
        Ω = box()
        @test_throws ArgumentError(
            "`spaces` has keys that are not dependent variables of the system: v; the dependent variables are u(t, x, y)"
        ) lower([heat], [dirichlet(Ω); ic], Ω; spaces = Dict(v => Lagrange{RefTriangle, 2}()))
    end

    @testset "mixed cell types need a space per variable" begin
        nodes = [Node(Vec(0.0, 0.0)), Node(Vec(1.0, 0.0)), Node(Vec(0.0, 1.0)), Node(Vec(2.0, 0.0)), Node(Vec(1.0, 1.0))]
        grid = Grid(Ferrite.AbstractCell[Triangle((1, 2, 3)), Quadrilateral((2, 4, 5, 3))], nodes)
        Ω = MeshDomain(grid)
        @test_throws ArgumentError(
            "the mesh has mixed cell types, so every dependent variable needs a space in `spaces`; u(t, x, y) has none"
        ) lower([heat], [ic], Ω)
    end

    @testset "opaque coefficient functions" begin
        Ω = box()
        state = Dt(u(t, x, y)) ~ opaque(u(t, x, y)) * Dx(u(t, x, y))
        @test occursin("operator opaque is not supported by the strong-form analysis", errmsg(() -> lower([state], [dirichlet(Ω); ic], Ω)))
        nondiv = Dt(u(t, x, y)) ~ opaque(x) * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
        @test occursin("is not differentiable; write the equation in divergence form or register the derivative", errmsg(() -> lower([nondiv], [dirichlet(Ω); ic], Ω)))
        # Spec §8.2: an opaque coefficient without a registered derivative is accepted in divergence form.
        div = Dt(u(t, x, y)) ~ Dx(opaque(x) * Dx(u(t, x, y))) + Dy(opaque(x) * Dy(u(t, x, y)))
        sol = solve(lower([div], [[u(t, Γ) ~ 1 for Γ in sides(Ω)]; ic], Ω), EXACT_ROWS)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "parameters" begin
        Ω = box()
        @parameters k
        eq = Dt(u(t, x, y)) ~ k * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
        system(ps...) = PDESystem([eq], [dirichlet(Ω); ic], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], ps...; name = :pdesys)
        # An undeclared parameter is reported the same way whether `ps` is empty or left out.
        for pdesys in (system([]), system())
            @test occursin("unresolved symbols", errmsg(() -> discretize(pdesys, FEMDiscretization(Ω, t))))
        end
        @test_throws ArgumentError("parameter k needs a numeric value in the PDESystem, as k => value") discretize(
            system([k]), FEMDiscretization(Ω, t)
        )
    end

    @testset "a piece with no condition warns when its flux is not zero" begin
        Ω = box()
        Γl, Γr, Γb, Γt = sides(Ω)
        @test_logs (:warn, r"u: 2 of 8 boundary facets have no condition: Γ_top \(2 facets\)") match_mode = :any lower([heat], [u(t, Γl) ~ 0, u(t, Γr) ~ 0, u(t, Γb) ~ 0, ic], Ω)
    end
end
