# Every M2 error path of the front end, with its message. Each is raised by `discretize` (which is
# `DAEProblem(pdesys, disc)`), before any solver runs.
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
        @test occursin("unknown facet set :nosuch", errmsg(() -> lower([heat], [u(t, Γ) ~ 0, ic], Ω)))
    end

    @testset "overlapping pieces" begin
        grid = rectangle_grid(Triangle, (2, 2))
        addfacetset!(grid, "inlet", x -> x[1] ≈ 0)
        Ω = MeshDomain(grid)
        Γ, Γl = boundarypiece.(Ω, (:inlet, :left))
        msg = errmsg(() -> lower([heat], [u(t, Γ) ~ 0, u(t, Γl) ~ 1, ic], Ω))
        @test occursin("share 2 facets", msg) && occursin("boundary pieces must be disjoint", msg)
    end

    @testset "trace without its time argument" begin
        Ω = box()
        Γ = first(sides(Ω))
        @test occursin("trace is missing the time argument; write `u(t, Γ_left)`", errmsg(() -> lower([heat], [u(Γ) ~ 0, ic], Ω)))
    end

    @testset "normal derivative on another piece" begin
        Ω = box()
        Γl, Γr = sides(Ω)
        msg = errmsg(() -> lower([heat], [NormalDerivative(Γl)(u(t, Γr)) ~ 0, ic], Ω))
        @test occursin("which is not on Γ_left", msg)
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
        msg = errmsg(() -> lower([heat], [u(t, Γl) ~ u(t, Γr), ic], Ω))
        @test startswith(msg, "ArgumentError: periodic conditions need milestone M4 (Ferrite PeriodicDirichlet)")
    end

    @testset "reserved options" begin
        Ω = box()
        @test occursin("codegen = :mtk lands in M3", errmsg(() -> FEMDiscretization(Ω, t; codegen = :mtk)))
        @test occursin("dirichlet = :eliminate lands in M4", errmsg(() -> FEMDiscretization(Ω, t; dirichlet = :eliminate)))
    end

    @testset "nonlinear and time-dependent operators" begin
        Ω = box()
        nl = Dt(u(t, x, y)) ~ Dx((1 + u(t, x, y)^2) * Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
        @test occursin("nonlinear terms land in M3", errmsg(() -> lower([nl], [dirichlet(Ω); ic], Ω)))
        td = Dt(u(t, x, y)) ~ Dx((1 + t) * Dx(u(t, x, y))) + Dy((1 + t) * Dy(u(t, x, y)))
        @test occursin("time-dependent coefficients in the mass or stiffness operator land in M3", errmsg(() -> lower([td], [dirichlet(Ω); ic], Ω)))
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
        msg = errmsg(() -> lower([heat], [ic], Ω; domains))
        @test occursin("has no spatial domain of the form (x, y) ∈ MeshDomain(mesh)", msg)
    end

    @testset "time must be an interval" begin
        Ω = box()
        domains = [(x, y) ∈ Ω, t ∈ UnionDomain(Interval(0.0, 1.0), Interval(2.0, 3.0))]
        @test occursin("time t must be paired with an interval", errmsg(() -> lower([heat], [ic], Ω; domains)))
    end

    @testset "time must be the first argument" begin
        Ω = box()
        eq = Dt(u(x, y, t)) ~ Dx(Dx(u(x, y, t))) + Dy(Dy(u(x, y, t)))
        bcs = [u(x, y, 0) ~ 0]
        @test_throws ArgumentError("time must be the first argument of u(x, y, t); time-last ordering lands later") lower(
            [eq], bcs, Ω; dvs = [u(x, y, t)]
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
        @test occursin("mixed cell types, so every dependent variable needs a space", errmsg(() -> lower([heat], [ic], Ω)))
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

    @testset "a piece with no condition warns when its flux is not zero" begin
        Ω = box()
        Γl, Γr, Γb, Γt = sides(Ω)
        @test_logs (:warn, r"boundary piece Γ_top has no condition") match_mode = :any lower([heat], [u(t, Γl) ~ 0, u(t, Γr) ~ 0, u(t, Γb) ~ 0, ic], Ω)
    end
end
