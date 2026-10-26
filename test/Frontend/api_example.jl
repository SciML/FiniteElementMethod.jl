# The user API example at the top of the spec, on the Gmsh L-shape, with the M2 exit criteria of
# spec §7.8.
using FiniteElementMethod, ModelingToolkit, DomainSets, FerriteGmsh, OrdinaryDiffEqBDF
using OrdinaryDiffEqBDF: NLNewton
using ModelingToolkit: t_nounits as t
using Ferrite: Ferrite, Vec, PointEvalHandler, evaluate_at_points
using SciMLBase: SciMLBase
using Test

include(joinpath(@__DIR__, "..", "shared", "lshape.jl"))

@testset "API example on the Gmsh L-shape" begin
    @parameters k α
    @variables x y u(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

    grid = lshape_grid()                   # Gmsh physical groups "inlet", "wall", "outlet" become facet sets
    Ω = MeshDomain(grid)
    # `Ref(Ω)`: PDEBase does not yet make a MeshDomain broadcast as a scalar.
    Γ_in, Γ_wall, Γ_out = boundarypiece.(Ref(Ω), (:inlet, :wall, :outlet))
    Dn_wall, Dn_out = NormalDerivative(Γ_wall), NormalDerivative(Γ_out)

    eqs = [Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y))) + 1]
    bcs = [
        u(t, Γ_in) ~ sin(t),
        Dn_wall(u(t, Γ_wall)) ~ 0,
        k * Dn_out(u(t, Γ_out)) + α * u(t, Γ_out) ~ α * 0.3,
        u(0, x, y) ~ 0,
    ]
    domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y)], [k => 2.0, α => 5.0])

    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 1}()))
    tbuild = @elapsed prob = DAEProblem(pdesys, disc)
    tsolve = @elapsed sol = solve(prob, DFBDF())
    @info "API example" ndofs = length(prob.u0) tbuild tsolve

    @test SciMLBase.successful_retcode(sol)
    @test sol isa SciMLBase.PDETimeSeriesSolution
    @test sol.disc_data isa FEMMetadata
    nv = Ferrite.getnnodes(grid)
    @test size(sol[u(t, x, y)]) == (length(sol.t), nv)
    @test length(sol[x]) == length(sol[y]) == nv

    @testset "point values equal a direct PointEvalHandler evaluation" begin
        pts = [(0.3, 0.4), (1.5, 0.5), (0.5, 1.7), (0.05, 1.95)]
        ph = PointEvalHandler(grid, [Vec(p) for p in pts])
        for k_ in (2, length(sol.t) ÷ 2, length(sol.t))
            direct = evaluate_at_points(ph, sol.disc_data.space.dh, sol.original_sol.u[k_], :u)
            for (i, (px, py)) in enumerate(pts)
                @test abs(sol(sol.t[k_], px, py; dv = u(t, x, y)) - direct[i]) <= 1.0e-14
            end
        end
    end

    @testset "Dirichlet trace holds sin(t)" begin
        trace = sol[u(t, Γ_in)]
        @test size(trace, 1) == length(sol.t)
        # The algebraic rows u_d - sin(t) are solved exactly by a Newton step with a current W.
        # OrdinaryDiffEq reuses W across step-size changes and then rescales the Newton step
        # ("relaxed Newton"), which leaves those rows off by about the solver tolerance.
        @test maximum(abs, trace .- sin.(sol.t)) <= 1.0e-5
        fresh = solve(prob, DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0)))
        @test maximum(abs, fresh[u(t, Γ_in)] .- sin.(fresh.t)) <= 1.0e-12
    end

    @test isnan(sol(0.5, 1.5, 1.5; dv = u(t, x, y)))     # the L's missing quadrant
    @test_throws DomainError sol(sol.t[end] + 1, 0.3, 0.4; dv = u(t, x, y))
    @test size(sol[Dn_out(u(t, Γ_out))]) == (length(sol.t), length(Ferrite.getfacetset(grid, "outlet")))

    mktempdir() do dir
        pvd = write_vtk(sol, joinpath(dir, "heat"))
        @test isfile(pvd)
        @test count(endswith(".vtu"), readdir(dir)) == length(sol.t)
    end
end
