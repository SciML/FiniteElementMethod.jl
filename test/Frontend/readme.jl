# The README's code block runs as written and its solution satisfies the conditions it states.
# Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "..", "examples", "readme_code.jl"))
    using SciMLBase: SciMLBase

    @testset "README showcase" begin
        m = Base.invokelatest(run_readme)   # defined by the include in this same block
        sol, u, x, y, t = m.sol, m.u, m.x, m.y, m.t
        @test SciMLBase.successful_retcode(sol)
        @test sol.t[end] == 6.0
        # The Dirichlet trace holds the pulsing source to about the solver tolerance (DFBDF's default
        # reltol 1e-3): OrdinaryDiffEq's relaxed Newton leaves the algebraic rows off by that much
        # (see test/shared/frontend_tests/api_example.jl).
        @test maximum(abs, sol[u(t, m.Γ_in)] .- sin.(sol.t) .^ 2) <= 1.0e-3
        # The temperature stays within the source data's range [0, 1] up to the tiny undershoot a
        # consistent P1 mass matrix allows (measured -1.3e-6; it has no discrete maximum principle),
        # and heat has crossed the plate past the heated hole (measured 0.32 at (1, 0.9)).
        U = sol[u(t, x, y)]
        @test -1.0e-4 <= minimum(U) && maximum(U) <= 1 + 1.0e-4
        @test sol(6.0, 1.0, 0.9; dv = u(t, x, y)) > 0.1
        @test isnan(sol(6.0, 0.65, 0.55; dv = u(t, x, y)))   # the centre of the heated hole
    end
else
    @testset "README showcase (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier, StrongForm boundary routing or pairing-ordered mesh coordinates (PDEBase#158)"
    end
end
