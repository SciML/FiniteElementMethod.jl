# The README's code block runs as written and its solution satisfies the conditions it states.
using FiniteElementMethod, Test
include(joinpath(@__DIR__, "..", "..", "examples", "readme_code.jl"))
using SciMLBase: SciMLBase

@testset "README showcase" begin
    m = Base.invokelatest(run_readme)   # defined by the include above
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

@testset "README Navier–Stokes" begin
    m = Base.invokelatest(run_readme; block = 2, names = (:sol, :u, :v, :x, :y, :t, :tend, :Γ_in, :Γ_cyl))
    sol, u, v, x, y, t = m.sol, m.u, m.v, m.x, m.y, m.t
    @test SciMLBase.successful_retcode(sol)
    @test sol.t[end] == m.tend == 6.0
    # No slip on the cylinder, and the parabolic inflow of peak 1.5 once the ramp 1 - exp(-t²) has saturated.
    @test maximum(abs, sol[u(t, m.Γ_cyl)]) <= 1.0e-6 && maximum(abs, sol[v(t, m.Γ_cyl)]) <= 1.0e-6
    @test maximum(sol[u(t, m.Γ_in)][end, :]) ≈ 1.5 atol = 1.0e-3
    # The wake sheds: the cross-stream velocity behind the cylinder alternates sign with the DFG 2D-2
    # Strouhal number f D / Ū ≈ 0.3 (measured 0.28-0.31 on this mesh; D = 0.1, mean inflow Ū = 1).
    times = 4.0:0.01:6.0
    probe = sol(collect(times), [0.6], [0.2]; dv = v(t, x, y))[:, 1, 1]
    crossings = [times[k] for k in 1:(length(times) - 1) if probe[k] * probe[k + 1] < 0]
    @test length(crossings) >= 8
    half_period = (crossings[end] - crossings[1]) / (length(crossings) - 1)
    @test 0.25 <= 0.1 / (2half_period) <= 0.35
end
