# What `discretize` accepts besides the PDESystem and the FEMDiscretization: problem and solve
# keywords are forwarded to the problem; keywords the discretization computes, further positional
# arguments and a first argument that is not a PDESystem raise.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

box() = MeshDomain(rectangle_grid(Triangle, (2, 2)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

function heat_system(Ω)
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ 0]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    return pdesys
end

function poisson_system(Ω)
    eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ -1
    @named pdesys = PDESystem([eq], [u(Γ) ~ 0 for Γ in sides(Ω)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    return pdesys
end

const DERIVED_MESSAGES = (
    tspan = "keyword `tspan` cannot be passed: the time span is the time domain of the PDESystem; change that domain to change it",
    u0 = "keyword `u0` cannot be passed: the initial state comes from the PDESystem's initial conditions " *
        "(a steady problem starts from zero with its Dirichlet data); change those, or call remake(prob; u0 = ...) on the returned problem",
    p = "keyword `p` cannot be passed: the parameter values are the PDESystem's parameter defaults; change those and " *
        "discretize again (the operators are assembled at those values, so solve rejects remake(prob; p = ...))",
    differential_vars = "keyword `differential_vars` cannot be passed: which unknowns are differential is computed " *
        "from the weak form (the fields with a time derivative)",
)

const DERIVED_VALUES = (tspan = (0.0, 2.0), u0 = zeros(9), p = [1.0], differential_vars = trues(9))

@testset verbose = true "discretize arguments" begin
    Ω = box()
    pdesys = heat_system(Ω)
    disc = FEMDiscretization(Ω, t)

    @testset "solve keywords reach the problem and the solve" begin
        calls = Ref(0)
        callback = SciMLBase.DiscreteCallback((u, t, integrator) -> true, integrator -> (calls[] += 1); save_positions = (false, false))
        prob = discretize(pdesys, disc; saveat = 0.25, tstops = [0.3], abstol = 1.0e-8, callback)
        @test prob.kwargs[:saveat] == 0.25
        @test prob.kwargs[:tstops] == [0.3]
        @test prob.kwargs[:abstol] == 1.0e-8
        @test prob.kwargs[:callback] === callback
        sol = solve(prob, EXACT_ROWS)
        @test SciMLBase.successful_retcode(sol)
        @test sol.t ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
        @test calls[] > 0
    end

    @testset "an unknown keyword is forwarded too" begin
        @test discretize(pdesys, disc; foo = 1).kwargs[:foo] == 1
    end

    @testset "keyword `$k` is derived" for k in keys(DERIVED_MESSAGES)
        kw = NamedTuple{(k,)}((DERIVED_VALUES[k],))
        @test_throws ArgumentError(DERIVED_MESSAGES[k]) discretize(pdesys, disc; kw...)
    end

    @testset "derived keywords in FEMDiscretization and NonlinearProblem" begin
        @test_throws ArgumentError(DERIVED_MESSAGES.tspan) FEMDiscretization(Ω, t; tspan = (0.0, 2.0))
        @test_throws ArgumentError(DERIVED_MESSAGES.p) FEMDiscretization(Ω; p = [1.0])
        steady = poisson_system(Ω)
        @test_throws ArgumentError(DERIVED_MESSAGES.u0) discretize(steady, FEMDiscretization(Ω); u0 = zeros(9))
    end

    @testset "further positional arguments" begin
        @test_throws ArgumentError("discretize(pdesys, disc; kwargs...) takes no further positional arguments; got 1") discretize(
            pdesys, disc, (0.0, 1.0)
        )
        @test_throws ArgumentError("discretize(pdesys, disc; kwargs...) takes no further positional arguments; got 2") discretize(
            pdesys, disc, 1, 2; saveat = 0.1
        )
    end

    @testset "first argument not a PDESystem" begin
        @test_throws ArgumentError("FEMDiscretization discretizes a ModelingToolkit.PDESystem; got Float64") discretize(1.0, disc)
        @test_throws ArgumentError("FEMDiscretization discretizes a ModelingToolkit.PDESystem; got Vector{Int64}") discretize(
            [1, 2], disc, 1
        )
    end
end
