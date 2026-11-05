# The 1D heat equation through the full interface: PDESystem -> FEMDiscretization -> discretize ->
# solve -> comparison with the exact solution, in piecewise-linear (P1) and piecewise-quadratic (P2)
# elements, with the convergence rate read off a refinement sequence.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x u(..)
Dt, Dx = Differential(t), Differential(x)

# Solves u_t = u_xx on a mesh of `n` cells with the given element order, returns the L2 error at t = 0.1.
function heat_1d_error(n, order)
    Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    eqs = [Dt(u(t, x)) ~ Dx(Dx(u(t, x)))]
    bcs = [u(t, left) ~ 1, u(t, right) ~ 1, u(0, x) ~ 1 + sin(pi * x)]
    domains = [(x,) ∈ Ω, t ∈ Interval(0.0, 0.1)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, order}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
    exact(x_) = 1 + exp(-pi^2 * 0.1) * sin(pi * x_)
    return l2_error(sol, u(t, x), exact, 0.1, sample_points(400))
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), meshes of 4, 8, 16 and 32 line cells
# BCs:    u(t, 0) = u(t, 1) = 1 (constant Dirichlet); IC u(0, x) = 1 + sin(pi x)
# Exact:  u = 1 + exp(-pi^2 t) sin(pi x)
# Shows:  P1 elements converge at the optimal L2 rate 2.
@testset "P1 elements on a line converge at rate 2 for the heat equation with constant Dirichlet data" begin
    ns = (4, 8, 16, 32)
    errors = heat_1d_error.(ns, 1)
    @test issorted(errors; rev = true)
    @test all(observed_rate(errors, 1 ./ ns) .> 1.9)
end

# PDE:    u_t = u_xx
# Domain: x in (0, 1), t in (0, 0.1), meshes of 2, 4, 8 and 16 line cells
# BCs:    u(t, 0) = u(t, 1) = 1 (constant Dirichlet); IC u(0, x) = 1 + sin(pi x)
# Exact:  u = 1 + exp(-pi^2 t) sin(pi x)
# Shows:  P2 elements, chosen with `spaces`, converge at the optimal L2 rate 3.
@testset "P2 elements on a line converge at rate 3 for the heat equation with constant Dirichlet data" begin
    ns = (2, 4, 8, 16)
    errors = heat_1d_error.(ns, 2)
    @test issorted(errors; rev = true)
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end
