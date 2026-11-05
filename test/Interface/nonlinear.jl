# Nonlinear terms through the full interface: state-dependent diffusion coefficients and convection `u u_x`, solved
# in a transient problem against manufactured solutions. The nonlinear part of the operator is assembled
# with its analytic Jacobian, and the source is built symbolically so that the exact solution solves the equation.
# (The logistic reaction is in `reaction_diffusion.jl`, the steady nonlinear problem in `steady.jl`.)
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Symbolics, Test
using Ferrite: generate_grid, Line, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# PDE:    u_t + u u_x = nu u_xx + f with nu = 0.1 (viscous Burgers equation with a source)
# Domain: x in (0, 1), t in (0, 0.2), meshes of 2, 4, 8 and 16 line cells, P2
# BCs:    u(t, 0) = u(t, 1) = 0; IC u(0, x) = sin(pi x)
# Exact:  u = (1 + t) sin(pi x); f = u_t + u u_x - nu u_xx
# Shows:  a nonlinear convection term is solved at the optimal rate 3.
@testset "P2 elements on a line converge at rate 3 for the viscous Burgers equation" begin
    nu, tend = 0.1, 0.2
    exact = (1 + t) * sin(pi * x)
    source = expand_derivatives(Dt(exact) + exact * Dx(exact) - nu * Dx(Dx(exact)))
    ns = (2, 4, 8, 16)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
        left, right = boundarypiece.(Ω, (:left, :right))
        eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) ~ nu * Dx(Dx(u(t, x))) + source
        @named pdesys = PDESystem([eq], [u(t, left) ~ 0, u(t, right) ~ 0, u(0, x) ~ sin(pi * x)], [(x,) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x], [u(t, x)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefLine, 2}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
        l2_error(sol, u(t, x), x_ -> (1 + tend) * sin(pi * x_), tend, sample_points(200))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end

# PDE:    u_t = (k(u) u_x)_x + (k(u) u_y)_y + f with k(u) = 1 + u^2 (nonlinear diffusion)
# Domain: the unit square, t in (0, 0.1), meshes of 4 x 4, 8 x 8 and 16 x 16 triangle pairs, P2
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = (1 + t) sin(pi x) sin(pi y); f = u_t - div(k(u) grad u)
# Shows:  a state-dependent diffusion coefficient in divergence form is solved at the optimal rate 3.
@testset "P2 elements converge at rate 3 for nonlinear diffusion with k(u) = 1 + u^2" begin
    tend = 0.1
    exact = (1 + t) * sin(pi * x) * sin(pi * y)
    source = expand_derivatives(Dt(exact) - Dx((1 + exact^2) * Dx(exact)) - Dy((1 + exact^2) * Dy(exact)))
    ns = (4, 8, 16)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
        U = u(t, x, y)
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + source
        bcs = [[u(t, Γ) ~ 0 for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ sin(pi * x) * sin(pi * y)]
        @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [U], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
        l2_error(sol, U, (x_, y_) -> (1 + tend) * sin(pi * x_) * sin(pi * y_), tend, sample_points(40), sample_points(40))
    end
    @test all(observed_rate(errors, 1 ./ ns) .> 2.8)
end
