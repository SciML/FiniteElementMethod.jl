# Incompressible Navier-Stokes through the full interface, at a size that solves in seconds: a steady mixed problem
# for velocity and pressure with Taylor-Hood elements (P2 velocity, P1 pressure), checked against Kovasznay's exact flow.
# (The DFG cylinder benchmark, ~149k unknowns, is the NavierStokes test group, not this one.)
using FiniteElementMethod, ModelingToolkit, DomainSets, NonlinearSolveFirstOrder, Test
using Ferrite: generate_grid, Triangle, Vec
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..) v(..) p(..)
Dx, Dy = Differential(x), Differential(y)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))

const ν = 1 / 20
const λ = 1 / (2ν) - sqrt(1 / (4ν^2) + 4pi^2)
kovasznay_u(x_, y_) = 1 - exp(λ * x_) * cos(2pi * y_)
kovasznay_v(x_, y_) = λ / 2pi * exp(λ * x_) * sin(2pi * y_)
kovasznay_p(x_, y_) = (1 - exp(2λ * x_)) / 2

# PDE:    ν Δu - (u.grad)u - grad p = 0 and div u = 0 with ν = 1/20 (Reynolds number 20), no source
# Domain: x in (-1/2, 1), y in (-1/2, 3/2), meshes of 6 x 8, 12 x 16 and 24 x 32 triangle pairs (up to 7 195 unknowns)
# BCs:    velocity = the exact velocity on all four sides; the pressure is fixed to the exact pressure on the right side
# Exact:  u = 1 - exp(λx) cos(2πy), v = λ/(2π) exp(λx) sin(2πy), p = (1 - exp(2λx))/2, λ = Re/2 - sqrt(Re^2/4 + 4π^2)
# Shows:  a steady nonlinear saddle-point problem: `pairing` tests the continuity equation with p, Newton finds the flow, and
#         the velocity converges at rate 3 and the pressure at rate 2.
@testset "steady Navier-Stokes with Taylor-Hood elements reproduces Kovasznay's flow at rates 3 and 2" begin
    continuity = 0 ~ Dx(u(x, y)) + Dy(v(x, y))
    eqs = [
        ν * Δ(u(x, y)) - u(x, y) * Dx(u(x, y)) - v(x, y) * Dy(u(x, y)) - Dx(p(x, y)) ~ 0,
        ν * Δ(v(x, y)) - u(x, y) * Dx(v(x, y)) - v(x, y) * Dy(v(x, y)) - Dy(p(x, y)) ~ 0,
        continuity,
    ]
    exact_u, exact_v = 1 - exp(λ * x) * cos(2pi * y), λ / 2pi * exp(λ * x) * sin(2pi * y)
    ns = (2, 4, 8)
    errors = map(ns) do k
        Ω = MeshDomain(generate_grid(Triangle, (3k, 4k), Vec(-0.5, -0.5), Vec(1.0, 1.5)))
        sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [[u(Γ) ~ exact_u for Γ in sides]; [v(Γ) ~ exact_v for Γ in sides]; p(sides[2]) ~ (1 - exp(2λ * x)) / 2]
        @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω], [x, y], [u(x, y), v(x, y), p(x, y)], [])
        P2, P1 = Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 1}()
        disc = FEMDiscretization(Ω; spaces = Dict(u => P2, v => P2, p => P1), pairing = Dict(continuity => p))
        sol = solve(discretize(pdesys, disc), NewtonRaphson(); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        xs, ys = sample_points(40; lo = -0.5, hi = 1.0), sample_points(40; lo = -0.5, hi = 1.5)
        (l2_error(sol, u(x, y), kovasznay_u, xs, ys), l2_error(sol, v(x, y), kovasznay_v, xs, ys), l2_error(sol, p(x, y), kovasznay_p, xs, ys))
    end
    h = 1 ./ ns
    @test all(observed_rate(getindex.(errors, 1), h) .> 2.5) && all(observed_rate(getindex.(errors, 2), h) .> 2.5)
    @test all(observed_rate(getindex.(errors, 3), h) .> 1.7)
end
