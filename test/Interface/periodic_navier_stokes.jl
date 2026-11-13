# Incompressible Navier-Stokes with Taylor-Hood elements (P2 or Q2 velocity, P1 or Q1 pressure) and periodic conditions
# in the cross-stream direction, so that three fields with two different spaces share the periodic pairs.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, NonlinearSolveFirstOrder, LinearSolve, Test
using Ferrite: Grid, addfacetset!, generate_grid, Quadrilateral, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..) v(..) p(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))

const ν_kovasznay = 1 / 20
const λ = 1 / (2ν_kovasznay) - sqrt(1 / (4ν_kovasznay^2) + 4pi^2)
kovasznay_u(x_, y_) = 1 - exp(λ * x_) * cos(2pi * y_)
kovasznay_v(x_, y_) = λ / 2pi * exp(λ * x_) * sin(2pi * y_)
kovasznay_p(x_, y_) = (1 - exp(2λ * x_)) / 2

# PDE:    ν Δu - (u.grad)u - grad p = 0 and div u = 0 with ν = 1/20 (Reynolds number 20), no source
# Domain: x in (-1/2, 1), y in (-1/2, 3/2) (two periods of the flow in y), meshes of 6 x 8, 12 x 16 and 24 x 32 triangle pairs
# BCs:    velocity = the exact velocity on left and right and the pressure fixed on the right, as in navier_stokes.jl; bottom and
#         top are periodic for u, v and p instead of carrying exact data
# Exact:  Kovasznay's flow, which has period 1 in y
# Shows:  a steady nonlinear saddle-point problem with periodic pairs for velocity (P2) and pressure (P1): the velocity
#         converges at rate 3 and the pressure at rate 2, as with exact data on the sides.
@testset "steady Kovasznay flow with periodic bottom and top converges at rates 3 and 2" begin
    continuity = 0 ~ Dx(u(x, y)) + Dy(v(x, y))
    eqs = [
        ν_kovasznay * Δ(u(x, y)) - u(x, y) * Dx(u(x, y)) - v(x, y) * Dy(u(x, y)) - Dx(p(x, y)) ~ 0,
        ν_kovasznay * Δ(v(x, y)) - u(x, y) * Dx(v(x, y)) - v(x, y) * Dy(v(x, y)) - Dy(p(x, y)) ~ 0,
        continuity,
    ]
    exact_u, exact_v = kovasznay_u(x, y), kovasznay_v(x, y)
    ns = (2, 4, 8)
    errors = map(ns) do k
        Ω = MeshDomain(generate_grid(Triangle, (3k, 4k), Vec(-0.5, -0.5), Vec(1.0, 1.5)))
        left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [
            u(left) ~ exact_u, u(right) ~ exact_u, v(left) ~ exact_v, v(right) ~ exact_v, p(right) ~ (1 - exp(2λ * x)) / 2,
            u(bottom) ~ u(top), v(bottom) ~ v(top), p(bottom) ~ p(top),
        ]
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

# The channel [0, 2] x [0, 1] of `nx` x `ny` quadrilaterals with the 2 x 2 cells at 0.4 < x, y < 0.6 removed, with the
# facet sets inlet, outlet, bottom, top and block (the sides of the removed cells).
function channel_with_block(nx, ny)
    full = generate_grid(Quadrilateral, (nx, ny), Vec(0.0, 0.0), Vec(2.0, 1.0))
    center(cell) = sum(n -> full.nodes[n].x, cell.nodes) / 4
    cells = [c for c in full.cells if !(0.4 < center(c)[1] < 0.6 && 0.4 < center(c)[2] < 0.6)]
    grid = Grid(cells, full.nodes)
    tol = 1.0e-8
    addfacetset!(grid, "inlet", X -> X[1] < tol)
    addfacetset!(grid, "outlet", X -> X[1] > 2 - tol)
    addfacetset!(grid, "bottom", X -> X[2] < tol)
    addfacetset!(grid, "top", X -> X[2] > 1 - tol)
    addfacetset!(grid, "block", X -> 0.4 - tol < X[1] < 0.6 + tol && 0.4 - tol < X[2] < 0.6 + tol)
    return grid
end

# PDE:    u_t + (u.grad)u = ν Δu - grad p and div u = 0 with ν = 1/100, from rest, t in (0, 2)
# Domain: the channel [0, 2] x [0, 1] with a square block of side 0.2 at its centre, 196 quadrilaterals, Q2/Q1 (1 934 unknowns)
# BCs:    inflow u = 1 - exp(-t²), v = 0; no slip on the block; p = 0 at the outlet; u, v and p periodic between bottom and top
# Shows:  the flow past a block in a channel periodic in the cross-stream direction runs to the end time, the two sides of the
#         seam carry the same values, the flow speeds up around the block, and the flux through the outlet equals the flux
#         through the inlet (mass is conserved) to 1e-3.
@testset "flow past a block in a channel periodic in the cross-stream direction conserves mass" begin
    Ω = MeshDomain(channel_with_block(20, 10))
    inlet, outlet, bottom, top, block = boundarypiece.(Ω, (:inlet, :outlet, :bottom, :top, :block))
    ν = 0.01
    U, V, P = u(t, x, y), v(t, x, y), p(t, x, y)
    eqs = [
        Dt(U) ~ ν * Δ(U) - U * Dx(U) - V * Dy(U) - Dx(P),
        Dt(V) ~ ν * Δ(V) - U * Dx(V) - V * Dy(V) - Dy(P),
        0 ~ Dx(U) + Dy(V),
    ]
    bcs = [
        u(t, inlet) ~ 1 - exp(-t^2), v(t, inlet) ~ 0, u(t, block) ~ 0, v(t, block) ~ 0, p(t, outlet) ~ 0,
        u(t, bottom) ~ u(t, top), v(t, bottom) ~ v(t, top), p(t, bottom) ~ p(t, top), u(0, x, y) ~ 0, v(0, x, y) ~ 0,
    ]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 2.0)], [t, x, y], [U, V, P], [])
    Q2, Q1 = Lagrange{RefQuadrilateral, 2}(), Lagrange{RefQuadrilateral, 1}()
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Q2, v => Q2, p => Q1), pairing = Dict(eqs[3] => p), natural = [outlet], initializealg = CheckInit())
    prob = discretize(pdesys, disc)
    # The default `algebraic_abstol = :auto` detects the index-2 pressure despite its periodic identification rows, so the
    # problem carries the same tolerances as the non-periodic twin (1e8 on the algebraic rows) and no `abstol` is passed.
    @test prob.kwargs[:abstol] == [d ? 1.0e-6 : 1.0e8 for d in prob.differential_vars]
    sol = solve(prob, DFBDF(linsolve = UMFPACKFactorization()); reltol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol)
    ys = collect(range(0.005, 0.995; length = 100))
    flux(x_) = sum(sol(2.0, [x_], ys; dv = U)) / length(ys)
    @test flux(0.0) ≈ 1 - exp(-4) rtol = 1.0e-3
    @test flux(2.0) ≈ flux(0.0) rtol = 1.0e-3
    seam = [0.5, 1.0, 1.5]
    for dv in (U, V, P)
        @test sol(2.0, seam, [0.0]; dv) ≈ sol(2.0, seam, [1.0]; dv) atol = 1.0e-10
    end
    @test maximum(filter(!isnan, sol(2.0, collect(range(0.0, 2.0; length = 41)), ys; dv = U))) > 1.3
end
