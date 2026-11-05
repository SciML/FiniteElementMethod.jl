# Fields with their own element spaces and their own regions: `spaces` gives each dependent variable an interpolation
# (here P2 for one field and P1 for the other), and `regions` restricts a field to the cells of a cell set.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Symbolics, Test
using Ferrite: generate_grid, addcellset!, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..) w(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))

# PDE:    u_t = Δu + f_u and w_t = Δw + u + f_w (w reads u, u does not read w)
# Domain: the unit square, t in (0, 0.25), meshes of 4 x 4, 8 x 8 and 16 x 16 triangle pairs
# BCs:    Dirichlet data from the exact solution on all sides for both fields; IC from the exact solution
# Exact:  u = (1 + t) sin(pi x) sin(pi y) + x in P2, w = (1 + t) cos(x + y) in P1; f_u and f_w follow from the equations
# Shows:  `spaces = Dict(u => P2, w => P1)` gives the two fields different orders; each converges at its own rate (3 and 2).
@testset "a P2 field and a P1 field coupled in one system converge at rates 3 and 2" begin
    us, ws = (1 + t) * sin(pi * x) * sin(pi * y) + x, (1 + t) * cos(x + y)
    eqs = [Dt(u(t, x, y)) ~ Δ(u(t, x, y)) + expand_derivatives(Dt(us) - Δ(us)), Dt(w(t, x, y)) ~ Δ(w(t, x, y)) + u(t, x, y) + expand_derivatives(Dt(ws) - Δ(ws) - us)]
    u_exact(t_, x_, y_) = (1 + t_) * sin(pi * x_) * sin(pi * y_) + x_
    w_exact(t_, x_, y_) = (1 + t_) * cos(x_ + y_)
    ns = (4, 8, 16)
    errors = map(ns) do n
        Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
        sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [[u(t, Γ) ~ us for Γ in sides]; [w(t, Γ) ~ ws for Γ in sides]; u(0, x, y) ~ substitute(us, t => 0); w(0, x, y) ~ substitute(ws, t => 0)]
        @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.25)], [t, x, y], [u(t, x, y), w(t, x, y)], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}(), w => Lagrange{RefTriangle, 1}()))
        sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
        pts = (sample_points(40), sample_points(40))
        (l2_error(sol, u(t, x, y), (a, b) -> u_exact(0.25, a, b), 0.25, pts...), l2_error(sol, w(t, x, y), (a, b) -> w_exact(0.25, a, b), 0.25, pts...))
    end
    @test all(observed_rate(first.(errors), 1 ./ ns) .> 2.7)
    @test all(observed_rate(last.(errors), 1 ./ ns) .> 1.8)
end

# PDE:    u_t = Δu on the whole square, and w_t = Δw + u + f_w only on the left half x <= 1/2
# Domain: the unit square, t in (0, 0.5), 4 x 4 triangle pairs with the cell set "left_half", P2 for both fields
# BCs:    u = x^2 + y^2 + 4 t on all sides; w = 1 + 1/4 + t on the left side only (w is natural zero-flux elsewhere)
# Exact:  u = x^2 + y^2 + 4 t; w = 1 + (x - 1/2)^2 + t on the left half, whose normal derivative vanishes on the
#         bottom, top and the cut x = 1/2, so the natural condition holds; f_w = 1 - Δw - u
# Shows:  `regions = Dict(w => Region(Ω, :left_half))` puts w on the left half only; it reads NaN elsewhere.
@testset "a field restricted to a region is solved there and reads NaN outside" begin
    grid = generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0))
    addcellset!(grid, "left_half", p -> p[1] <= 0.5 + 1.0e-12)
    Ω = MeshDomain(grid)
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    us, ws = x^2 + y^2 + 4t, 1 + (x - 1 / 2)^2 + t
    eqs = [Dt(u(t, x, y)) ~ Δ(u(t, x, y)), Dt(w(t, x, y)) ~ Δ(w(t, x, y)) + u(t, x, y) + expand_derivatives(Dt(ws) - Δ(ws) - us)]
    bcs = [[u(t, Γ) ~ us for Γ in sides]; w(t, first(sides)) ~ 5 / 4 + t; u(0, x, y) ~ x^2 + y^2; w(0, x, y) ~ 1 + (x - 1 / 2)^2]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, x, y], [u(t, x, y), w(t, x, y)], [])
    P2 = Lagrange{RefTriangle, 2}()
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => P2, w => P2), regions = Dict(w => Region(Ω, :left_half)))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    inside = sol[x] .<= 0.5 + 1.0e-12
    @test all(isfinite, sol[w(t, x, y)][:, inside]) && all(isnan, sol[w(t, x, y)][:, .!inside])
    @test sol[w(t, x, y)][end, inside] ≈ [1 + (a - 0.5)^2 + 0.5 for a in sol[x][inside]] atol = 1.0e-7
    @test sol(0.5, 0.3, 0.6; dv = w(t, x, y)) ≈ 1 + 0.2^2 + 0.5 atol = 1.0e-7
    @test isnan(@test_logs((:warn, r"assigned to cells where field :w is not defined"), sol(0.5, 0.8, 0.6; dv = w(t, x, y))))
    @test sol(0.5, 0.8, 0.6; dv = u(t, x, y)) ≈ 0.8^2 + 0.6^2 + 2 atol = 1.0e-7
end
