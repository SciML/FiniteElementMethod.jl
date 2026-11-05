# Time as the last argument of the dependent variable, `u(x, y, t)`, as MethodOfLines accepts it: the front end and
# the solution follow the declared argument order, so arrays put the time axis last and calls take the time last.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

unit_square(n) = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
exact(t_, x_, y_) = exp(-2pi^2 * t_) * sin(pi * x_) * sin(pi * y_)

# Solves the heat equation on a P2 mesh of 8 x 8 triangle pairs. `U(args...)` is u(t, args...) or u(args..., t), and `ic` the
# matching initial condition.
function solve_heat(U, ic)
    Ω = unit_square(8)
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    eq = Dt(U(x, y)) ~ Dx(Dx(U(x, y))) + Dy(Dy(U(x, y)))
    bcs = [[U(Γ) ~ 0 for Γ in sides]; ic]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [U(x, y)], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()), saveat = 0.025)
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = u_xx + u_yy, written once as u(t, x, y) and once as u(x, y, t)
# Domain: the unit square, t in (0, 0.1), 8 x 8 triangle pairs, P2; saved every 0.025
# BCs:    u = 0 on all sides; IC u = sin(pi x) sin(pi y) at t = 0
# Exact:  u = exp(-2 pi^2 t) sin(pi x) sin(pi y)
# Shows:  both forms give the same solution; `sol[u]` is nt x nvertices time-first and nvertices x nt time-last,
#         and `sol(t, x, y; dv)` becomes `sol(x, y, t; dv)`.
@testset "the heat equation written with u(x, y, t) gives the u(t, x, y) solution with the time axis last" begin
    start = sin(pi * x) * sin(pi * y)
    time_first = solve_heat((args...) -> u(t, args...), u(0, x, y) ~ start)
    time_last = solve_heat((args...) -> u(args..., t), u(x, y, 0) ~ start)
    Uf, Ul = u(t, x, y), u(x, y, t)
    nt, nv = length(time_first.t), length(time_first[x])
    @test time_last.t == time_first.t
    @test size(time_first[Uf]) == (nt, nv) && size(time_last[Ul]) == (nv, nt)
    @test time_last[Ul] ≈ permutedims(time_first[Uf]) atol = 1.0e-10
    @test time_last(0.3, 0.4, 0.05; dv = Ul) ≈ time_first(0.05, 0.3, 0.4; dv = Uf) atol = 1.0e-10
    @test time_last(0.3, 0.4, 0.05; dv = Ul) ≈ exact(0.05, 0.3, 0.4) atol = 1.0e-3
    xs, ys, ts = range(0, 1, length = 5), range(0, 1, length = 4), [0.0, 0.05, 0.1]
    @test size(time_last(xs, ys, ts; dv = Ul)) == (5, 4, 3)
    @test time_last(xs, ys, ts; dv = Ul) ≈ permutedims(time_first(ts, xs, ys; dv = Uf), (2, 3, 1)) atol = 1.0e-10
end

# PDE:    u_t = u_xx + u_yy + 2 t (1 + x + 2y)
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2
# BCs:    Dirichlet u = (1 + x + 2y)(1 + t^2) on left and bottom; Neumann du/dn = 1 + t^2 on right and 2(1 + t^2) on top,
#         all written with the trace `u(Γ, t)`; IC u(x, y, 0) = 1 + x + 2y
# Exact:  u = (1 + x + 2y)(1 + t^2), in the P2 space
# Shows:  traces and normal derivatives take the time last too, and a time-dependent source works.
@testset "time-last traces carry Dirichlet and Neumann data and the solution is exact" begin
    Ω = unit_square(4)
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U = u(x, y, t)
    eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + 2t * (1 + x + 2y)
    bcs = [
        u(left, t) ~ (1 + 2y) * (1 + t^2), u(bottom, t) ~ (1 + x) * (1 + t^2),
        NormalDerivative(right)(u(right, t)) ~ 1 + t^2, NormalDerivative(top)(u(top, t)) ~ 2 * (1 + t^2),
        u(x, y, 0) ~ 1 + x + 2y,
    ]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    xs = range(0.1, 0.9, length = 5)
    @test sol(xs, 0.3, 0.7; dv = U) ≈ [(1 + a + 0.6) * (1 + 0.7^2) for a in xs] atol = 1.0e-6
    @test sol[U][:, end] ≈ [(1 + a + 2b) * 2 for (a, b) in zip(sol[x], sol[y])] atol = 1.0e-6
end
