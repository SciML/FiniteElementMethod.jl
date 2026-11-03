# The wave equation u_tt = c^2 Δu has a second time derivative, which the discretization rejects (see
# rejections.jl). Written as the first-order system u_t = v, v_t = c^2 Δu it is accepted: u has no spatial
# operator in its own equation, v carries the Laplacian. These cases solve the system for standing waves
# and compare both fields with the exact solution.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@parameters c
@variables x y u(..) v(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# Solves u_t = v, v_t = c^2 u_xx on a mesh of `n` line cells in order-`order` elements with wave speed `speed`;
# returns the L2 errors of u and v at t = 0.5 against u = cos(c pi t) sin(pi x), v = -c pi sin(c pi t) sin(pi x).
function wave_1d_errors(n, order, speed)
    Ω = MeshDomain(generate_grid(Line, (n,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    eqs = [Dt(u(t, x)) ~ v(t, x), Dt(v(t, x)) ~ c^2 * Dx(Dx(u(t, x)))]
    bcs = [u(t, left) ~ 0, u(t, right) ~ 0, v(t, left) ~ 0, v(t, right) ~ 0, u(0, x) ~ sin(pi * x), v(0, x) ~ 0]
    @named pdesys = PDESystem(eqs, bcs, [x ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, x], [u(t, x), v(t, x)], [c => speed])
    space = Lagrange{RefLine, order}()
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => space, v => space))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    points = sample_points(200)
    return (
        l2_error(sol, u(t, x), x_ -> cos(speed * pi * 0.5) * sin(pi * x_), 0.5, points),
        l2_error(sol, v(t, x), x_ -> -speed * pi * sin(speed * pi * 0.5) * sin(pi * x_), 0.5, points),
    )
end

# PDE:    u_t = v, v_t = c^2 u_xx with c = 1
# Domain: x in (0, 1), t in (0, 0.5), meshes of 4, 8, 16 and 32 line cells, P1 or P2 for both fields
# BCs:    u = v = 0 at both ends; ICs u(0, x) = sin(pi x), v(0, x) = 0
# Exact:  u = cos(pi t) sin(pi x), v = -pi sin(pi t) sin(pi x), a standing wave of period 2
# Shows:  the first-order form of the wave equation converges in both fields: with elements of order p, both
#         u and v converge at the optimal L2 rate p + 1 (u is superconvergent for P2, rate 4).
@testset "the first-order wave system converges in u and v with $name elements on a line" for (name, order) in (("P1", 1), ("P2", 2))
    ns = (4, 8, 16, 32)
    errors = [wave_1d_errors(n, order, 1.0) for n in ns]
    @test all(observed_rate(first.(errors), 1 ./ ns) .> order + 0.8)
    @test all(observed_rate(last.(errors), 1 ./ ns) .> order + 0.8)
end

# PDE:    u_t = v, v_t = c^2 u_xx with the wave speed c = 2 given in `ps`
# Domain: x in (0, 1), t in (0, 0.5), 16 P2 line cells
# BCs, ICs: as in the previous case
# Exact:  u = cos(2 pi t) sin(pi x), v = -2 pi sin(2 pi t) sin(pi x): the wave oscillates twice as fast
# Shows:  the wave speed is a parameter; the period is 1, so at t = 0.5 the wave has flipped sign:
#         u(0.5, x) = -sin(pi x), v(0.5, x) = 0, to the discretization error.
@testset "the wave speed c in ps sets the oscillation frequency" begin
    error_u, error_v = wave_1d_errors(16, 2, 2.0)
    @test error_u < 1.0e-4 && error_v < 1.0e-4
end

# PDE:    u_t = v, v_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.25), 8 x 8 triangle pairs, P2 for both fields
# BCs:    u = v = 0 on all four sides; ICs u(0, x, y) = sin(pi x) sin(pi y), v(0, x, y) = 0
# Exact:  u = cos(sqrt(2) pi t) sin(pi x) sin(pi y), v = -sqrt(2) pi sin(sqrt(2) pi t) sin(pi x) sin(pi y)
# Shows:  the same first-order system in 2D, with angular frequency sqrt(2) pi from the eigenvalue 2 pi^2.
@testset "a 2D standing wave on triangles is followed by both fields" begin
    Ω = MeshDomain(generate_grid(Triangle, (8, 8), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U, V = u(t, x, y), v(t, x, y)
    bcs = [
        [u(t, Γ) ~ 0 for Γ in sides]; [v(t, Γ) ~ 0 for Γ in sides];
        u(0, x, y) ~ sin(pi * x) * sin(pi * y); v(0, x, y) ~ 0
    ]
    @named pdesys = PDESystem([Dt(U) ~ V, Dt(V) ~ Dx(Dx(U)) + Dy(Dy(U))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.25)], [t, x, y], [U, V], [])
    space = Lagrange{RefTriangle, 2}()
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => space, v => space))), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    ω = sqrt(2) * pi
    points = (sample_points(30), sample_points(30))
    @test l2_error(sol, U, (x_, y_) -> cos(ω * 0.25) * sin(pi * x_) * sin(pi * y_), 0.25, points...) < 2.0e-3
    @test l2_error(sol, V, (x_, y_) -> -ω * sin(ω * 0.25) * sin(pi * x_) * sin(pi * y_), 0.25, points...) < 1.0e-2
end
