# Streamline-upwind Petrov-Galerkin stabilisation for convection-dominated problems: `test_space = SUPG(τ)` tests each
# equation with v + τ b.grad(v), where b is the advection velocity read from the equation. Plain Galerkin behaves like a
# centred scheme and oscillates once the cell Peclet number exceeds one.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Line, Vec
using ModelingToolkit: t_nounits as t

@variables x u(..)
Dt, Dx = Differential(t), Differential(x)

const ε = 0.01
layer(x_) = expm1(x_ / ε) / expm1(1 / ε)

# Marches u_t = ε u_xx - u_x to its steady state at t = 5 on 20 line cells (cell Peclet number 2.5) and returns the
# nodal values of u and their largest distance from the exact boundary layer.
function boundary_layer(test_space)
    Ω = MeshDomain(generate_grid(Line, (20,), Vec(0.0), Vec(1.0)))
    left, right = boundarypiece.(Ω, (:left, :right))
    eq = Dt(u(t, x)) ~ ε * Dx(Dx(u(t, x))) - Dx(u(t, x))
    @named pdesys = PDESystem([eq], [u(t, left) ~ 0, u(t, right) ~ 1, u(0, x) ~ x], [(x,) ∈ Ω, t ∈ Interval(0.0, 5.0)], [t, x], [u(t, x)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t; test_space)), DFBDF(); abstol = 1.0e-11, reltol = 1.0e-11)
    nodal = sol[u(t, x)][end, :]
    return nodal, maximum(abs.(nodal .- layer.(sol[x])))
end

# PDE:    u_t = ε u_xx - u_x with ε = 0.01 (convection-diffusion, advection velocity 1)
# Domain: x in (0, 1), t in (0, 5), 20 line cells, P1
# BCs:    u(t, 0) = 0, u(t, 1) = 1; IC u(0, x) = x
# Exact:  the steady boundary layer u = (exp(x / ε) - 1) / (exp(1 / ε) - 1)
# Shows:  Galerkin undershoots below -0.4 at this Peclet number; SUPG with the optimal τ(hK, b) is nodally exact.
@testset "SUPG removes the oscillations of Galerkin in a boundary layer and is nodally exact with the optimal τ" begin
    nodal, error_galerkin = boundary_layer(Galerkin())
    @test minimum(nodal) < -0.4 && error_galerkin > 0.4
    optimal_τ(hK, b) = (Pe = abs(b[1]) * hK / (2ε); hK / (2abs(b[1])) * (coth(Pe) - 1 / Pe))
    nodal, error_supg = boundary_layer(SUPG(optimal_τ))
    @test minimum(nodal) >= -1.0e-10 && maximum(nodal) <= 1 + 1.0e-10
    @test error_supg < 1.0e-9
end

# PDE, Domain, BCs, Exact: as in the first case
# Shows:  τ may be a plain number with the velocity `b` given, or a function of the cell size and velocity, here hK / (2|b|);
#         both are monotone and smear the layer over about a cell, and the two agree on this uniform mesh (hK = 1/20).
@testset "SUPG accepts a constant τ with an explicit velocity or a function of the cell size" begin
    nodal_function, error_function = boundary_layer(SUPG((hK, b) -> hK / (2abs(b[1]))))
    nodal_constant, _ = boundary_layer(SUPG(1 / 40; b = [1.0]))
    @test minimum(nodal_function) >= -1.0e-10 && maximum(nodal_function) <= 1 + 1.0e-10
    @test error_function < 0.2
    @test nodal_constant ≈ nodal_function atol = 1.0e-8
end
