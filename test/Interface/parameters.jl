# Parameters of the PDESystem: values given in the `ps` list (`[D => 2.0, ...]`) enter the diffusivity, the
# source, the reaction rate and the boundary data. The numeric value is fixed when `discretize` assembles the
# operators, so each value is a new PDESystem (and a new discretization) and the exact solution tracks it.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@parameters D A g q λ
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# Solves `eq` on the unit square (n x n triangle pairs) with conditions `bcs(left, right, bottom, top)`, parameter values `ps`, to `tend`.
function solve_square(eq, bcs, ps, tend; n = 4)
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(eq, bcs(left, right, bottom, top), [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)], [t, x, y], [u(t, x, y)], ps)
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
    return solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = D (u_xx + u_yy) + A, with D, A and the boundary offset g given in `ps`
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2
# BCs:    u = x^2 + y^2 + (4D + A) t + g on all four sides; IC u(0, x, y) = x^2 + y^2 + g
# Exact:  u = x^2 + y^2 + (4D + A) t + g, which lies in the P2 space, so the error is the solver tolerance
# Shows:  the diffusivity, the source amplitude and the Dirichlet value are all read from `ps`, and the
#         computed solution follows the exact one for each of three parameter sets.
@testset "diffusivity, source amplitude and boundary value come from ps, for $ps" for ps in (
        (D = 1.0, A = 0.0, g = 0.0), (D = 0.5, A = 3.0, g = 1.0), (D = 4.0, A = -2.0, g = -0.5),
    )
    exact = x^2 + y^2 + (4D + A) * t + g
    eq = [Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) + A]
    bcs(sides...) = [[u(t, Γ) ~ exact for Γ in sides]; u(0, x, y) ~ x^2 + y^2 + g]
    sol = solve_square(eq, bcs, [D => ps.D, A => ps.A, g => ps.g], 1.0)
    final(x_, y_) = x_^2 + y_^2 + 4ps.D + ps.A + ps.g
    @test l2_error(sol, u(t, x, y), final, 1.0, sample_points(20), sample_points(20)) < 1.0e-7
end

# PDE:    u_t = D (u_xx + u_yy)
# Domain: the unit square, t in (0, 1), P2
# BCs:    u = D q t on the left; u_x = q on the right (flux value from `ps`); u_y = 0 on top and bottom
#         (Neumann data enters through `NormalDerivative`); IC u(0, x, y) = q x^2 / 2
# Exact:  u = q x^2 / 2 + D q t, in the P2 space
# Shows:  a parameter can be the value of a Neumann condition, and the solution scales with it.
@testset "a Neumann flux read from ps scales the solution, for q = $flux" for flux in (0.5, 2.0, -1.0)
    diffusivity = 1.5
    eq = [Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))]
    bcs(left, right, bottom, top) = [
        u(t, left) ~ D * q * t, NormalDerivative(right)(u(t, right)) ~ q,
        NormalDerivative(bottom)(u(t, bottom)) ~ 0, NormalDerivative(top)(u(t, top)) ~ 0, u(0, x, y) ~ q * x^2 / 2,
    ]
    sol = solve_square(eq, bcs, [D => diffusivity, q => flux], 1.0)
    final(x_, y_) = flux * x_^2 / 2 + diffusivity * flux
    @test l2_error(sol, u(t, x, y), final, 1.0, sample_points(20), sample_points(20)) < 1.0e-7
end

# PDE:    u_t = D (u_xx + u_yy) - λ u
# Domain: the unit square, t in (0, 0.05), 16 x 16 triangle pairs, P2
# BCs:    u = 0 on all four sides; IC u(0, x, y) = sin(pi x) sin(pi y)
# Exact:  u = exp(-(2 pi^2 D + λ) t) sin(pi x) sin(pi y)
# Shows:  a parameter in the reaction term changes the decay rate, and the final amplitude at the centre
#         tracks exp(-(2 pi^2 D + λ) t) for each pair (D, λ).
@testset "the decay rate of an eigenmode follows D and the reaction rate λ in ps, for $ps" for ps in (
        (D = 1.0, λ = 0.0), (D = 0.5, λ = 10.0), (D = 2.0, λ = 3.0),
    )
    eq = [Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) - λ * u(t, x, y)]
    bcs(sides...) = [[u(t, Γ) ~ 0 for Γ in sides]; u(0, x, y) ~ sin(pi * x) * sin(pi * y)]
    sol = solve_square(eq, bcs, [D => ps.D, λ => ps.λ], 0.05; n = 16)
    final(x_, y_) = exp(-(2pi^2 * ps.D + ps.λ) * 0.05) * sin(pi * x_) * sin(pi * y_)
    @test l2_error(sol, u(t, x, y), final, 0.05, sample_points(30), sample_points(30)) < 1.0e-4
end
