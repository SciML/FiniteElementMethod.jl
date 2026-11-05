# Neumann, Robin and mixed boundary conditions on the sides of a rectangle, through the full
# interface. The manufactured solution u = x^2 + y^2 + 4t solves u_t = u_xx + u_yy without a source
# and lies in the P2 and Q2 spaces, so each case must reproduce it to the solver tolerance. Its
# outward normal derivatives on the unit square are 2 (right), 2 (top), 0 (left) and 0 (bottom).
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y u(..)
@parameters k α
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))

# Solves u_t = k (u_xx + u_yy) with k = 1 and α = 3 under `bcs(left, right, bottom, top)` and compares with the exact solution.
function exact_error(bcs, celltype, ref)
    Ω = unit_square(celltype, 4)
    eq = Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y)))
    conditions = [bcs(boundarypiece.(Ω, (:left, :right, :bottom, :top))...); u(0, x, y) ~ x^2 + y^2]
    @named pdesys = PDESystem([eq], conditions, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [k => 1.0, α => 3.0])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{ref, 2}()))
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    return l2_error(sol, u(t, x, y), (x_, y_) -> x_^2 + y_^2 + 4, 1.0, sample_points(20), sample_points(20))
end

cells = ((Triangle, RefTriangle), (Quadrilateral, RefQuadrilateral))

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 1)
# BCs:    Dirichlet u = x^2 + y^2 + 4t on the left; Neumann du/dn = 2, 2, 0 on the right, top, bottom
# Exact:  u = x^2 + y^2 + 4t
# Shows:  a Neumann condition is written with NormalDerivative(Γ) and reproduces the exact flux.
@testset "Neumann conditions on three sides reproduce the exact solution on $(first(c)) meshes" for c in cells
    err = exact_error(c...) do left, right, bottom, top
        [
            u(t, left) ~ y^2 + 4t, NormalDerivative(right)(u(t, right)) ~ 2,
            NormalDerivative(bottom)(u(t, bottom)) ~ 0, NormalDerivative(top)(u(t, top)) ~ 2,
        ]
    end
    @test err < 1.0e-7
end

# PDE:    u_t = k (u_xx + u_yy), with the parameter k = 1
# Domain: the unit square, t in (0, 1)
# BCs:    Dirichlet on the left, bottom and top; Robin k du/dn + alpha u = alpha (1 + y^2 + 4t) + 2 k on the right
# Exact:  u = x^2 + y^2 + 4t
# Shows:  a Robin condition with an explicit conormal coefficient k and a parameter alpha.
@testset "a Robin condition with parameters reproduces the exact solution on $(first(c)) meshes" for c in cells
    err = exact_error(c...) do left, right, bottom, top
        [
            u(t, left) ~ y^2 + 4t, u(t, bottom) ~ x^2 + 4t, u(t, top) ~ x^2 + 1 + 4t,
            k * NormalDerivative(right)(u(t, right)) + α * u(t, right) ~ α * (1 + y^2 + 4t) + 2k,
        ]
    end
    @test err < 1.0e-7
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 1)
# BCs:    Dirichlet on the left, Neumann on the bottom, Robin du/dn + alpha u = ... on the top and right
# Exact:  u = x^2 + y^2 + 4t
# Shows:  Dirichlet, Neumann and Robin conditions on different sides of one rectangle.
@testset "Dirichlet, Neumann and Robin sides of one rectangle reproduce the exact solution on $(first(c)) meshes" for c in cells
    err = exact_error(c...) do left, right, bottom, top
        [
            u(t, left) ~ y^2 + 4t, NormalDerivative(bottom)(u(t, bottom)) ~ 0,
            k * NormalDerivative(top)(u(t, top)) + α * u(t, top) ~ α * (x^2 + 1 + 4t) + 2k,
            k * NormalDerivative(right)(u(t, right)) + α * u(t, right) ~ α * (1 + y^2 + 4t) + 2k,
        ]
    end
    @test err < 1.0e-7
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.1)
# BCs:    zero flux on all four sides; IC u(0, x, y) = 1 + cos(pi x) cos(pi y)
# Exact:  u = 1 + exp(-2 pi^2 t) cos(pi x) cos(pi y)
# Shows:  a fully insulated box relaxes to its mean value 1 along the exact cosine mode.
@testset "a fully insulated box decays along its cosine mode on triangle meshes" begin
    Ω = unit_square(Triangle, 16)
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    bcs = [[NormalDerivative(Γ)(u(t, Γ)) ~ 0 for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ 1 + cos(pi * x) * cos(pi * y)]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [u(t, x, y)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t)), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    exact(x_, y_) = 1 + exp(-2pi^2 * 0.1) * cos(pi * x_) * cos(pi * y_)
    @test l2_error(sol, u(t, x, y), exact, 0.1, sample_points(50), sample_points(50)) < 5.0e-3
end
