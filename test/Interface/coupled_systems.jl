# Coupled two-field systems through the full interface: two dependent variables u and v of equal order, each
# with its own equation, boundary conditions and initial condition, coupled through the right-hand sides.
# `spaces` names a Lagrange space for each field; the solution is read per field with `dv`.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Quadrilateral, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@parameters a b
@variables x y u(..) v(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
U, V = u(t, x, y), v(t, x, y)
laplacian(f) = Dx(Dx(f)) + Dy(Dy(f))

unit_square(celltype, n) = MeshDomain(generate_grid(celltype, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))

# Solves the system `eqs` for (u, v) on Ω with conditions `bcs` to `tend`, both fields in `space`.
function solve_pair(eqs, bcs, Ω, space, tend, ps = [])
    domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [U, V], ps)
    prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => space, v => space)))
    return solve(prob, DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# PDE:    u_t = Δu + a v + f1,  v_t = Δv - b u + f2, with a = 2, b = 3 in `ps`
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2 for both fields
# BCs:    u and v equal to their exact values on all four sides; ICs from the exact values at t = 0
# Exact:  u = x^2 + y^2 + t, v = x y + 2t; the sources f1, f2 are chosen so these solve the system
# Shows:  the two fields are coupled in both directions (u feeds v and v feeds u), each is read back with
#         its own `dv`, and a solution inside the element space is reproduced to the solver tolerance.
@testset "a bidirectionally coupled pair is reproduced exactly by P2 elements" begin
    Ω = unit_square(Triangle, 4)
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    u_exact, v_exact = x^2 + y^2 + t, x * y + 2t
    eqs = [
        Dt(U) ~ laplacian(U) + a * V + (1 - 4 - a * v_exact),
        Dt(V) ~ laplacian(V) - b * U + (2 + b * u_exact),
    ]
    bcs = [
        [u(t, Γ) ~ u_exact for Γ in sides]; [v(t, Γ) ~ v_exact for Γ in sides];
        u(0, x, y) ~ x^2 + y^2; v(0, x, y) ~ x * y
    ]
    sol = solve_pair(eqs, bcs, Ω, Lagrange{RefTriangle, 2}(), 1.0, [a => 2.0, b => 3.0])
    points = (sample_points(20), sample_points(20))
    @test l2_error(sol, U, (x_, y_) -> x_^2 + y_^2 + 1, 1.0, points...) < 1.0e-7
    @test l2_error(sol, V, (x_, y_) -> x_ * y_ + 2, 1.0, points...) < 1.0e-7
end

# PDE:    u_t = Δu + v,  v_t = Δv + u
# Domain: the unit square, t in (0, 0.05), 16 x 16 cells, P2 for both fields, triangles and quadrilaterals
# BCs:    u = v = 0 on all four sides; IC u = sin(pi x) sin(pi y), v = 0
# Exact:  u = e^(-2 pi^2 t) cosh(t) φ, v = e^(-2 pi^2 t) sinh(t) φ with φ = sin(pi x) sin(pi y)
# Shows:  the coupling u <-> v turns the single eigenmode φ into the pair (cosh t, sinh t), so a correct
#         off-diagonal block gives v a nonzero amplitude although v starts at zero.
@testset "the coupled eigenmode pair (cosh, sinh) is followed on $name meshes" for (name, celltype, space) in (
        ("triangle", Triangle, Lagrange{RefTriangle, 2}()), ("quadrilateral", Quadrilateral, Lagrange{RefQuadrilateral, 2}()),
    )
    Ω = unit_square(celltype, 16)
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    eqs = [Dt(U) ~ laplacian(U) + V, Dt(V) ~ laplacian(V) + U]
    bcs = [
        [u(t, Γ) ~ 0 for Γ in sides]; [v(t, Γ) ~ 0 for Γ in sides];
        u(0, x, y) ~ sin(pi * x) * sin(pi * y); v(0, x, y) ~ 0
    ]
    sol = solve_pair(eqs, bcs, Ω, space, 0.05)
    φ(x_, y_) = sin(pi * x_) * sin(pi * y_)
    points = (sample_points(40), sample_points(40))
    @test l2_error(sol, U, (x_, y_) -> exp(-2pi^2 * 0.05) * cosh(0.05) * φ(x_, y_), 0.05, points...) < 1.0e-4
    @test l2_error(sol, V, (x_, y_) -> exp(-2pi^2 * 0.05) * sinh(0.05) * φ(x_, y_), 0.05, points...) < 1.0e-4
end

# PDE:    u_t = Δu + v + f1,  v_t = Δv + u + f2
# Domain: the unit square, t in (0, 0.5), 4, 8 and 16 triangle pairs per side, P1 for both fields
# BCs:    u and v equal to their exact values on all four sides
# Exact:  u = e^(x + y + t), v = cos(x) e^(y - t) (not in the element space); f1, f2 from the equations
# Shows:  each field of a coupled P1 system converges at the optimal L2 rate 2.
@testset "both fields of a coupled P1 system converge at rate 2" begin
    u_exact, v_exact = exp(x + y + t), cos(x) * exp(y - t)
    eqs = [
        Dt(U) ~ laplacian(U) + V + expand_derivatives(Dt(u_exact) - laplacian(u_exact) - v_exact),
        Dt(V) ~ laplacian(V) + U + expand_derivatives(Dt(v_exact) - laplacian(v_exact) - u_exact),
    ]
    ns = (4, 8, 16)
    errors = map(ns) do n
        Ω = unit_square(Triangle, n)
        sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        bcs = [
            [u(t, Γ) ~ u_exact for Γ in sides]; [v(t, Γ) ~ v_exact for Γ in sides];
            u(0, x, y) ~ exp(x + y); v(0, x, y) ~ cos(x) * exp(y)
        ]
        sol = solve_pair(eqs, bcs, Ω, Lagrange{RefTriangle, 1}(), 0.5)
        points = (sample_points(50), sample_points(50))
        [
            l2_error(sol, U, (x_, y_) -> exp(x_ + y_ + 0.5), 0.5, points...),
            l2_error(sol, V, (x_, y_) -> cos(x_) * exp(y_ - 0.5), 0.5, points...),
        ]
    end
    @test all(observed_rate(first.(errors), 1 ./ ns) .> 1.9)
    @test all(observed_rate(last.(errors), 1 ./ ns) .> 1.9)
end

# PDE:    u_t = Δu + a v + f1,  v_t = Δv - b u + f2 (as in the first case)
# Domain: the unit square, t in (0, 1), P2 for both fields
# BCs:    u Dirichlet on all sides; v Dirichlet on left and bottom, ∂ₙv = y on right and ∂ₙv = x on top (Neumann)
# Exact:  u = x^2 + y^2 + t, v = x y + 2t
# Shows:  each field has its own boundary conditions: a Neumann condition on v while u is held by Dirichlet data.
@testset "the two fields carry different kinds of boundary condition" begin
    Ω = unit_square(Triangle, 4)
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    u_exact, v_exact = x^2 + y^2 + t, x * y + 2t
    eqs = [
        Dt(U) ~ laplacian(U) + a * V + (1 - 4 - a * v_exact),
        Dt(V) ~ laplacian(V) - b * U + (2 + b * u_exact),
    ]
    bcs = [
        [u(t, Γ) ~ u_exact for Γ in (left, right, bottom, top)];
        v(t, left) ~ v_exact; v(t, bottom) ~ v_exact;
        NormalDerivative(right)(v(t, right)) ~ y; NormalDerivative(top)(v(t, top)) ~ x;
        u(0, x, y) ~ x^2 + y^2; v(0, x, y) ~ x * y
    ]
    sol = solve_pair(eqs, bcs, Ω, Lagrange{RefTriangle, 2}(), 1.0, [a => 2.0, b => 3.0])
    points = (sample_points(20), sample_points(20))
    @test l2_error(sol, U, (x_, y_) -> x_^2 + y_^2 + 1, 1.0, points...) < 1.0e-7
    @test l2_error(sol, V, (x_, y_) -> x_ * y_ + 2, 1.0, points...) < 1.0e-7
end
