# Observed expressions: any expression of the dependent variables, their first spatial derivatives, the coordinates,
# time and the parameters can be a key of the solution, `sol[expr]`, or the `dv` of a point evaluation,
# `sol(t, xs...; dv = expr)`. The expression is evaluated at every mesh vertex (or point) from the finite-element fields.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t

@parameters D
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# PDE:    u_t = D (u_xx + u_yy) with D = 2
# Domain: the unit square, t in (0, 1), 4 x 4 triangle pairs, P2; saved every 0.25
# BCs:    u = x^2 + y^2 + 4 D t on all sides; IC u(0, x, y) = x^2 + y^2
# Exact:  u = x^2 + y^2 + 4 D t, so grad u = (2x, 2y) and the expressions below have closed forms
# Shows:  `sol[expr]` and `sol(t, x, y; dv = expr)` for expressions of u, grad u, coordinates, time and a parameter.
@testset "expressions of the solution, its gradient, the coordinates, time and a parameter are observed" begin
    Ω = MeshDomain(generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U = u(t, x, y)
    @named pdesys = PDESystem(
        [Dt(U) ~ D * (Dx(Dx(U)) + Dy(Dy(U)))], [[u(t, Γ) ~ x^2 + y^2 + 4D * t for Γ in sides]; u(0, x, y) ~ x^2 + y^2],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], [D => 2.0]
    )
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()), saveat = 0.25)
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    T, X, Y = sol.t, permutedims(sol[x]), permutedims(sol[y])

    @test size(sol[U^2 + D * x]) == (length(T), length(X))
    @test sol[U^2 + D * x] ≈ sol[U] .^ 2 .+ 2 .* X atol = 1.0e-12
    @test sol[U^2 + D * x] ≈ (X .^ 2 .+ Y .^ 2 .+ 8 .* T) .^ 2 .+ 2 .* X atol = 1.0e-5
    @test sol[Dx(U)^2 + Dy(U)^2] ≈ repeat(4 .* (X .^ 2 .+ Y .^ 2), length(T)) atol = 1.0e-6

    @test sol(0.5, 0.3, 0.2; dv = U - x^2 - y^2) ≈ 8 * 0.5 atol = 1.0e-6
    xs, ys = range(0, 1, length = 5), range(0, 1, length = 4)
    @test sol(T, xs, ys; dv = t * U) ≈ [T[n] * (a^2 + b^2 + 8T[n]) for n in eachindex(T), a in xs, b in ys] atol = 1.0e-6
    @test isnan(sol(0.5, 2.0, 0.0; dv = 2x))
end
