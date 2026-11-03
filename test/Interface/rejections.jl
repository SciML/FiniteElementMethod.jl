# Inputs the discretization refuses, each through the full interface: the PDESystem is built as a user
# would build it, and `discretize` (or the `FEMDiscretization` constructor) throws before any solver runs.
# Every case asserts the part of the documented message that names the problem (`@test_throws "text" expr`).
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: Ferrite
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t

@variables x y u(..) v(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
ic = u(0, x, y) ~ 0

Ω = MeshDomain(generate_grid(Triangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0)))
left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
dirichlet = [u(t, Γ) ~ 0 for Γ in (left, right, bottom, top)]

# Builds the PDESystem for `eqs` and `bcs` and discretizes it; the keywords go to `FEMDiscretization`.
function lower(eqs, bcs; mesh = Ω, dvs = [u(t, x, y)], domains = [(x, y) ∈ mesh, t ∈ Interval(0.0, 1.0)], ivs = [t, x, y], kw...)
    @named pdesys = PDESystem(eqs, bcs, domains, ivs, dvs, [])
    return discretize(pdesys, FEMDiscretization(mesh, t; kw...))
end

# PDE:    the 2D heat equation on the unit square, t in (0, 1), on a 2 x 2 triangle mesh
# BCs:    a malformed or unsupported boundary condition on one of the four sides
# Shows:  the message names the mistake and, where there is one, the form to write instead.
@testset "boundary conditions that cannot be imposed are rejected with the fix in the message" begin
    @test_throws "trace is missing the time argument; write `u(t, Γ_left)`" lower([heat], [u(left) ~ 0, ic])
    @test_throws "write a trace on a boundary piece instead" lower([heat], [u(t, 0, y) ~ 0, ic])
    @test_throws "which is not on Γ_left; write the trace on the operator's piece" lower(
        [heat], [NormalDerivative(left)(u(t, right)) ~ 0, ic]
    )
    @test_throws "periodic conditions need milestone M4" lower([heat], [u(t, left) ~ u(t, right), ic])
    @test_throws "Dirichlet data depends on another field" lower(
        [heat, Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))],
        [u(t, left) + v(t, left) ~ 1, ic, v(0, x, y) ~ 0]; dvs = [u(t, x, y), v(t, x, y)]
    )
end

# PDE:    heat equations with a nonlinear coefficient, a time-dependent coefficient, third order in space
#         or second order in time, with zero Dirichlet data
# Shows:  an operator the Galerkin weak form cannot assemble yet is named in the error.
@testset "equations outside the supported classes are rejected naming the term" begin
    nonlinear = Dt(u(t, x, y)) ~ Dx((1 + u(t, x, y)^2) * Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    @test_throws "nonlinear terms land in M3" lower([nonlinear], [dirichlet; ic])
    varying = Dt(u(t, x, y)) ~ Dx((1 + t) * Dx(u(t, x, y))) + Dy((1 + t) * Dy(u(t, x, y)))
    @test_throws "time-dependent coefficients in the mass or stiffness operator land in M3" lower([varying], [dirichlet; ic])
    @test_throws "third-order" lower([Dt(u(t, x, y)) ~ Dx(Dx(Dx(u(t, x, y))))], [dirichlet; ic])
    @test_throws "second order in time; write as a first-order system" lower(
        [Differential(t, 2)(u(t, x, y)) ~ Dx(Dx(u(t, x, y)))], [dirichlet; ic]
    )
end

# PDE:    the 2D heat equation, or a steady Laplace equation
# Shows:  options reserved for later releases, and unknown `spaces` keys, fail with the supported default named.
@testset "options that are not available yet are rejected when the discretization is built" begin
    @test_throws "codegen = :mtk lands in M3; use the default codegen = :numeric" FEMDiscretization(Ω, t; codegen = :mtk)
    @test_throws "dirichlet = :eliminate lands in M4; use the default dirichlet = :algebraic" FEMDiscretization(Ω, t; dirichlet = :eliminate)
    @named steady = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], [u(Γ) ~ 0 for Γ in (left, right, bottom, top)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    @test_throws "steady problems land in M4; give FEMDiscretization a time variable" discretize(steady, FEMDiscretization(Ω))
    @test_throws "`spaces` has keys that are not dependent variables of the system: v" lower(
        [heat], [dirichlet; ic]; spaces = Dict(v => Lagrange{RefTriangle, 2}())
    )
end

# PDE:    the 2D heat equation, with a Rectangle domain, a split time interval or time as the last argument
# Shows:  the PDESystem's domains must be a mesh for space and one interval for time, with time first.
@testset "domains the discretization cannot use are rejected naming what is expected" begin
    box = [(x, y) ∈ Rectangle([0.0, 0.0], [1.0, 1.0]), t ∈ Interval(0.0, 1.0)]
    @test_throws "FEMDiscretization discretizes on a mesh" lower([heat], [ic]; domains = box)
    split_time = [(x, y) ∈ Ω, t ∈ UnionDomain(Interval(0.0, 1.0), Interval(2.0, 3.0))]
    @test_throws "time t must be paired with an interval" lower([heat], [ic]; domains = split_time)
    time_last = Dt(u(x, y, t)) ~ Dx(Dx(u(x, y, t))) + Dy(Dy(u(x, y, t)))
    @test_throws "time must be the first argument of u(x, y, t)" lower([time_last], [u(x, y, 0) ~ 0]; dvs = [u(x, y, t)])
end

# PDE:    the 2D heat equation on the unit square
# BCs:    Dirichlet data on the facet sets "inlet" and "left", which are the same two facets
# Shows:  boundary pieces must be disjoint.
@testset "two boundary pieces that share facets are rejected naming the count" begin
    grid = generate_grid(Triangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0))
    Ferrite.addfacetset!(grid, "inlet", p -> p[1] ≈ 0)   # the same facets as "left"
    mesh = MeshDomain(grid)
    inlet, left_side = boundarypiece.(mesh, (:inlet, :left))
    @test_throws "share 2 facets" lower([heat], [u(t, inlet) ~ 0, u(t, left_side) ~ 1, ic]; mesh)
end
