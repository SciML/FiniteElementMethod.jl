# Inputs the discretization refuses, each through the full interface: the PDESystem is built as a user
# would build it, and `discretize` (or the `FEMDiscretization` constructor) throws before any solver runs.
# Every case asserts the part of the documented message that names the problem (`@test_throws "text" expr`).
# The last cases, on solution queries, solve a four-cell problem first.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: Ferrite
using Ferrite: generate_grid, Line, Triangle, Vec
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
    @test_throws "periodic boundary conditions are not supported" lower([heat], [u(t, left) ~ u(t, right), ic])
    @test_throws "Dirichlet data depends on another field" lower(
        [heat, Dt(v(t, x, y)) ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))],
        [u(t, left) + v(t, left) ~ 1, ic, v(0, x, y) ~ 0]; dvs = [u(t, x, y), v(t, x, y)]
    )
end

# PDE:    heat equations with a time-dependent coefficient, a state-dependent mass, third order in space, second
#         order in time, or a nonlinear Dirichlet condition, with zero Dirichlet data
# Shows:  an operator the Galerkin weak form cannot assemble is named in the error.
@testset "equations outside the supported classes are rejected naming the term" begin
    varying = Dt(u(t, x, y)) ~ Dx((1 + t) * Dx(u(t, x, y))) + Dy((1 + t) * Dy(u(t, x, y)))
    @test_throws "time-dependent coefficients in the mass or stiffness operator are not supported" lower([varying], [dirichlet; ic])
    @test_throws "state-dependent mass" lower([(1 + u(t, x, y)^2) * Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))], [dirichlet; ic])
    @test_throws "third-order" lower([Dt(u(t, x, y)) ~ Dx(Dx(Dx(u(t, x, y))))], [dirichlet; ic])
    @test_throws "second order in time; write as a first-order system" lower(
        [Differential(t, 2)(u(t, x, y)) ~ Dx(Dx(u(t, x, y)))], [dirichlet; ic]
    )
    @test_throws "Dirichlet data must be linear in one field" lower([heat], [u(t, left)^2 ~ 1, u(t, right) ~ 0, u(t, bottom) ~ 0, u(t, top) ~ 0, ic])
end

# PDE:    the 2D heat equation, or a steady Laplace equation
# Shows:  options that do not exist, a time-dependent condition with `dirichlet = :eliminate`, and a steady problem
#         paired with a time-dependent discretization (or the reverse) fail naming the fix.
@testset "options and discretizations that do not fit the problem are rejected when the problem is built" begin
    @test_throws "codegen = :mtk is not supported; use the default codegen = :numeric" FEMDiscretization(Ω, t; codegen = :mtk)
    @test_throws "dirichlet must be :algebraic or :eliminate, got :condense" FEMDiscretization(Ω, t; dirichlet = :condense)
    timed = [u(t, Γ) ~ sin(t) for Γ in (left, right, bottom, top)]
    @test_throws "dirichlet = :eliminate needs Dirichlet data that are constant in time" lower([heat], [timed; ic]; dirichlet = :eliminate)
    @named steady = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], [u(Γ) ~ 0 for Γ in (left, right, bottom, top)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
    @named transient = PDESystem([heat], [dirichlet; ic], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    @test_throws "for a time-dependent problem write FEMDiscretization(Ω, t)" discretize(transient, FEMDiscretization(Ω))
    @test_throws "`spaces` has keys that are not dependent variables of the system: v" lower(
        [heat], [dirichlet; ic]; spaces = Dict(v => Lagrange{RefTriangle, 2}())
    )
end

# PDE:    the 2D heat equation, with a Rectangle domain, a split time interval or time between the spatial arguments
# Shows:  the PDESystem's domains must be a mesh for space and one interval for time; time is the first or last argument.
@testset "domains the discretization cannot use are rejected naming what is expected" begin
    box = [(x, y) ∈ Rectangle([0.0, 0.0], [1.0, 1.0]), t ∈ Interval(0.0, 1.0)]
    @test_throws "FEMDiscretization discretizes on a mesh" lower([heat], [ic]; domains = box)
    split_time = [(x, y) ∈ Ω, t ∈ UnionDomain(Interval(0.0, 1.0), Interval(2.0, 3.0))]
    @test_throws "time t must be paired with an interval" lower([heat], [ic]; domains = split_time)
    between = Dt(u(x, t, y)) ~ Dx(Dx(u(x, t, y))) + Dy(Dy(u(x, t, y)))
    @test_throws "time must be the first or the last argument of u(x, t, y)" lower([between], [u(x, 0, y) ~ 0]; dvs = [u(x, t, y)])
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

Ω1 = MeshDomain(generate_grid(Line, (4,), Vec(0.0), Vec(1.0)))
ends = boundarypiece.(Ω1, (:left, :right))
line_system(eq) = PDESystem([eq], [u(t, ends[1]) ~ 0, u(t, ends[2]) ~ 1, u(0, x) ~ x], [(x,) ∈ Ω1, t ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)], []; name = :line)

# PDE:    u_t = 0.01 u_xx - u_x - u^2, discretized with SUPG; and the options of `test_space`
# Shows:  SUPG stabilises linear terms only, and `test_space` must be `Galerkin()` or `SUPG(τ)`.
@testset "SUPG with a nonlinear term, or a test space that is not Galerkin or SUPG, is rejected" begin
    nonlinear = Dt(u(t, x)) ~ 0.01 * Dx(Dx(u(t, x))) - Dx(u(t, x)) - u(t, x)^2
    @test_throws "SUPG stabilises linear terms only" discretize(line_system(nonlinear), FEMDiscretization(Ω1, t; test_space = SUPG(0.05)))
    @test_throws "test_space must be Galerkin() or SUPG(τ; b = nothing), got :supg" FEMDiscretization(Ω1, t; test_space = :supg)
end

# PDE:    u_t = u_xx on a line, solved, then queried with expressions the solution cannot evaluate
# Shows:  an observed expression may contain only the dependent variables, their first spatial derivatives, the
#         coordinates, time and parameters; anything else is named in the error.
@testset "observed expressions with unknown symbols, second derivatives or fixed arguments are rejected" begin
    sol = solve(discretize(line_system(Dt(u(t, x)) ~ Dx(Dx(u(t, x)))), FEMDiscretization(Ω1, t)), DFBDF())
    @test_throws "is not a dependent variable, an independent variable or a parameter" sol[u(t, x) * v(t, x)]
    @test_throws "is not a first spatial derivative of a dependent variable" sol[Dx(Dx(u(t, x))) + 1]
    @test_throws "fixes arguments of u(t, x); write the dependent variable as u(t, x)" sol[u(0, x) + 1]
    @test_throws "is not a dependent variable or a first spatial derivative of one" sol(0.5, 0.5; dv = Dt(u(t, x)))
end

# PDE:    u_t = u_xx - u^3
# Shows:  `operators` returns the linear time-independent operators only; a nonlinear term is named.
@testset "operators rejects a nonlinear term" begin
    nonlinear = Dt(u(t, x)) ~ Dx(Dx(u(t, x))) - u(t, x)^3
    @test_throws "operators returns linear operators only" FiniteElementMethod.operators(line_system(nonlinear), FEMDiscretization(Ω1, t))
end
