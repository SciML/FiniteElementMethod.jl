# Discretization

`FEMDiscretization`, `Galerkin` and `SUPG` are exported, as are the PDEBase names a model needs
(`MeshDomain`, `Region`, `boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's
`Lagrange` and reference shapes.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.SciMLBase.NonlinearProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.SciMLBase.symbolic_discretize(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
FiniteElementMethod.assemble_sources!
```

## Multi-field systems, unequal orders and regions

Each dependent variable is one field with its own interpolation, given in `spaces`; fields
without an entry get `Lagrange{RefShape, 1}()`. All fields share one quadrature rule, of order
`2p` for the highest order `p` unless `quadrature` is given. A Taylor–Hood Stokes system pairs
a quadratic velocity with a linear pressure:

```julia
@variables x y u1(..) u2(..) p(..)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))
continuity = 0 ~ Dx(u1(t, x, y)) + Dy(u2(t, x, y))
eqs = [
    Dt(u1(t, x, y)) ~ Δ(u1(t, x, y)) - Dx(p(t, x, y)) + f1,
    Dt(u2(t, x, y)) ~ Δ(u2(t, x, y)) - Dy(p(t, x, y)) + f2,
    continuity,
]
P2, P1 = Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 1}()
disc = FEMDiscretization(Ω, t; spaces = Dict(u1 => P2, u2 => P2, p => P1), pairing = Dict(continuity => p))
```

The continuity equation has no time derivative and no pressure, so `pairing` names the field
it is tested with. The pressure rows are algebraic, and with Dirichlet velocity on the whole
boundary the pressure is fixed only up to a constant: give it a Dirichlet condition on one
small piece (a single facet set), or leave an outflow piece with the natural condition.

The pressure is absent from its own rows, so the DAE has index 2. Two things make `DFBDF` step
it: start from a velocity that is discretely divergence free (zero, with zero Dirichlet data
at the initial time) and a pressure initial condition, and leave the algebraic rows out of the
error control with a large absolute tolerance on them, since the pressure's local error does
not shrink with the step. `FEMDiscretization` does the second by default: with
`algebraic_abstol = :auto` it detects an algebraic field absent from its own equation and gives
the problem the `abstol` vector `1e8` on every algebraic row (the algebraic fields and the
Dirichlet rows of all fields) and `1e-6` elsewhere:

```julia
prob = discretize(pdesys, disc)        # prob.kwargs[:abstol] is the vector
sol = solve(prob, DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0)); reltol = 1.0e-8)
```

A number `abstol` given to `FEMDiscretization` or `discretize` sets the differential rows'
entries, as in `discretize(pdesys, disc; abstol = 1.0e-8)`; an array is used as it is.
`algebraic_abstol = 1.0e4` sets another value on the algebraic rows, of any system, and
`algebraic_abstol = nothing` sets none. An `abstol` passed to `solve` replaces the whole vector
(SciMLBase merges the problem's keywords first and the `solve` keywords over them), so
`solve(prob, alg; abstol = 1.0e-8)` puts the algebraic rows back under error control; write the
vector by hand in that case, `abstol = [d ? 1.0e-8 : 1.0e8 for d in prob.differential_vars]`.
Systems without an index-2 field, such as `0 ~ Δv + u` paired with `v`, get no `abstol`.
The keyword applies to the time-dependent (DAE) path only: a steady `FEMDiscretization(Ω)`
accepts `:auto` and `nothing` and does nothing with them, and raises for a number.

Without the large tolerance, the step size collapses near the initial time (`dt` below `eps`
at `t ≈ 1e-10` on a 16 × 16 Taylor–Hood mesh). When the linear solver then reports the Newton
matrix as singular, LinearSolve falls back to a sparse QR factorization, which can take
minutes and tens of gigabytes on a saddle-point system.

`algebraic_abstol` is not in the spec (§7.1, §7.6 list the `FEMDiscretization` keywords
without it): it was added after the Taylor–Hood Stokes and DFG cylinder benchmarks needed the
tolerance in user code every time.

A field can live on part of the mesh: `regions = Dict(w => Region(Ω, :name))` restricts `w`
to the cell set `name`, through a Ferrite `SubDofHandler`. Outside its region the field has no
dofs, its equation is not assembled, it enters the other equations as zero, and the solution
reads `NaN` there (`sol[w(t, x, y)]` at vertices outside the region, `sol(t, x, y; dv)` at
points outside it). Its boundary conditions must be on pieces whose facets belong to the
region's cells; on the rest of the region's boundary, such as an interface with the other
cells, it has the natural zero-flux condition.

## Streamline-upwind Petrov–Galerkin

`test_space = SUPG(τ)` ([`SUPG`](@ref)) tests every equation with `v + τ b·∇v` on every cell, where `b` is the
advection velocity read from the equation (the coefficient of `∇u`), or `SUPG(τ; b = [b1, b2])`.
`τ` is a number, an expression, or a function `τ(hK, b)` of the cell diameter `hK` and `b`:

```julia
eq = Dt(u(t, x)) ~ ε * Dx(Dx(u(t, x))) - Dx(u(t, x))
τ(hK, b) = (Pe = abs(b[1]) * hK / (2ε); hK / (2abs(b[1])) * (coth(Pe) - 1 / Pe))
disc = FEMDiscretization(Ω, t; test_space = SUPG(τ))
```

The stabilisation is consistent: the strong residual, time derivative and source included, is
tested with `τ b·∇v`, which needs the shape-function Hessians, so the `CellValues` are built
with them. Limits: linear terms only, `τ` and `b` must be numbers or expressions in the
coordinates, time and `hK` (not parameters of the `PDESystem`), and boundary terms stay
Galerkin.

## Steady problems

A `PDESystem` with no time variable, discretized by a `FEMDiscretization` built without one,
lowers to a `NonlinearProblem` `K u + N(u) - F = 0`, where `N` holds the terms whose
coefficients depend on the state; without them Newton solves it in one step. Dirichlet rows
are the algebraic equations `u_d - g_d = 0`, the Jacobian `K + ∂N/∂u` is sparse, and `solve` returns a
`PDENoTimeSolution` with the same interface as a transient solution, without the time argument:
`sol[u(x, y)]`, `sol(x, y; dv)`, traces `sol[u(Γ)]`, derivatives such as `sol[Dx(u(x, y))]`,
`normal_flux` and `write_vtk` (one `.vtu` file). Traces are written `u(Γ)`, since the dependent
variables have no time argument.

```julia
using FiniteElementMethod, ModelingToolkit, NonlinearSolve
using Ferrite: Triangle
@variables x y u(..)
Dx, Dy = Differential(x), Differential(y)
Ω = MeshDomain(FiniteElementMethod.rectangle_grid(Triangle, (16, 16)))
eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ -1
@named pdesys = PDESystem([eq], [u(Γ) ~ 0 for Γ in boundarypieces(Ω)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
prob = discretize(pdesys, FEMDiscretization(Ω))
sol = solve(prob, NewtonRaphson())
sol(0.5, 0.5; dv = u(x, y))
```

Newton starts from zero, with the Dirichlet data on any Dirichlet dofs the unknown vector
holds; another initial guess is a vector laid out like `prob.u0`, passed as `remake(prob; u0)`. With the
default `dirichlet = :algebraic` that vector holds every dof and field `name` occupies
`FiniteElementMethod.field_range(prob.problem_type.space, name)`. Equations with no entry in
`pairing` are tested with the remaining dependent variables in declared order.
`algebraic_abstol` belongs to the time-dependent (DAE) path: a steady discretization accepts
`:auto` and `nothing` and does nothing with them, and raises for a number.

## Dirichlet conditions

`dirichlet = :algebraic` (the default) keeps every dof as an unknown and makes each prescribed
row the algebraic equation `u_d = g_d(t)`, so it accepts Dirichlet data of any time dependence.
`dirichlet = :eliminate` removes the prescribed dofs instead: the `DAEProblem` has only the free
dofs as unknowns, with `M[fr, fr]`, `K[fr, fr]` and the load `F[fr] - K[fr, cd] g`; nonlinear
terms are evaluated at the full state with `u_cd = g` and keep their rows `fr`. It needs
Dirichlet data that are constant in time, though they may vary in space; time-dependent data
raise an `ArgumentError` naming the boundary piece. The solution is the same either way:
`sol[u]`, `sol(t, x, y)`, traces `sol[u(t, Γ)]`, `normal_flux` and `write_vtk` all return the
full field with `g` at the prescribed nodes. Only `sol.original_sol` and the array unknowns
hold the free dofs alone.

```julia
disc = FEMDiscretization(Ω, t; dirichlet = :eliminate)
```
