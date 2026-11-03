# Boundary conditions and limitations

This page answers the questions that the
[SciMLBase PDE interface](https://docs.sciml.ai/SciMLBase/dev/interfaces/PDE/) asks every
discretizer to document: the supported geometries and coordinates; the supported initial,
Dirichlet, Neumann, Robin, periodic and interface conditions; how conditions are validated and
imposed; and the restrictions of the discretization.

## Summary

The table and list below are the output of `PDEBase.supported_conditions(disc)` for a
`FEMDiscretization`, rendered with `show(io, MIME"text/markdown"(), …)`. They are copied here so
that this page builds without the PDESystem front end; a test checks that the copy equals the
rendering.

| Condition | Supported |
|:--- |:--- |
| Geometries | mesh |
| Dirichlet | yes |
| Neumann | yes |
| Robin | yes |
| Periodic | no |
| Interface | no |
| Time-dependent | yes |

Restrictions of this release:

  - Geometries are `Ferrite.Grid` meshes (1D, 2D and 3D cell shapes) wrapped as `MeshDomain(grid)`, with boundary pieces named by the grid's facet sets. Only 1D and 2D are evidenced; meshes with mixed cell types are rejected until M4.
  - Coordinates are Cartesian and named by the `(x, y) ∈ Ω` pairing. Time is the first or last argument of every dependent variable, never between spatial arguments.
  - Dirichlet data may depend on space, time and parameters but not on another field. Neumann and Robin conditions are written with `NormalDerivative(Γ)`, with any conormal coefficient written explicitly. Neumann and Robin data may depend on time; the Robin coefficient multiplying `u(t, Γ)` may depend only on space and parameters until M3.
  - Periodic conditions `u(t, Γa) ~ u(t, Γb)` are rejected until M4. Interface conditions between regions have no syntax; every field lives on the whole mesh.
  - Boundary pieces must be disjoint. A piece with no condition gets the natural zero-flux condition, with a warning when the operator has a flux.
  - Where two Dirichlet pieces of one field meet, the condition listed last in the `PDESystem` wins.
  - Every field with a time derivative needs an initial condition, linear in that one field and finite at every node.
  - `dirichlet = :eliminate`, fields with different interpolations and test spaces other than `Galerkin()` are rejected until M4. Nonlinear operators, time-dependent mass or stiffness coefficients and `codegen = :mtk` are rejected until M3.

"Time-dependent" means boundary data that varies with time, for every condition kind.

## Geometries and coordinates

The spatial domain is a `MeshDomain(grid)` of a `Ferrite.Grid`, for example one read from Gmsh
with `FerriteGmsh.togrid` or made with `Ferrite.generate_grid`. Boundary pieces are the grid's
facet sets: `boundarypiece(Ω, :inlet)` is the facet set `"inlet"`. Pieces must be disjoint; a
facet in two pieces that both carry conditions is an error naming the first shared facet.
Line, triangle and quadrilateral meshes are tested; tetrahedral and hexahedral meshes construct
but are not evidenced (see [Supported equation classes](@ref)). Every cell of the mesh must
have the same shape.

Coordinates are Cartesian. Their names and order come from the `(x, y) ∈ Ω` pairing (see
[Coordinate binding](@ref)). A curvilinear operator, such as a radial Laplacian, is written
out in Cartesian-coordinate form in the equation.

## Initial conditions

An initial condition is written as for MethodOfLines, `u(0, x, y) ~ f(x, y)`, at the start of
the time interval. It must give the value of one dependent variable, be linear in it, and be
finite at every node of the mesh; a non-finite value is an error naming the node. Every field
with a time derivative needs one. A field without a time derivative, such as `v` in
`0 ~ Dx(Dx(v(t, x, y))) + …`, is algebraic: it needs no initial condition, and its equation
must be paired with it through `FEMDiscretization(…; pairing = Dict(eq => v))` when the pairing
cannot be inferred.

## Dirichlet conditions

`u(t, Γ) ~ g` with `g` constant or a function of the coordinates, time and parameters:

```julia
u(t, Γ_in) ~ 1
u(t, Γ_in) ~ sin(t)
u(t, Γ_in) ~ (1 + x + 2y) * (1 + t)
```

Each Dirichlet node becomes an algebraic row `u_d = g_d(t)` of the `DAEProblem`, so the
condition is imposed exactly at the nodes, not weakly. Time-dependent data is evaluated at
the current time of every residual evaluation. The data may not depend on another field: `u(t, Γ) + v(t, Γ) ~ 1` is an
error.

With default solver settings these rows hold to about the solver tolerance; see "Exact
Dirichlet rows" in [Getting started](@ref) for the setting that holds them to round-off.

**Corners.** Where two pieces with Dirichlet conditions on the same field meet, the shared
nodes take the condition listed last in the `PDESystem`'s boundary conditions, as in Ferrite's
`ConstraintHandler`. Listing a Dirichlet condition on the same piece and field twice is an
error.

## Neumann conditions

`Dn(u(t, Γ)) ~ h` with `Dn = NormalDerivative(Γ)`, the derivative along the outward unit
normal of `Γ`. The data `h` may depend on the coordinates, time and parameters. The condition
enters the weak form as a facet integral, so it holds weakly, not node by node.

When the equation's flux has a coefficient, write it: for `Dt(u) ~ Dx(k * Dx(u)) + …` the
natural condition is on `k * Dn(u(t, Γ))`. A condition written against the wrong coefficient is
rescaled when it is proportional to the conormal derivative; a condition that is not, such as
`Dx(u(t, Γ)) ~ h` on a piece that is not perpendicular to the `x` axis, is an error asking for
the normal derivative.

A boundary piece with no condition gets the natural zero-flux condition. When the equations
have a flux, the discretization warns about each such piece.

## Robin conditions

`k * Dn(u(t, Γ)) + α * u(t, Γ) ~ β`, with the conormal coefficient written explicitly as for
Neumann conditions. The data `β` may depend on the coordinates, time and parameters. The
coefficient `α` may depend on the coordinates and parameters but not on time in this release:
a time-dependent `α` makes the boundary part of the stiffness operator time-dependent, which is
rejected with `time-dependent coefficients in the mass or stiffness operator land in M3`.
Like a Neumann condition, it enters the weak form as a facet integral.

## Periodic conditions

Not supported in this release. An equation between traces on two pieces,
`u(t, Γ_left) ~ u(t, Γ_right)`, is rejected with
`periodic conditions need milestone M4 (Ferrite PeriodicDirichlet)`.

## Interface conditions

Not supported. There is no syntax for a condition on an interior interface between two
regions, and every field is defined on the whole mesh. Discontinuous coefficients across an
interface are not evidenced (see [Supported equation classes](@ref)).

## Validation

Every condition is checked when the system is discretized, by `discretize(pdesys, disc)`, before a solver runs. An unsupported condition, operator or option
is an error that names the problem; nothing falls back to `mtkcompile`.

## Restrictions

| Input | Result in this release |
|:--- |:--- |
| A nonlinear operator, such as `Dx((1 + u^2) * Dx(u))` | error; lands in M3 |
| A time-dependent coefficient in the mass or stiffness operator, such as `Dx((1 + t) * Dx(u))` | error; lands in M3 |
| A time-dependent Robin coefficient, such as `Dn(u(t, Γ)) + (1 + t) * u(t, Γ) ~ 0` | error; lands in M3 |
| `FEMDiscretization(…; codegen = :mtk)` | error; lands in M3 |
| `FEMDiscretization(…; dirichlet = :eliminate)` | error; lands in M4 |
| A test space other than `Galerkin()` | error; lands in M4 |
| A mesh with mixed cell types | error; lands in M4 |
| Fields with different interpolations, such as P2 velocity and P1 pressure | error; lands in M4 |
| A periodic condition `u(t, Γa) ~ u(t, Γb)` | error; lands in M4 |
| A time argument between spatial arguments, such as `u(x, t, y)` | error |
| Two Dirichlet pieces of one field sharing nodes | the condition listed last wins |
| A boundary piece with no condition | natural zero-flux condition, with a warning when the equations have a flux |
