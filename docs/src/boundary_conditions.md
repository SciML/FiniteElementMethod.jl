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
| Periodic | yes |
| Interface | no |
| Time-dependent | yes |

Restrictions of this release:

  - Geometries are `Ferrite.Grid` meshes (1D, 2D and 3D cell shapes) wrapped as `MeshDomain(grid)`, with boundary pieces named by the grid's facet sets. 1D, 2D and 3D meshes are tested; meshes with mixed cell types are rejected.
  - Coordinates are Cartesian and named by the `(x, y) ∈ Ω` pairing. Time is the first or last argument of every dependent variable, never between spatial arguments.
  - Dirichlet data may depend on space, time and parameters but not on another field. Neumann and Robin conditions are written with `NormalDerivative(Γ)`, with any conormal coefficient written explicitly. Neumann and Robin data may depend on time; the Robin coefficient multiplying `u(t, Γ)` may depend only on space and parameters.
  - Periodic conditions `u(t, Γa) ~ u(t, Γb)` pair two pieces that are translates of each other; they need `dirichlet = :algebraic` and are not returned by `operators`. Interface conditions between regions have no syntax; every field lives on the whole mesh.
  - Boundary pieces must be disjoint and lie on the boundary. Boundary facets with no condition for a field get the natural zero-flux condition, with a warning when that field's equation has a flux; `natural` declares them and `boundary_check` makes the warning an error or turns it off.
  - Where two Dirichlet pieces of one field meet, the condition listed last in the `PDESystem` wins.
  - Every field with a time derivative needs an initial condition, linear in that one field and finite at every node.
  - Time-dependent mass or stiffness coefficients and `codegen = :mtk` are rejected.

"Time-dependent" means boundary data that varies with time, for every condition kind.

## Geometries and coordinates

The spatial domain is a `MeshDomain(grid)` of a `Ferrite.Grid`, for example one read from Gmsh
with `FerriteGmsh.togrid`, made with `Ferrite.generate_grid` or converted from a Meshes.jl mesh
with [`ferrite_grid`](@ref). Boundary pieces are the grid's
facet sets: `boundarypiece(Ω, :inlet)` is the facet set `"inlet"`. Pieces must be disjoint; a
facet in two pieces that both carry conditions is an error naming the first shared facet, and a
piece containing an interior facet (one shared by two cells) is an error, since a piece lies on
the boundary.
Line, triangle, quadrilateral, tetrahedral and hexahedral meshes are tested (3D heat
convergence in `test/Integration/three_d.jl`; see [Supported equation classes](@ref)). Every cell of the mesh must
have the same shape.

**Pieces from submeshes.** A boundary may also be given as a lower-dimensional mesh: the boundary
mesh that Gmsh gives for a physical group, or any `Ferrite.Grid` whose cells are facets of the
parent (line cells on a 2D mesh, surface cells on a 3D mesh). [`add_boundary_pieces!`](@ref)
matches each submesh cell to a facet of the parent by the coordinates of its nodes, within a
tolerance `atol` (the two meshes number their nodes independently), and adds the facets as the
facet set `name`:

```julia
grid = rectangle_grid(Triangle, (4, 4))
add_boundary_pieces!(grid, :inlet => inlet_mesh, :wall => wall_mesh)
Ω = MeshDomain(grid)
Γ_in = boundarypiece(Ω, :inlet)
```

A submesh cell that matches no facet of the parent, or only an interior facet, is an error that
names the cell and where it is. The grid's own facet sets work as before.

With Meshes.jl loaded, a submesh may also be a Meshes.jl mesh, such as the rings of a polygon:

```julia
using Meshes: Meshes
outer = Meshes.Ring((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
hole = Meshes.Ring((1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0))
grid = ferrite_grid(Meshes.discretize(Meshes.Pgon(outer, hole)))
add_boundary_pieces!(grid, :outer => Meshes.discretize(outer), :hole => Meshes.discretize(hole))
Ω = MeshDomain(grid)
```

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

Boundary facets with no condition get the natural zero-flux condition; see
[Boundary completeness](@ref).

## Robin conditions

`k * Dn(u(t, Γ)) + α * u(t, Γ) ~ β`, with the conormal coefficient written explicitly as for
Neumann conditions. The data `β` may depend on the coordinates, time and parameters. The
coefficient `α` may depend on the coordinates and parameters but not on time in this release:
a time-dependent `α` makes the boundary part of the stiffness operator time-dependent, which is
rejected with `time-dependent coefficients in the mass or stiffness operator are not supported`.
Like a Neumann condition, it enters the weak form as a facet integral.

## Periodic conditions

An equation between the traces of one field on two pieces, `u(t, Γ_left) ~ u(t, Γ_right)` (or
`u(Γ_left) ~ u(Γ_right)` for a steady problem), makes the field take equal values on the two
pieces. The pieces must be translates of each other: the translation is the difference of their
centroids, and every node of one piece (every dof of the field, for a higher-order space) must
have a node of the other at the translated place. Otherwise `discretize` throws an `ArgumentError`
that names both pieces and the mismatch. The pairs are per field, so one field may be periodic
where another carries Dirichlet pieces, and they work with any element space, including the
Taylor–Hood pair of a Navier–Stokes system, in the time-dependent and steady paths and with
`SUPG` test spaces.

For a fully periodic box write one pair per direction. A corner node belongs to several pairs;
the nodes of all pairs of a field are collected into classes of equal nodes, and each class keeps
one master. Where a periodic piece meets a Dirichlet piece, the class contains a prescribed node
and that node is the master, so the Dirichlet value holds at the shared corner. If a class holds more than one
prescribed node (a Dirichlet piece touching both ends of a periodic pair), their data must agree, at the initial time
and at four further times of the time interval; otherwise `discretize` throws an `ArgumentError` naming the periodic pieces,
the nodes and the two values.

```julia
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

Ω = MeshDomain(generate_grid(Triangle, (16, 16), Vec(0.0, 0.0), Vec(1.0, 1.0)))
left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
@named pdesys = PDESystem(
    [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
    [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ sin(2π * (x + y))],
    [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], [])
prob = discretize(pdesys, FEMDiscretization(Ω, t))
```

The solution is `exp(-8π² t) sin(2π (x + y))` and converges at the optimal rate;
`test/Interface/periodic.jl` runs this case, the steady and Navier–Stokes ones.

Periodic pairs work in every problem class of this package, including nonlinear terms (the
nonlinear residual and Jacobian get the same row sums and ties inside Newton's method), SUPG test
spaces, steady and transient problems and coupled fields with different spaces:

| Problem class with a periodic pair | Status | Covered by |
|:--- |:--- |:--- |
| Scalar linear, transient, 2D, P1 and P2, triangles and quadrilaterals, fully periodic | supported | `test/Interface/periodic.jl`, heat on a periodic box |
| Periodic in one direction with Dirichlet pieces in the other | supported | `test/Interface/periodic.jl`, periodic in x |
| Scalar linear on a line, transient, Galerkin and SUPG | supported | `test/Interface/periodic.jl`, periodic line |
| Scalar linear on a hexahedral mesh, steady, fully periodic | supported | `test/Interface/periodic.jl`, periodic cube |
| Steady (`NonlinearProblem`), linear terms, Galerkin and SUPG | supported | `test/Interface/periodic.jl`, steady advection–diffusion |
| Steady, nonlinear terms | supported | `test/Interface/periodic.jl`, steady nonlinear diffusion |
| Transient (`DAEProblem`), nonlinear terms and a time-dependent source | supported | `test/Interface/periodic.jl`, viscous Burgers |
| Coupled fields, different spaces per field, per-field pairs | supported | `test/Interface/periodic.jl`, coupled P2/P1 |
| Taylor–Hood Navier–Stokes, steady and transient (index-2 DAE) | supported | `test/Interface/periodic_navier_stokes.jl` |
| Boundary integrals (Neumann, Robin) on the other, non-periodic pieces | supported | assembled as before; the periodic pieces take no boundary integral |
| Pieces that are not translates of each other | error `periodic pieces Γa and Γb of u do not match under a translation: …` | `test/Interface/periodic.jl`, rejected pairs |
| A periodic piece that also carries another condition for that field | error `… has a periodic condition on Γ and a Dirichlet condition on the same piece` | same |
| `dirichlet = :eliminate`, or `operators` | error `… does not support periodic conditions …` | same |
| A space that is not a scalar Lagrange space (edge elements need orientation signs across the map) | error `… is not a nodal Lagrange space; periodic conditions are supported for scalar Lagrange spaces only …` | same, edge-element case |

The operators keep every dof as an unknown. Each node of a class other than its master is tied
to it by an algebraic row `u_node − u_master = 0`, and the Galerkin equation of the tied node is
added to its master's, which is the condensed system `PᵀKP` of the identification. No boundary
integral is assembled on a periodic piece, since the two contributions cancel for a periodic
solution. The initial value of a tied node is its master's. A piece of a periodic pair cannot
also carry a Dirichlet, Neumann or Robin condition for that field. `dirichlet = :eliminate` and
[`operators`](@ref FiniteElementMethod.operators) do not support periodic conditions.

## Interface conditions

Not supported. There is no syntax for a condition on an interior interface between two
regions, and every field is defined on the whole mesh. Discontinuous coefficients across an
interface are not evidenced (see [Supported equation classes](@ref)).

## Boundary completeness

A boundary facet is covered for a field when it lies in a piece with a Dirichlet condition on
that field, or a Neumann or Robin condition in that field's equation. Where part of the boundary
is not covered for a field whose equation has a flux, the discretization treats it as the natural
zero-flux condition. That is often intended (an insulated wall, a symmetry plane, the outflow
`ν ∂ₙu = 0` of a channel flow), so it is not an error by default, but it is also what a forgotten
piece looks like. The check is therefore configurable on [`FEMDiscretization`](@ref):

  - By default (`boundary_check = :warn`) `discretize` logs one warning per call, listing for
    each such field the uncovered pieces and their number of facets (and the facets that belong
    to no named piece).
  - `natural = [Γ_out, :symmetry]` declares pieces as deliberately natural for every field; the
    warning is then silent for them. An explicit zero Neumann condition `Dn(u(t, Γ)) ~ 0` also
    covers its piece.
  - `boundary_check = :error` throws the same message as an `ArgumentError`, for strict
    completeness; `boundary_check = :off` skips the check.
  - A field whose equation has no flux, such as a pressure, is not checked.

These are errors whatever `boundary_check` is: a facet in two pieces that both carry conditions
(which includes two Dirichlet values on one facet through overlapping pieces, and the same piece
and field listed twice), and a piece that is not on the boundary.

Detecting a singular pure-Neumann problem is not part of this check.

## Validation

Every condition is checked when the system is discretized, by `discretize(pdesys, disc)`, before a solver runs. An unsupported condition, operator or option
is an error that names the problem; nothing falls back to `mtkcompile`.

## Restrictions

| Input | Result in this release |
|:--- |:--- |
| A time-dependent coefficient in the mass or stiffness operator, such as `Dx((1 + t) * Dx(u))` | error |
| A time-dependent Robin coefficient, such as `Dn(u(t, Γ)) + (1 + t) * u(t, Γ) ~ 0` | error |
| `FEMDiscretization(…; codegen = :mtk)` | error |
| A mesh with mixed cell types | error |
| A periodic condition `u(t, Γa) ~ u(t, Γb)` between pieces that are not translates of each other | error naming both pieces |
| A periodic piece that also carries a Dirichlet, Neumann or Robin condition for that field | error |
| A periodic condition with `dirichlet = :eliminate`, or in `operators` | error |
| A time argument between spatial arguments, such as `u(x, t, y)` | error |
| Two Dirichlet pieces of one field sharing nodes | the condition listed last wins |
| Boundary facets with no condition for a field | natural zero-flux condition, with one warning per `discretize` call when that field's equation has a flux; `natural` declares them, `boundary_check = :error` or `:off` |
| A piece that is not on the boundary | error |
| A submesh cell that matches no facet, or an interior facet, of the parent | error from `add_boundary_pieces!` |
