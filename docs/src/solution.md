# Solutions

`solve` on a FiniteElementMethod problem returns a `SciMLBase.PDETimeSeriesSolution` (a
`SciMLBase.PDENoTimeSolution` for a steady problem) whose metadata is a [`FEMMetadata`](@ref).
It is indexed and called in the variables of the `PDESystem`, like a MethodOfLines solution,
with the differences below.

## Coordinate binding

A mesh domain carries no coordinate names. The names and their order come only from the
`(x, y) ∈ Ω` pairing of the `PDESystem`: tuple position `i` is coordinate component `i` of the
mesh. The same pairing orders `sol[x]`, `sol[y]` and the arguments of `sol(t, x, y)`.

## Shapes that differ from MethodOfLines

  - `sol[x]` is the vector of mesh vertex coordinates, not an axis grid, so tensor-grid
    plotting calls such as `heatmap(sol[x], sol[y], …)` fail with a shape error. Resample with
    `sol(t, xs, ys; dv = u(t, x, y))` instead.
  - `sol[u(t, x, y)]` is `nt × nvertices`, and column `i` pairs with `(sol[x][i], sol[y][i])`.
    For a P2 field the edge dofs are not included; `sol[u(t, x, y), k, :]` is the row at `t_k`.
  - The raw dof coefficients are in `sol.original_sol`, for example
    `sol.original_sol[u_arr]` for the array unknown of a field.
  - `sol(t, x, y; dv)` interpolates the dof vector in time and evaluates it at the points.
    Vector arguments give a tensor product and scalar arguments drop their axis;
    `sol(t, pts::Vector{<:Vec}; dv)` evaluates at scattered points, and `sol(t, x, y)` with no
    `dv` returns one value per dependent variable.
  - Points outside the mesh return `NaN` by default, which a non-convex mesh needs;
    `outside = :error` throws a `DomainError` instead. A time outside the saved span throws a
    `DomainError`.
  - The trace `sol[u(t, Γ)]` on a boundary piece `Γ` is `nt × length(piece_nodes(sol, Γ))`,
    and `piece_nodes(sol, Γ)` also indexes `sol[x]`. `sol(t, Γ; dv)` is the trace at time `t`.
  - Only first spatial derivatives are accepted as keys: `sol[Differential(x)(u(t, x, y))]`
    is the cellwise gradient L2-projected onto the field's space and read at the vertices, and
    `sol(t, x, y; dv = Differential(x)(u(t, x, y)))` is the pointwise gradient.
  - [`normal_flux`](@ref) is a post-processed facet-mean flux `∇u·n`, not the imposed datum.

## API

```@docs
FieldInfo
FEMMetadata
piece_nodes
normal_flux
write_vtk
```
