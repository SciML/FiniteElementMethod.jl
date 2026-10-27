# Differences from MethodOfLines

A `PDESystem` for FiniteElementMethod.jl is an ordinary ModelingToolkit `PDESystem`, and most of
it reads as it does for MethodOfLines. This page lists, in order, the places where a mesh makes
the package behave differently from what a MethodOfLines user expects.

## Coordinate binding

!!! warning "Read this first"
    A `MeshDomain` carries no coordinate names. The names and their order come only from the
    `(x, y) ∈ Ω` pairing in the domains of the `PDESystem`: tuple position `i` is coordinate
    component `i` of the mesh.

  - The tuple must have one entry per mesh coordinate, which is the mesh's spatial dimension;
    for a surface mesh embedded in 3D it is the embedding dimension 3. A mismatch is an error
    when the system is discretized:
    `domain MeshDomain of a … mesh has 3 coordinates but the pairing names 2`.
  - The order of the pairing, not the argument order of the dependent variables, decides
    which name is which mesh coordinate: with `(y, x) ∈ Ω` and `u(t, x, y)`, `y` is mesh
    coordinate 1 and `x` is mesh coordinate 2.
  - The same pairing orders `sol[x]` and `sol[y]` (the node coordinates) and the spatial
    arguments of `sol(t, x, y)`.
  - Trace and `NormalDerivative` conditions never mention coordinate names, so they need none.
    The spatial variables appear only in the equations, in initial conditions such as
    `u(0, x, y) ~ x * y`, and in data such as `u(t, Γ) ~ x * sin(t)`.

## Trace syntax

A boundary condition is an equation on the trace of a dependent variable on a named boundary
piece `Γ`, a facet set of the mesh obtained from `boundarypiece(Ω, :name)`.

  - **The piece replaces exactly the spatial arguments.** For `u(t, x, y)` the trace is
    `u(t, Γ)`: time stays explicit and in its declared position, as in MethodOfLines'
    `u(t, 0, y)`. Every other argument is unchanged.
  - **`u(Γ)` is an error on a time-dependent problem.** The message shows the form to write:
    `u(Γ_left): trace is missing the time argument; write u(t, Γ_left)`. `u(Γ)` is the trace
    only on a steady problem whose dependent variables have no time argument, such as `u(x, y)`.
    Steady problems are not discretized yet (see [Restrictions](@ref)), so in practice every
    trace is `u(t, Γ)`.
  - **`Dn_Γ(u(t, Γ))` is the trace of the normal derivative**, where `Dn_Γ = NormalDerivative(Γ)`:
    differentiate along the outward unit normal of `Γ`, then evaluate on `Γ`. This is the
    reading of MethodOfLines' `Dx(u(t, 0, y))`. Write a conormal coefficient explicitly, as in
    `k * Dn_Γ(u(t, Γ)) ~ h` for the flux `k ∇u`.
  - **The operator's piece must match the trace's piece.** `NormalDerivative(Γa)(u(t, Γb))`
    with `Γa ≠ Γb` is an error when the system is discretized.
  - **The MethodOfLines endpoint form is not accepted on a mesh.** A mesh domain has no
    coordinate intervals, so `u(t, 0, y) ~ 0` cannot name a face even when the mesh is a box;
    write a trace on a boundary piece instead.
  - **Pieces on a mesh broadcast; pieces on an analytic box need `Ref`.**
    `boundarypiece.(Ω, (:inlet, :wall))` works for a `MeshDomain`. On an analytic DomainSets
    box, such as a `Rectangle` in a system also given to MethodOfLines, write
    `boundarypiece.(Ref(rect), (:left, :right))`, because DomainSets domains do not broadcast
    as scalars. A rectangle's pieces are named `:bottom, :right, :top, :left`.
    `FEMDiscretization` itself needs a `MeshDomain`.

## Solution shapes that differ from MethodOfLines

`solve` returns a `SciMLBase.PDETimeSeriesSolution` whose metadata is a [`FEMMetadata`](@ref).
It is indexed and called in the variables of the `PDESystem`, like a MethodOfLines solution,
with these differences.

  - **`sol[x]` is the vector of mesh node coordinates, not an axis grid.** For a mesh of
    linear cells the nodes are the vertices. Tensor-grid plotting calls such as
    `heatmap(sol[x], sol[y], …)` fail with a shape error; resample with
    `sol(t, xs, ys; dv = u(t, x, y))` instead, which gives the tensor product of `xs` and `ys`.
  - **`sol[u(t, x, y)]` is `nt × nnodes`**, and column `i` pairs with `(sol[x][i], sol[y][i])`.
    It is the field evaluated at the mesh nodes, so for a P2 field the edge dofs are not
    included. `sol[u(t, x, y), k, :]` is the row at `t_k`.
  - **The raw dof coefficients are in `sol.original_sol`**, the solution of the `DAEProblem`;
    `sol.original_sol.u[k]` is the dof vector at `t_k`.
  - **Points outside the mesh return `NaN`** by default, which a non-convex mesh needs, since
    any resampling grid has points outside it. `sol(t, x, y; dv, outside = :error)` throws a
    `DomainError` instead.
  - **A time outside the saved span throws a `DomainError`**; the DAE interpolation is never
    extrapolated.
  - `sol(t, x, y; dv)` interpolates the dof vector in time and evaluates it at the points.
    Vector arguments give a tensor product and scalar arguments drop their axis;
    `sol(t, pts::Vector{<:Vec}; dv)` evaluates at scattered points, and `sol(t, x, y)` with no
    `dv` returns one value per dependent variable.
  - **Only first spatial derivatives are accepted as keys.** `sol[Differential(x)(u(t, x, y))]`
    is the cellwise gradient L2-projected onto the field's space and read at the nodes, and
    `sol(t, x, y; dv = Differential(x)(u(t, x, y)))` is the pointwise gradient. Higher
    derivatives are an error naming the accepted keys.
  - **Traces.** `sol[u(t, Γ)]` is `nt × length(piece_nodes(sol, Γ))`; [`piece_nodes`](@ref)
    also indexes `sol[x]`. `sol(t, Γ; dv)` is the trace at time `t`.
  - **`sol[Dn_Γ(u(t, Γ))]` is a post-processed flux, not the imposed datum.** It is the
    facet-mean outward normal derivative `∇u·n` on each facet of `Γ`, `nt × nfacets`, the same
    as [`normal_flux`](@ref). It converges to the boundary datum under refinement but does not
    equal it on a given mesh.
  - [`write_vtk`](@ref)`(sol, base)` writes one `.vtu` file per saved time and a ParaView
    collection `base.pvd`.

The docstrings are on the [API](@ref "API") page.
