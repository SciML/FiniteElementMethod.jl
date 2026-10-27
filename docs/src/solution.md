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

## Observed expressions

`sol[expr]` evaluates an expression in the dependent variables, their first spatial
derivatives, the independent variables and the parameters of the `PDESystem`, for example
`sol[u(t, x, y)^2 + k * v(t, x, y)]`, `sol[2u(t, x, y)]` or `sol[x^2 + y]`. It has the shape of
`sol[u(t, x, y)]`, `nt × nvertices` (a vector for a steady solution), even when the expression
names only coordinates. `sol(t, xs, ys; dv = expr)` and `sol(t, Γ; dv = expr)` evaluate it at
points or on a boundary piece, with the shapes of a field, and `NaN` outside the mesh.

  - The expression is compiled once with `build_function` and broadcast over the vertex values
    of its leaves; nothing is substituted per vertex. A first derivative such as
    `Differential(x)(u(t, x, y))` uses the same values as `sol[Differential(x)(u(t, x, y))]`.
  - A parameter takes the value given in the `PDESystem`, as `k => 2.0`; the front end has
    assembled the operators with that value, so `remake` with a new one does not change it.
  - An expression that cannot be evaluated throws an `ArgumentError` beginning
    "cannot evaluate the expression … on this solution:", which names a variable that is not
    of the system, a trace `u(t, Γ)` (index the trace on its own, or pass `Γ` to the call
    instead), a derivative that is not a first spatial derivative, or a dependent variable at
    fixed arguments such as `u(t, 0, y)`.

The docstrings are on the [API](@ref "API") page.
