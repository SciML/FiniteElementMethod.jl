# Extension contract

```@meta
CurrentModule = FiniteElementMethod
```

SciMLBase's
[Core Extension Contract](https://docs.sciml.ai/SciMLBase/stable/interfaces/PDE/#Core-Extension-Contract)
asks a discretizer for a `discretize` that returns a solver-ready problem, forwards the
problem-construction keywords, rejects what it cannot discretize with a clean error and keeps
the metadata to rebuild the PDE solution; `symbolic_discretize` is optional. This page states
how `FEMDiscretization` meets each clause.

## `discretize` returns a solver-ready problem

[`discretize(pdesys, disc; kwargs...)`](@ref SciMLBase.discretize(::ModelingToolkit.PDESystem, ::FEMDiscretization))
accepts a `ModelingToolkit.PDESystem` on a `MeshDomain` and returns

  - a `DAEProblem`
    when `disc` has a time variable, `FEMDiscretization(Ω, t)`, for a DAE solver such as `DFBDF`;
  - a [`NonlinearProblem`](@ref SciMLBase.NonlinearProblem(::ModelingToolkit.PDESystem, ::FEMDiscretization))
    when it has none, `FEMDiscretization(Ω)`, for a nonlinear solver such as `NewtonRaphson`.

Both carry the sparse analytic Jacobian and its prototype, so `solve(prob, alg)` needs nothing
else. `DAEProblem(pdesys, disc)` and `NonlinearProblem(pdesys, disc)` build the same problems
directly.

## Keywords

Keywords given to `FEMDiscretization` are stored in `disc.kwargs` and passed to the problem
first, then the keywords of the `discretize` call, so a call keyword wins over the same
keyword in `disc`. Every keyword not listed below is forwarded to the problem, as other SciML
problem constructors do: `saveat`, `tstops`, `abstol`, `reltol`, `callback` and any other solve
option are stored in `prob.kwargs` and reach `solve`.

```julia
disc = FEMDiscretization(Ω, t; abstol = 1.0e-8)
prob = discretize(pdesys, disc; saveat = 0.1, abstol = 1.0e-10)  # abstol 1e-10 wins
```

Ten keywords are reserved and raise an `ArgumentError`, from `FEMDiscretization` and from the
problem constructors alike. Six are set by the discretization itself and cannot be set at all:
`iv`, `unknowns`, `du0`, `coordinate`, `name` and `parameters`. Four are defined by the
`PDESystem` or computed by the discretization, and are set there instead:

| Keyword             | Comes from                       | To change it                                                        |
|:------------------- |:-------------------------------- |:------------------------------------------------------------------- |
| `tspan`             | the time domain of the PDESystem | change that domain, `t ∈ Interval(t0, t1)`                          |
| `u0`                | its initial conditions (steady: zero with the Dirichlet data) | change them, or `remake(prob; u0 = ...)` on the returned problem |
| `p`                 | its parameter defaults           | change them and discretize again; the operators are assembled at those values, so `solve` rejects `remake(prob; p = ...)` |
| `differential_vars` | the weak form: the fields with a time derivative | change the equations                              |

The error message says the same, for example
``keyword `tspan` cannot be passed: the time span is the time domain of the PDESystem; change that domain to change it``.

## Validation

What the discretization cannot handle raises an `ArgumentError` that says what is unsupported.
The checks include:

  - the call itself: further positional arguments
    (`discretize(pdesys, disc; kwargs...) takes no further positional arguments; got 1`) and a
    first argument that is not a `PDESystem`
    (`FEMDiscretization discretizes a ModelingToolkit.PDESystem; got ...`);
  - domains: no `(x, y) ∈ MeshDomain(mesh)` pairing, a `MeshDomain` other than the
    discretization's, a time variable not paired with an interval, a time interval with a steady
    `disc` or the reverse, mixed cell types, interpolations on another reference shape;
  - equations: an equation count that differs from the number of dependent variables, unknown
    `pairing`, `spaces` or `regions` keys, time-dependent mass or stiffness coefficients,
    parameters without a numeric value, SUPG with parameters in `τ` or `b`;
  - boundary conditions: periodic conditions, conditions on facet sets the mesh lacks, a
    region's field with conditions outside the region, time-dependent Dirichlet data with
    `dirichlet = :eliminate`.

These are checked before the operators are assembled. Initial conditions are checked against
the node coordinates during assembly: one that does not name a single dependent variable, is not
linear in it or is not finite at a node raises, as does a field with a time derivative and no
initial condition. The [Discretization](@ref) page describes what is supported.

## Metadata and the solution

The problem's `problem_type` is a [`FEMMetadata`](@ref), a subtype of
`SciMLBase.AbstractDiscretizationMetadata{hasTime}` that holds the dof handler, the fields,
the boundary pieces and the `PDESystem`. After the solve, `SciMLBase.wrap_sol` passes the
solver's solution and that metadata to the constructors FiniteElementMethod defines,
`PDETimeSeriesSolution(sol, md::FEMMetadata)` (with time) and
`PDENoTimeSolution(sol, md::FEMMetadata)` (steady), so `solve` returns a solution indexed and
called in the `PDESystem`'s variables, `sol[u(t, x, y)]` and `sol(t, x, y; dv)`; see
[Solution shapes that differ from MethodOfLines](@ref). The solver's own solution stays in `sol.original_sol`.

## `symbolic_discretize`

[`symbolic_discretize(pdesys, disc; kwargs...)`](@ref SciMLBase.symbolic_discretize(::ModelingToolkit.PDESystem, ::FEMDiscretization))
returns the tuple `(sys, tspan)`, the shape PDEBase and MethodOfLines return; this return type
is part of the public interface. It runs the code path of `discretize`, takes the same keywords
with the same errors, and returns `prob.f.sys`, the completed `ModelingToolkit.System` with the
`FEMMetadata` attached, and `prob.tspan`. A steady `disc` raises an `ArgumentError`, because its
`NonlinearProblem` has no symbolic system; use `discretize` for it.

## Subtyping

Nothing in the contract requires a discretizer to subtype `SciMLBase.AbstractDiscretization`,
and nothing in FiniteElementMethod dispatches on it. `FEMDiscretization{T}` subtypes
`PDEBase.AbstractMeshDiscretization` (under `PDEBase.AbstractEquationSystemDiscretization` and
`SciMLBase.AbstractDiscretization`) so that PDEBase's mesh-domain machinery, such as
`PDEBase.interface_errors` and the boundary parsing, applies to it.
