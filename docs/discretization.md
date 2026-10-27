# Discretization

`FEMDiscretization` is exported, as are the PDEBase names a model needs (`MeshDomain`,
`boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's `Lagrange` and reference
shapes.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
FiniteElementMethod.assemble_sources!
```

## Dirichlet conditions

`dirichlet = :algebraic` (the default) keeps every dof as an unknown and makes each prescribed
row the algebraic equation `u_d = g_d(t)`, so it accepts Dirichlet data of any time dependence.
`dirichlet = :eliminate` removes the prescribed dofs instead: the `DAEProblem` has only the free
dofs as unknowns, with `M[fr, fr]`, `K[fr, fr]` and the load `F[fr] - K[fr, cd] g`. It needs
Dirichlet data that are constant in time, though they may vary in space; time-dependent data
raise an `ArgumentError` naming the boundary piece. The solution is the same either way:
`sol[u]`, `sol(t, x, y)`, traces `sol[u(t, Γ)]`, `normal_flux` and `write_vtk` all return the
full field with `g` at the prescribed nodes. Only `sol.original_sol` and the array unknowns
hold the free dofs alone.

```julia
disc = FEMDiscretization(Ω, t; dirichlet = :eliminate)
```
