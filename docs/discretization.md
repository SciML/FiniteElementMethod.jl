# Discretization

`FEMDiscretization` is exported, as are the PDEBase names a model needs (`MeshDomain`,
`boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's `Lagrange` and reference
shapes.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.SciMLBase.NonlinearProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.assemble_sources!
```

## Steady problems

A `PDESystem` with no time variable, discretized by a `FEMDiscretization` built without one,
lowers to a `NonlinearProblem` `K u - F = 0`, which Newton solves in one step. Dirichlet rows
are the algebraic equations `u_d - g_d = 0`, the Jacobian `K` is sparse, and `solve` returns a
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

Nonlinear terms are rejected on the steady path, as on the time-dependent one. Newton starts
from zero, with the Dirichlet data on any Dirichlet dofs the unknown vector holds; another
initial guess is a vector laid out like `prob.u0`, passed as `remake(prob; u0)`. With the
default `dirichlet = :algebraic` that vector holds every dof and field `name` occupies
`FiniteElementMethod.field_range(prob.problem_type.space, name)`. Equations with no entry in
`pairing` are tested with the remaining dependent variables in declared order.
