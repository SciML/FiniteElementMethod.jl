# FiniteElementMethod.jl

FiniteElementMethod.jl is a SciML discretizer that takes a
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) `PDESystem` in strong form
on a mesh, derives the Galerkin weak form, assembles it with
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) and lowers it to a `DAEProblem`.

!!! warning
    This package is under construction. Problems with Galerkin testing are supported,
    including nonlinear terms whose state dependence is in the coefficients, sources and
    natural conditions (`∇·((1 + u²)∇u)`, `u ∂ₓu`, `u²`, `∂ₙu = −u⁴`); periodic
    conditions, time-dependent mass or stiffness coefficients and other test spaces raise
    an error.

```@docs
FiniteElementMethod
```
