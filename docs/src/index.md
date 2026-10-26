# FiniteElementMethod.jl

FiniteElementMethod.jl is a SciML discretizer that takes a
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) `PDESystem` in strong form
on a mesh, derives the Galerkin weak form, assembles it with
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) and lowers it to a `DAEProblem`.

!!! warning
    This package is under construction. Linear scalar problems with Galerkin testing are
    supported; nonlinear terms, periodic conditions and other test spaces raise an error.

```@docs
FiniteElementMethod
```
