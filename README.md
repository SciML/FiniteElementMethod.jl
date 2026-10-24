# FiniteElementMethod.jl

FiniteElementMethod.jl is a SciML discretizer that takes a
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) `PDESystem` in strong form on a
mesh, derives the Galerkin weak form, assembles it with
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) and lowers it to a `DAEProblem`.

## Status

Under construction. This repository currently contains only the package skeleton; none of the
above is implemented yet.
