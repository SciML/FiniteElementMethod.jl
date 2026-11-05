# FiniteElementMethod.jl

FiniteElementMethod.jl is a SciML discretizer that takes a
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) `PDESystem` in strong form
on a mesh, derives the Galerkin weak form, assembles it with
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) and lowers it to a `DAEProblem`.

!!! warning
    This package is under construction. Problems with Galerkin or SUPG testing are supported,
    including nonlinear terms whose state dependence is in the coefficients, sources and
    natural conditions (`∇·((1 + u²)∇u)`, `u ∂ₓu`, `u²`, `∂ₙu = −u⁴`); periodic
    conditions and time-dependent mass or stiffness coefficients raise an error. [Boundary conditions and limitations](@ref) lists every restriction.

[Getting started](@ref) solves a heat equation on a Gmsh mesh. A MethodOfLines user should
then read [Differences from MethodOfLines](@ref): coordinates are bound differently, boundary
conditions are written on named boundary pieces, and solutions have different shapes.
[Supported equation classes](@ref) states what is supported and on what evidence.

```@docs
FiniteElementMethod
```
