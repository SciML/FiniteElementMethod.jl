# Discretization

`FEMDiscretization` is exported, as are the PDEBase names a model needs (`MeshDomain`,
`boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's `Lagrange` and reference
shapes. [Getting started](@ref) shows them in use.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
FiniteElementMethod.assemble_sources!
```

`discretize(pdesys, disc)` is `DAEProblem(pdesys, disc)`. `symbolic_discretize(pdesys, disc)`
returns `(sys, tspan)`, where `sys` is the array-residual `System` attached to that problem
(see [`FiniteElementMethod.fem_system`](@ref)): one block equation per field over the
assembled operators, not scalar equations per node as MethodOfLines returns.

## Supported conditions

`PDEBase.supported_conditions(disc)` returns the conditions this discretization supports. For a
`FEMDiscretization` it renders as follows; [Boundary conditions and limitations](@ref)
explains each entry.

```@eval
using FiniteElementMethod, Markdown
using FiniteElementMethod: PDEBase, Ferrite, rectangle_grid
if isdefined(PDEBase, :supported_conditions)
    disc = FEMDiscretization(MeshDomain(rectangle_grid(Ferrite.Triangle, (1, 1))))
    Markdown.parse(repr(MIME"text/markdown"(), PDEBase.supported_conditions(disc)))
else
    Markdown.parse("The installed PDEBase predates `supported_conditions`.")
end
```
