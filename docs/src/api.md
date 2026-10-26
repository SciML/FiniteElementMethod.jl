# API

## Discretization

`FEMDiscretization` is exported, as are the PDEBase names a model needs (`MeshDomain`,
`boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's `Lagrange` and reference
shapes. The rest of this page documents the internal building blocks of the Ferrite backend.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
```

## Meshes

```@docs
FiniteElementMethod.rectangle_grid
FiniteElementMethod.check_facetsets
```

## Discrete space

```@docs
FiniteElementMethod.FEMSpace
FiniteElementMethod.field_range
FiniteElementMethod.prescribed_dofs
FiniteElementMethod.dirichlet_values!
```

## DAE lowering

```@docs
FiniteElementMethod.FEMOperators
FiniteElementMethod.FEMOp
FiniteElementMethod.FEMLoad
FiniteElementMethod.load!
FiniteElementMethod.dirichlet_rates
FiniteElementMethod.fem_system
FiniteElementMethod.numeric_dae_problem
FiniteElementMethod.consistent_du0
FiniteElementMethod.check_finite_initial_condition
```

## Solution interface

```@docs
FieldInfo
FEMMetadata
FiniteElementMethod.FEMInterpolant
piece_nodes
normal_flux
write_vtk
```

## Weak form, kernels and assembly

```@docs
FiniteElementMethod.TestSlots
FiniteElementMethod.VolumeTerm
FiniteElementMethod.FacetTerm
FiniteElementMethod.WeakForm
FiniteElementMethod.KernelSpec
FiniteElementMethod.classify
FiniteElementMethod.state_slots
FiniteElementMethod.weak_form
FiniteElementMethod.emit_kernels
FiniteElementMethod.emit_dirichlet
FiniteElementMethod.AssemblyContext
FiniteElementMethod.assemble_linear
FiniteElementMethod.assemble_nonlinear
FiniteElementMethod._split
```
