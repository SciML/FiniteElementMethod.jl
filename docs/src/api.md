# API

These are the internal building blocks of the Ferrite backend. The PDESystem front end is
documented on the Discretization page.

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
FiniteElementMethod.FEMNonlinear
FiniteElementMethod.load!
FiniteElementMethod.dirichlet_rates
FiniteElementMethod.fem_system
FiniteElementMethod.numeric_dae_problem
FiniteElementMethod.consistent_du0
FiniteElementMethod.eliminate_dirichlet
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
FiniteElementMethod.find_differential
FiniteElementMethod.underivable_function
FiniteElementMethod.AssemblyContext
FiniteElementMethod.assemble_linear
FiniteElementMethod.assemble_nonlinear
FiniteElementMethod._split
FiniteElementMethod.assemble_linear!
FiniteElementMethod.SUPG
FiniteElementMethod.emit_strong
FiniteElementMethod.compile_coeffs
FiniteElementMethod.CellCoeffs
FiniteElementMethod.StructuralZero
FiniteElementMethod.fused_terms
FiniteElementMethod.LinearBlock
```

## Operators

```@docs
FiniteElementMethod.NonlinearOperator
FiniteElementMethod.residual!
FiniteElementMethod.jacobian!
FiniteElementMethod.residual_jacobian!
FiniteElementMethod.LinearCache
FiniteElementMethod.update!
```

## Precompilation

```@docs
FiniteElementMethod.workload_chain
```
