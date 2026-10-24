# API

These are internal building blocks of the Ferrite backend; none is exported yet.

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
