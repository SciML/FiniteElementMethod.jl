# Tests

Test groups follow SciMLTesting's folder layout: one `test/<Group>/` folder per group declared in
`test_groups.toml`, with shared helpers in `test/shared/`. Run a group with

```sh
GROUP=Integration julia --project=. -e 'using Pkg; Pkg.test()'
```

## The strong-form analysis behind the Integration group

`test/shared/references.jl` picks the strong-form analysis at load time:

- **Seam path** (released PDEBase, and CI): PDEBase has no `StrongForm.analyse`, so the whole analysis comes from
  the test-only `test/shared/strongform_seam.jl`.
- **PDEBase path**: when `PDEBase.StrongForm.analyse` exists, the volume terms and every box-face boundary
  condition come from PDEBase. Conditions on mesh pieces still go through the seam's stage d against PDEBase's
  fluxes, until PDEBase's piece and `NormalDerivative` slotting (strongform PR 5) lands.

Checks that only PDEBase passes use `@test_pdebase`, which is `@test` on the PDEBase path and `@test_broken` on the
seam path.

To run the Integration group on the PDEBase path, develop the strongform branch
(xtalax-droid/PDEBase.jl `forge/strongform` at 80715ae, SciML/PDEBase.jl#148 and #149, with box-face boundary
conditions, fields in declared order and conditions in input order) into a throwaway environment. Never
commit a Project or Manifest that points at it. From the repository root:

```sh
env=$(mktemp -d)
julia --project="$env" -e 'using Pkg
    Pkg.develop(path = ".")
    Pkg.add(url = "https://github.com/xtalax-droid/PDEBase.jl", rev = "80715ae")
    Pkg.add(["Test", "SafeTestsets", "SciMLTesting", "LinearAlgebra", "Random", "Ferrite", "Symbolics",
        "DomainSets", "ModelingToolkit"])'
GROUP=Integration julia --project="$env" test/runtests.jl
```

The same command with the `Pkg.add(url = …)` line removed runs the seam path in that environment.
