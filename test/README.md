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
  condition come from PDEBase. With the whole PDEBase M0 stack (`MeshDomain`, `BoundaryPiece` and
  `NormalDerivative` slotting, PDEBase#154), conditions written with `NormalDerivative` on the pieces of a
  `MeshDomain` go through PDEBase as well; without it, interior-form conditions on non-box pieces fall back to the
  seam's stage d against PDEBase's fluxes.

Checks that only PDEBase passes use `@test_pdebase`, which is `@test` on the PDEBase path and `@test_broken` on the
seam path; checks that need the mesh pieces use `@test_pdebase_mesh`, which is `@test` only with the M0 stack.

To run the Integration group on the PDEBase path, add a PDEBase development revision to a throwaway environment.
Never commit a Project or Manifest that points at one. Two revisions of xtalax-droid/PDEBase.jl are in use:

- `80715ae` (`forge/strongform`, SciML/PDEBase.jl#148 and #149): box-face boundary conditions, fields in declared
  order, conditions in input order. The mesh-piece rows stay `@test_broken`.
- `f14989d` (`forge/m0-all`, the whole M0 stack including #154): also the mesh-piece rows.

From the repository root, with `rev` one of those:

```sh
rev=f14989d
env=$(mktemp -d)
julia --project="$env" -e "using Pkg
    Pkg.develop(path = \".\")
    Pkg.add(url = \"https://github.com/xtalax-droid/PDEBase.jl\", rev = \"$rev\")
    Pkg.add([\"Test\", \"SafeTestsets\", \"SciMLTesting\", \"LinearAlgebra\", \"Random\", \"Ferrite\", \"Symbolics\",
        \"DomainSets\", \"ModelingToolkit\", \"SciMLBase\"])"
GROUP=Integration julia --project="$env" test/runtests.jl
```

The same command with the `Pkg.add(url = …)` line removed runs the seam path in that environment.
