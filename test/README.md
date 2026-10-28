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

- `3dc036b` (`forge/m0-all`, the whole M0 stack: StrongForm, `MeshDomain`, `BoundaryPiece` and `NormalDerivative`
  slotting from #154, and the public `parsed_boundary_map` from #155): every row, including the mesh-piece ones. This
  is the default.
- `80715ae` (`forge/strongform`, SciML/PDEBase.jl#148 and #149), the minimal one: box-face boundary conditions,
  fields in declared order, conditions in input order. The mesh-piece rows stay `@test_broken`.

From the repository root, with `rev` one of those:

```sh
rev=3dc036b
env=$(mktemp -d)
julia --project="$env" -e "using Pkg
    Pkg.develop(path = \".\")
    Pkg.add(url = \"https://github.com/xtalax-droid/PDEBase.jl\", rev = \"$rev\")
    Pkg.add([\"Test\", \"SafeTestsets\", \"SciMLTesting\", \"LinearAlgebra\", \"Random\", \"Ferrite\", \"Symbolics\",
        \"DomainSets\", \"ModelingToolkit\", \"SciMLBase\"])"
GROUP=Integration julia --project="$env" test/runtests.jl
```

The same command with the `Pkg.add(url = …)` line removed runs the seam path in that environment.
Tests follow the SciML grouped layout (as in MethodOfLines.jl). `test/runtests.jl` calls
`SciMLTesting.run_tests()`, which picks the group from the `GROUP` environment variable and
discovers test files from folders. `test/test_groups.toml` is the single source of truth for
the group list and for the CI matrix: `.github/workflows/Tests.yml` fans out one job per entry.

| Group  | Files run                                  | Environment            |
|--------|--------------------------------------------|------------------------|
| `Core` | top-level `test/*.jl` (not `runtests.jl`)  | main test environment  |
| `QA`   | `test/qa/*.jl` (Aqua and JET)              | `test/qa/Project.toml` |
| `X`    | `test/X/*.jl`, for any other declared `X`  | `test/X/Project.toml` if present, else main |

An unset `GROUP` runs `Core` plus every declared group except `QA`; `GROUP=Everything` also
runs `QA`.

Run a group locally:

```sh
GROUP=Core julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=QA julia --project=. -e 'using Pkg; Pkg.test()'
```

## Adding a group

1. Create `test/<Group>/` containing one or more `*.jl` test files. Every `*.jl` in the folder
   runs, each in its own `@safetestset`, so nothing needs registering.
2. Add one entry to `test/test_groups.toml`:

   ```toml
   [<Group>]
   versions = ["lts", "1", "pre"]
   ```

3. Optional: if the group needs extra dependencies, add `test/<Group>/Project.toml`; it is
   activated with the package developed into it, as `test/qa/` does.

A declared group whose folder is missing or empty is an error, so only declare groups that
have tests. Planned milestone groups: `StrongForm`, `Assembly`, `Space`, `DAE`, `Solution`,
`Convergence`, `MOLComparison`, `Library`.

## Shared helpers

Put code that several groups `include` in `test/shared/`. It is not a declared group, so it is
never run on its own.

## Coverage

Coverage is collected by the reusable `SciML/.github/.github/workflows/grouped-tests.yml`
workflow that `Tests.yml` calls (it runs each group with coverage and uploads to Codecov using
the inherited `secrets`), so the caller needs no coverage steps of its own, as in MethodOfLines.jl
and PDEBase.jl.

Every code path needs an end-to-end test: a `PDESystem` is discretized, solved, and compared
with an exact or manufactured solution (or with MethodOfLines.jl) in a MOL-style group, not only
a unit test of the internals. The mediator keeps a checklist of code paths and their covering
tests in `.forge/branch-coverage.md`; update it when you add a path or a test.
