# Tests

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
have tests.

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

## The strong-form analysis behind the Integration group

The Integration group runs its chains through PDEBase: `test/shared/references.jl` adapts `PDEBase.StrongForm.analyse`
to the seam's interface, so the volume terms and every boundary condition (box faces, and the pieces of a
`MeshDomain` written with `NormalDerivative`) come from PDEBase. PDEBase numbers fields in the declared order and
emits conditions in input order. `test/shared/strongform_seam.jl` is the test-only stand-in the Assembly, Kernels
and WeakForm groups still use directly; the Integration adapter calls its `boundary_conditions` stage only for
seam `BC`s on facet sets that are not box faces.
