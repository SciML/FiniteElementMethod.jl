# GPU group

Tests of `FiniteElementMethodKAExt`, the KernelAbstractions extension. It loads when KernelAbstractions, Adapt,
GPUArrays and GPUArraysCore are all loaded. The group is opt-in (`in_all = false`) and runs on Julia 1 only, because
`Project.toml` develops the package through `[sources]`, which Julia 1.10 ignores. Device assembly needs Ferrite's
KernelAbstractions assembler, which no Ferrite release has yet: `[sources]` takes Ferrite from the branch of Ferrite.jl
PR #1496 (`do/gpu-ci`), while the package itself keeps `Ferrite = "1.7"`.

```sh
GROUP=GPU julia --project=. -e 'using Pkg; Pkg.test()'
```

## What is tested

The kernels come from real `PDESystem`s (`common/cases.jl`). They cover mass, bilinear and source coefficients, a
Neumann facet kernel, SUPG cell coefficients with the element size, a nonlinear `f` and `tangent`, `NaNMath.pow` with
constant and symbolic exponents, time-dependent Dirichlet data, and a registered function. For this layer the
end-to-end check runs from the `PDESystem` to its emitted kernels, through the device pass, to a KernelAbstractions
kernel whose result must equal the host Float64 kernel.

- `devicepass.jl` runs on the host. It checks that no emitted kernel keeps a Float64 literal, a closure, a ccall or
  `NaNMath.pow` after the pass. A Float32 kernel returns no Float64 and stays within 1e-6 relative of the Float64
  kernel. `NaNMath.lgamma` and the other library ccalls are refused when the kernel is prepared, and the error for a
  user function that fails to compile names that function.
- `assembly.jl` runs `discretize` with `backend` set on `KernelAbstractions.CPU()` and on JLArrays (`common/assembly.jl`):
  a 3D P1 hexahedron and a 2D P2 triangle heat problem and a two-field problem of P2 and P1, each with constant and
  time-dependent Dirichlet data, in Float32 and Float64, colored and atomic. M, K and F equal the host assembly within
  1e-5 relative in Float32 and 1e-12 in Float64, and the solution at a point equals the host solution. Colored
  assembly is deterministic, and a steady problem assembled on the device solves to the host solution.
- `extension.jl` checks that FEM's and Ferrite's KernelAbstractions extensions are loaded, the validation of the
  `backend`, `precision` and `assembly` keywords, the error for a Ferrite without GPU support, and the `ArgumentError`
  for each feature device assembly does not cover yet. The error for a `backend` without the extension is tested in
  `test/Frontend/backend.jl`, where the extension never loads.
- `backends.jl` evaluates every prepared kernel in a `@kernel` on `KernelAbstractions.CPU()` and on JLArrays, in
  Float32 and Float64.

## What the CPU backends cannot catch

`KernelAbstractions.CPU()` and JLArrays test host discipline only. JLArrays 0.3.3 is the newest release that accepts
KernelAbstractions 0.9. It runs every kernel as a `KernelAbstractions.CPU()` kernel over plain `Array`s and has no
`synchronize`, so the harness skips that call. Code that would not compile for a GPU, such as a registered function
that allocates, passes on both. `JLSparseMatrixCSC` fails with scalar indexing, while a `GenericSparseMatrixCSC` with
JLArray storage works.

## Vendor runs

`vendor/amdgpu/` is not a declared group. Run it by hand on a machine with an AMD GPU, one GPU process at a time:

```sh
julia --project=test/GPU/vendor/amdgpu -e 'using Pkg; Pkg.instantiate()'
julia --project=test/GPU/vendor/amdgpu test/GPU/vendor/amdgpu/runtests.jl
```

It runs the same kernels and the device assembly cases of `common/assembly.jl` on `ROCBackend()`. It also checks that a registered function that allocates a `Dict` fails to
compile with an error that names it.

| Backend | Status |
|---------|--------|
| `KernelAbstractions.CPU()`, JLArrays | supported: run in CI by the GPU group |
| AMDGPU | tested on Radeon 8060S (gfx1151), by hand |
| CUDA, oneAPI, Metal | untested |
