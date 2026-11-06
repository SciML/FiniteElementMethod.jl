# MaxwellGPU

Tests of `CuDSSSolver`, the cuDSS GPU solver of `solve_sweep` (the `FiniteElementMethodCUDSSExt` extension). The
group has its own environment (CUDA, CUDSS, JET) and is not part of `All` or of CI, which has no NVIDIA GPU.

Run it by hand on a machine with an NVIDIA GPU and a working CUDA driver, from the repository root:

```sh
GROUP=MaxwellGPU julia --project=. -e 'using Pkg; Pkg.test()'
```

Without a functional CUDA device (`CUDA.functional() == false`) the group only checks that the extension loads
and infers (JET), and the WR-90 sweep against UMFPACK is reported as skipped (one `Broken` test). On a CUDA
machine that comparison runs for `structure = "G"`, and also for `"S"` with CUDSS.jl 1.x.
