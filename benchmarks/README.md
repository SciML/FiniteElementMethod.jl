# Benchmarks

Reproducible scripts for the headline comparisons of the design specification (§11.5). They live in their own
environment, so MethodOfLines, OrdinaryDiffEqBDF and PDESystemLibrary never become dependencies of the package.

| script | benchmark |
|---|---|
| `accuracy_convdiff.jl` | (a) accuracy per degree of freedom against MethodOfLines on PDESystemLibrary's `convdiff2` |
| `assembly_cost.jl` | (b) assembly and residual cost against hand-written Ferrite, 100×100 P1, cases (i) and (iii) |
| `construction.jl` | (c) `DAEProblem` construction time at N = 1e4, numeric path against ModelingToolkit codegen when available |

## Running

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks benchmarks/run_all.jl          # all three, each in a fresh process
julia --project=benchmarks benchmarks/assembly_cost.jl    # or one at a time
julia benchmarks/render.jl                                # write docs/src/benchmarks.md from the results
```

The environment uses the package in this repository (`[sources]`, Julia 1.11 or later) and, until a PDEBase release
contains the mesh-discretization layer (PDEBase#158), PDEBase from that branch.

Each script writes `results/<name>.toml` with its numbers and an `[environment]` table: where it ran (`origin`,
set by the Benchmarks workflow through `BENCHMARK_ORIGIN`, otherwise "local run, not CI"), the date, the git
commit, the CPU and thread counts, and the Julia and package versions. The documentation page is rendered from
those files at build time; numbers are never pasted by hand.

Environment variables, all optional: `BENCHMARK_RESULTS_DIR` (where to write), `BENCHMARK_RUNS` (timing repetitions
for (b), default 10), `BENCHMARK_ASSEMBLY_NEL` (default 100), `BENCHMARK_CONVDIFF_SIZES` (default `16,32,64,128`),
`BENCHMARK_CONSTRUCTION_SIDES` (nodes per side, default `50,100`, i.e. N = 2500 and 1e4) and
`BENCHMARK_CONSTRUCTION_RUNS` (default 3).

Timings from the workflow come from shared GitHub runners and vary between runs; ratios to a hand-written baseline
measured in the same process ((b)) are the robust figures.
