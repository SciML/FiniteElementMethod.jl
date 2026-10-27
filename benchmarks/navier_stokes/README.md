# Benchmark (d): Navier–Stokes flow past a cylinder

`ns_cylinder.jl` measures, in one process and after a warm-up solve on each side:

1. Wall-clock against Ferrite's `ns_vs_diffeq.jl` tutorial (`ferrite_ns_vs_diffeq.jl`, the tutorial's code with
   the changes listed at its top) on the same mesh, `dfg_channel_grid(h = 0.05, length = 1.1)`, with the
   tutorial's settings: ν = 1e-3, inflow ramp `min(1.5t, 1.5)` (Re = 100), T = 6, abstol 1e-4, reltol 1e-5,
   output every 0.1. Both sides solve one discrete problem: the outlet treatment of the PDESystem example (∇p
   tested strongly, p = 0 on the outlet), which Ferrite's side gets from `outlet = :pressure`. Ferrite runs Rodas5P
   on its mass-matrix ODE; FiniteElementMethod discretizes the PDESystem of `examples/dfg_cylinder` to a
   DAEProblem and runs DFBDF. The horizontal velocities are compared at t = 1 and t = T at 20 points behind the
   cylinder, and the time ratio is recorded only if they agree to 1% of the largest velocity there. The tutorial as
   published (do-nothing outlet) is timed too, for reference. The time steppers and their error norms differ, so
   the times are comparable in magnitude, not a measure of which package is faster.
2. DFG 2D-1 (Re = 20) drag, lift and pressure difference on the full channel at the meshes of
   `BENCHMARK_NS_MESHES` (`h:hc` pairs, default `0.04:0.0025,0.02:0.00333,0.01:0.00167`, up to 149 094 unknowns),
   with their errors against the DFG reference values.

```sh
julia --project=benchmarks/navier_stokes -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/navier_stokes benchmarks/navier_stokes/ns_cylinder.jl
```

The results go to `benchmarks/results/ns_cylinder.toml` (or `BENCHMARK_RESULTS_DIR`) with the environment record of
`benchmarks/common.jl`: origin, date, commit, CPU, thread counts, Julia and package versions. Peak memory of the
default run is about 6.5 GB, from the 149 094-unknown 2D-1 solve; the run takes about 10 minutes on one thread.
The environment uses this repository's package and, until a PDEBase release has the mesh layer, PDEBase from the
`forge/mesh-axis-order` branch (`[sources]`).
