# Frequency-domain Maxwell

`MaxwellModel` is a separate entry point from the `PDESystem` front end: it assembles the time-harmonic
electric-field equation directly with Ferrite, using lowest-order Nédélec edge elements
(`Nedelec{RefTetrahedron, 1}`) on a tetrahedral mesh. With fields varying as `exp(+iωt)`, it solves

```math
\nabla \times \left(\mu_r^{-1} \nabla \times \mathbf{E}\right) + i\omega\mu_0\sigma\mathbf{E}
- \omega^2\mu_0\varepsilon_0\varepsilon_r\mathbf{E} = -i\omega\mu_0\mathbf{J},
```

with `n × E = 0` on the perfectly conducting (PEC) facet sets. Materials are assigned per cell set, as
`FerriteGmsh.togrid` produces them from Gmsh physical groups; unnamed cells are vacuum. A passive lossy
dielectric has `εr = ε' - iε''` with `ε'' ≥ 0`.

```julia
using FiniteElementMethod, Ferrite
grid = generate_grid(Tetrahedron, (8, 8, 8), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
addcellset!(grid, "substrate", x -> x[3] <= 0.25)
model = MaxwellModel(
    grid; materials = Dict("substrate" => MaxwellMaterial(; eps_r = 4.4 - 0.08im)),
    pec = ["left", "right", "bottom", "top", "front", "back"], length_scale = 1.0e-2,  # mesh in cm
)
sol = solve_driven(model, 2π * 5.0e9, x -> [0.0, 0.0, exp(-sum(abs2, x .- 0.5) / 0.01)])
field_values(sol, [Vec(0.5, 0.5, 0.5)]).E
```

The system matrix is a sum of real sparse blocks on one sparsity pattern, each with a complex coefficient
that is a function of `ω`, so a frequency sweep only recombines nonzero values with
[`maxwell_operator!`](@ref), and further terms (boundary conditions, ports) are added as blocks with
`FiniteElementMethod.add_term!`.

## Wave ports, absorbing boundaries and frequency sweeps

Rectangular-waveguide ports and first-order absorbing boundaries are model keywords. A [`WavePort`](@ref)
on a planar boundary facet set launches and absorbs the TE10 mode of the rectangle it spans; `a_direction`
names the broad wall. Facet sets in `absorbing` get the first-order Silver-Müller condition
`n × (μr⁻¹ ∇ × E) = -i k0 sqrt(εc/μr) n × (n × E)` (`εc = εr - iσ/(ωε0)` of the adjacent material), exact for a plane wave at normal incidence (a TE10 wave
reflects `Γ = (β - k0)/(β + k0)` from it). [`solve_sweep`](@ref) returns the scattering matrix at each
angular frequency; it keeps one operator matrix and one factorisation pattern, and solves every port's
excitation from a single factorisation per frequency. The linear solver is the `solver` keyword (see
[`FiniteElementMethod.umfpack_solver`](@ref)).

```julia
using FiniteElementMethod, Ferrite
# A 50 mm section of WR-90 (22.86 mm × 10.16 mm), mesh in millimetres.
grid = generate_grid(Tetrahedron, (16, 7, 34), Vec(0.0, 0.0, 0.0), Vec(22.86, 10.16, 50.0))
ports = [WavePort("bottom"; a_direction = (1, 0, 0)), WavePort("top"; a_direction = (1, 0, 0))]
model = MaxwellModel(grid; pec = ["left", "right", "front", "back"], ports, length_scale = 1.0e-3)
sweep = solve_sweep(model, 2π .* (8.0e9:0.5e9:12.0e9))
sweep.S[2, 1, 5]  # S21 at 10 GHz: |S21| ≈ 1, angle ≈ -93.3°
```

`Sᵢⱼ` is referred to the port faces, or to planes `deembed` mesh units inside them. Ports on different guides
are normalised through the TE10 wave impedance `ωμ0μr/β`. For a propagating port these are power waves. Below cutoff
the propagation constant is `β = -iα`, the evanescent branch, which carries no real axial power: the coefficients
are then generalized traveling-wave coefficients with the same normalisation, still supported, but without an
individual incident or outgoing power, and `S` is not power conserving.

A wave port is a single-mode TE10 termination: its admittance acts on the TE10 amplitude of the trace only (a rank-one
term that couples all the dofs of the port face), and the evanescent higher modes see the natural (magnetic-wall)
boundary at the port plane, so they are reflected, not absorbed, and a lossless single-mode structure conserves
power. Keep about one decay length `1 / sqrt(kc² - k²)` of the first evanescent mode (`kc = 2π/a` for TE20, `π/b` for
TE01) of uniform lead between a discontinuity and the port, otherwise `S` depends on the lead length. Wave ports need positive frequencies; a
zero or negative `ω` is rejected, since a real system has `S(-ω) = conj(S(ω))`.

A port must lie in an ordinary passive material: a complex `eps_r`, a nonzero conductivity, a non-positive `eps_r`
or a non-positive `mu_r` at a port is rejected, because the power-wave normalisation above holds for a real wave
impedance and the outward power of the TE10 mode is `Re(β/μr) > 0` only for positive `μr` and `εr`. Put loss behind the port, not in its face, and a
facet cannot belong to two ports or to both a port and an `absorbing` set.

### Solving on an NVIDIA GPU

`solver = CuDSSSolver()` moves the factorisation and solves of [`solve_sweep`](@ref) to an NVIDIA GPU through
[cuDSS](https://developer.nvidia.com/cudss); it needs `using CUDA, CUDSS` and a functional CUDA device.
Assembly stays on the CPU. The first frequency uploads the operator and runs the cuDSS analysis and
factorisation; later frequencies copy only the new nonzero values and refactorise, reusing the analysis, and
every port excitation is solved in one call.

```julia
using CUDA, CUDSS
sweep = solve_sweep(model, 2π .* (8.0e9:0.5e9:12.0e9); solver = CuDSSSolver())
```

The default is cuDSS's general LU (`structure = "G"`). `CuDSSSolver(; structure = "S")` uses its complex
symmetric LDLᵀ instead (the Maxwell operator is complex symmetric, not Hermitian), with about half the factor
storage; CUDSS.jl documents complex symmetric matrices only from its version 1.0. CUDSS.jl 1.0 currently cannot be
installed next to this package (its ArrayInterface bound conflicts with the one of DiffEqBase), so use CUDSS.jl 0.8
with the default general LU until it resolves; the extension accepts both.

The CPU side of this solver (the CSR layout and the per-frequency value update) is tested in the Maxwell test
group, and the `MaxwellGPU` group checks that the extension loads and infers. The comparison of the GPU sweep with
UMFPACK on a WR-90 guide runs only where `CUDA.functional()` holds; CI has no NVIDIA GPU, so run it by hand on a
CUDA machine with `GROUP=MaxwellGPU julia --project=. -e 'using Pkg; Pkg.test()'`.

```@docs
MaxwellModel
MaxwellMaterial
MaxwellTerm
maxwell_operator
maxwell_operator!
driven_problem
solve_driven
MaxwellSolution
field_values
FiniteElementMethod.maxwell_rhs
FiniteElementMethod.add_term!
WavePort
FiniteElementMethod.WavePortData
solve_sweep
MaxwellSweep
FiniteElementMethod.umfpack_solver
CuDSSSolver
FiniteElementMethod.tangential_mass
FiniteElementMethod.port_excitations
FiniteElementMethod.s_parameters
FiniteElementMethod.propagation_constant
FiniteElementMethod.cutoff_frequency
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```
