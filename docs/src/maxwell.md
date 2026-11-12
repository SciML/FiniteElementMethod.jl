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

A port must lie in a lossless material: a complex `eps_r` or a nonzero conductivity at a port is rejected, because the
power-wave normalisation above holds for a real wave impedance. Put loss behind the port, not in its face, and a
facet cannot belong to two ports or to both a port and an `absorbing` set.

## Lumped ports

A [`LumpedPort`](@ref) is a rectangular facet sheet bridging a gap between two conductors, of length `l`
along `direction` and width `w` across it, on the boundary or inside the mesh. It carries a uniform sheet
current `I/w` along `direction`; its voltage is the sheet average of the line integral of `E` across the gap,
`V = (1/w) ∮ E⋅l̂ dS`. Every port is loaded by its reference impedance `Z0`, which adds
`iωμ0 L/(w² Z0) g gᵀ` (`gᵢ = ∮ l̂⋅Nᵢ dS`) to the operator, and the driven port adds a 1 A current source in
parallel, so `Sᵢⱼ = 2Vᵢ/sqrt(Z0ᵢZ0ⱼ) - δᵢⱼ`. Because the load, the source and the voltage use the same `g`,
these circuit identities hold exactly for the discrete system: a lossless structure gives a unitary `S`.
Wave and lumped ports can share one model and one sweep; lumped ports and propagating wave ports are
normalised to power waves.

```julia
using FiniteElementMethod, Ferrite
# A parallel-plate line (PEC plates at y = 0 and y = 5 mm, open sides) with a 50 Ω port at each end.
grid = generate_grid(Tetrahedron, (2, 2, 20), Vec(0.0, 0.0, 0.0), Vec(10.0, 5.0, 20.0))
ports = [LumpedPort("bottom"; direction = (0, 1, 0)), LumpedPort("top"; direction = (0, 1, 0))]
model = MaxwellModel(grid; pec = ["front", "back"], ports, length_scale = 1.0e-3)
S = solve_sweep(model, [2π * 3.0e9]).S[:, :, 1]  # S11 ≈ 0.846 + 0.136i, S21 ≈ 0.082 - 0.508i
Z = FiniteElementMethod.z_parameters(model, S)
```

The port couples all its edge dofs to each other, which widens the sparsity pattern, so keep the sheet a
few elements across.

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
LumpedPort
FiniteElementMethod.LumpedPortData
FiniteElementMethod.port_voltage
FiniteElementMethod.port_current
FiniteElementMethod.z_parameters
solve_sweep
MaxwellSweep
FiniteElementMethod.umfpack_solver
FiniteElementMethod.tangential_mass
FiniteElementMethod.port_excitations
FiniteElementMethod.s_parameters
FiniteElementMethod.propagation_constant
FiniteElementMethod.cutoff_frequency
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```
