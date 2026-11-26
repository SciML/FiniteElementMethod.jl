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

## Perfectly matched layers

A [`CartesianPML`](@ref) on the `pml` keyword turns the cells of named cell sets, outside an axis-aligned
inner box (by default the bounding box of the other cells), into the uniaxial absorber
`εr Λ`, `σ Λ`, `μr Λ` with `Λ = diag(s_y s_z/s_x, s_x s_z/s_y, s_x s_y/s_z)` and `s_j = 1 - iα_j`. On each face
of the box, `α` grows as `α_max (ξ/d)^order` with the depth `ξ` into a layer of thickness `d`, and
`α_max = -(order + 1) ln R / (2 k_ref d)` sets the normal-incidence round-trip reflection `R` of a PEC-backed
layer at the reference angular frequency `omega`. Name the outer faces of the layer in `pec`.
The layer must be filled with materials of positive real `eps_r` and positive `mu_r` (loss is allowed); a
negative-index filling is rejected, because the stretching would amplify its outgoing wave.

The stretching is frequency-independent (the usual `s = 1 - iσ/ω` frozen at `ω = omega`), so the layer is a
fixed set of real blocks with polynomial coefficients in `ω`, like the rest of the operator, and a sweep still
only recombines them. The price is that the attenuation scales with the wavenumber: at `ω` the layer
reflects about `R^(k/k_ref)` (`R^(β/k_ref)` for a guided mode), which is stronger above the reference and
weaker below it. In practice the mesh reflection of the graded layer, not `R`, sets the floor: on the WR-90
example below, `|S11|` stays between 0.002 and 0.004 from 8 to 12 GHz with the reference at 10 GHz, within
about 20 % of a layer re-tuned to each frequency, and falls with the cells across the layer (0.018, 0.009, 0.004 and
0.002 for 4, 6, 10 and 15 cells). A layer of only a few cells absorbs better with a milder `R` (`1e-3`).

```julia
grid = generate_grid(Tetrahedron, (12, 6, 20), Vec(0.0, 0.0, 0.0), Vec(22.86, 10.16, 40.0))
addcellset!(grid, "pml", x -> x[3] >= 19.99)  # the last 20 mm absorb
model = MaxwellModel(
    grid; pec = ["left", "right", "front", "back", "top"], ports = [WavePort("bottom"; a_direction = (1, 0, 0))],
    pml = CartesianPML("pml"; omega = 2π * 10.0e9, R = 1.0e-6), length_scale = 1.0e-3,
)
abs(only(solve_sweep(model, [2π * 10.0e9]).S))  # |S11| ≈ 0.004, against 0.1396 for the first-order ABC
```

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
FiniteElementMethod.tangential_mass
FiniteElementMethod.port_excitations
FiniteElementMethod.s_parameters
FiniteElementMethod.propagation_constant
FiniteElementMethod.cutoff_frequency
CartesianPML
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```

## Material sensitivities

[`s_parameter_sensitivities`](@ref) returns the derivatives of the wave-port S-parameters with respect to
material parameters of named cell sets: `:eps_r` (the real part of `εr`), `:eps_r_imag` (its imaginary
part), `:mu_r` and `:sigma`, each shifting that parameter on every cell of the set. They are the exact
derivatives of the discrete S-parameters, by the discrete adjoint: the system matrix is complex symmetric,
so the adjoint field of port `i` is the forward field with port `i` driven, and the sensitivities cost no
linear solve beyond the sweep, only one sparse product per parameter and frequency. A parameter set may
touch an absorbing boundary (its `sqrt(εc/μr)` is differentiated) but not a wave port, whose mode is taken
in fixed material, and not a PML cell, whose stretched blocks also depend on the material: either is an `ArgumentError`.

```julia
grid = generate_grid(Tetrahedron, (12, 6, 15), Vec(0.0, 0.0, 0.0), Vec(22.86, 10.16, 30.0))
addcellset!(grid, "slab", x -> 10.0 - 1.0e-9 <= x[3] <= 20.0 + 1.0e-9; all = true)
ports = [WavePort(face; a_direction = (1, 0, 0), deembed = 10.0) for face in ("bottom", "top")]
model = MaxwellModel(grid; materials = Dict("slab" => MaxwellMaterial(; eps_r = 2.2)),
    pec = ["left", "right", "front", "back"], ports, length_scale = 1.0e-3)
sens = s_parameter_sensitivities(model, [2π * 10.0e9], [(:eps_r, "slab"), (:sigma, "slab")])
sens.dS[1, 1, 1, 1]  # dS11/dεr at 10 GHz
```

For a real objective such as `|S11|²`, the gradient is `2 Re(conj(S11) dS11/dp)`. Other real functionals
`J(u)` of the field take one adjoint solve with [`adjoint_gradient`](@ref).

```@docs
s_parameter_sensitivities
MaxwellSensitivities
adjoint_gradient
```
