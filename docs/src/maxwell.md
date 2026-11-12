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
individual incident or outgoing power, and `S` is not power conserving. Wave ports need positive frequencies; a
zero or negative `ω` is rejected, since a real system has `S(-ω) = conj(S(ω))`.

## From a PDESystem

`FEMDiscretization(Ω, t; omega, spaces = Dict(E => Nedelec{RefTetrahedron, 1}()))` discretizes the
time-domain equation `curl(ν curl E) + a₂ ∂ₜ²E + a₁ ∂ₜE + a₀ E = f` for a vector unknown `E` at the angular
frequency `omega`, and `discretize` returns the complex `LinearProblem` that [`driven_problem`](@ref) builds
for the equivalent `MaxwellModel` (in units with `ε0 = μ0 = 1`). The operator is written with [`curl`](@ref)
and the perfect conductor with [`facet_normal`](@ref).

```julia
using FiniteElementMethod, Ferrite, ModelingToolkit, DomainSets, LinearAlgebra, LinearSolve
grid = generate_grid(Tetrahedron, (8, 8, 8), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
Ω = MeshDomain(grid)
@parameters x y z t
@variables E(..)[1:3]
X, Ev, Dt = (x, y, z), [E(t, x, y, z)[c] for c in 1:3], Differential(t)
cc = curl(curl(Ev, X) / 2, X)    # μr = 2
eqs = [cc[c] + 4 * Dt(Dt(Ev[c])) + 0.1 * Dt(Ev[c]) ~ (c == 3) * exp(-10 * ((x - 0.5)^2 + (y - 0.5)^2)) for c in 1:3]
pec = reduce(vcat, [cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0 for Γ in boundarypieces(Ω)])
@named sys = PDESystem(eqs, pec, [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z)])
disc = FEMDiscretization(Ω, t; omega = 2.0, spaces = Dict(E => Nedelec{RefTetrahedron, 1}()))
sol = solve(discretize(sys, disc))
(; model, omega) = maxwell_model(sys, disc)
field_values(MaxwellSolution(model, omega, sol.u), [Vec(0.5, 0.5, 0.5)]).E
```

Coefficients are real, so a complex permittivity `ε' - iε''` is only available on the operator-level
`MaxwellModel`.

### Absorbing boundaries, ports and sweeps

The first-order absorbing condition is written in its time-domain form,
`n × (ν ∇ × E) = -sqrt(ε/μ) n × (n × ∂ₜE)` with `ε = a₂` and `ν = 1/μ` of the cells next to the piece. With
`∂ₜ = iω` and `ε0 = μ0 = 1` this is the model's `n × (μr⁻¹ ∇ × E) = -i k0 sqrt(εr/μr) n × (n × E)`, so the
piece goes to `absorbing`. Any equivalent set of equations is accepted; at the actual normal of every facet the
conditions must span exactly the two tangential equations of this one, and a different coefficient of
`curl E`, a different impedance, the opposite sign or an added term is an error. The cells next to the piece
must be lossless (`a₁ = a₀ = 0`); a lossy one has no time-domain form here and needs `MaxwellModel`'s
`absorbing` keyword.

A [`WavePort`](@ref) is modal data rather than a local condition, so ports are the `ports` keyword of
`FEMDiscretization`, on pieces without a condition in the `PDESystem`. `discretize` treats them as matched
terminations of the source problem; `solve_sweep(pdesys, disc, omegas)` drives them and returns the scattering
matrices, as `solve_sweep(maxwell_model(pdesys, disc).model, omegas)` does. With ports `a₀` must be zero, so
that the model does not depend on `omega`, and a sweep rejects a volume source.

```julia
grid = generate_grid(Tetrahedron, (12, 6, 15), Vec(0.0, 0.0, 0.0), Vec(22.86, 10.16, 30.0))  # WR-90, mm
Ω = MeshDomain(grid)
n, Eb = facet_normal(boundarypiece(Ω, :top)), [E(t, boundarypiece(Ω, :top))[c] for c in 1:3]
abc = cross(n, curl(Eb, X)) .~ -cross(n, cross(n, Dt.(Eb)))
pec = reduce(vcat, [cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0 for Γ in boundarypiece.(Ref(Ω), [:left, :right, :front, :back])])
eqs = [curl(curl(Ev, X), X)[c] + Dt(Dt(Ev[c])) ~ 0 for c in 1:3]
@named sys = PDESystem(eqs, [pec; abc], [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z)])
k0 = 2π * 10.0e9 / FiniteElementMethod.C0 * 1.0e-3  # 10 GHz in rad/mm
ports = [WavePort(boundarypiece(Ω, :bottom); a_direction = (1, 0, 0))]
disc = FEMDiscretization(Ω, t; omega = k0, spaces = Dict(E => Nedelec{RefTetrahedron, 1}()), ports)
solve_sweep(sys, disc, k0 .* [0.8, 1.0, 1.2]).S  # |S11| ≈ 0.14 at 10 GHz, the reflection of the absorbing end
```

```@docs
curl(::AbstractVector, ::NTuple{3, Any})
facet_normal
maxwell_model
```

## Operator level

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
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```
