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

## From a PDESystem

`FEMDiscretization(Ω, t; omega, spaces = Dict(E => Nedelec{RefTetrahedron, 1}()))` discretizes the
time-domain equation `curl(ν curl E) + a₂ ∂ₜ²E + a₁ ∂ₜE + a₀ E = f` for a vector unknown `E` at the angular
frequency `omega`, and `discretize` returns the complex `LinearProblem` that [`driven_problem`](@ref) builds
for the equivalent `MaxwellModel` (in units with `ε0 = μ0 = 1`). The operator is written with [`curl`](@ref)
and the perfect conductor with [`facet_normal`](@ref), which needs a PDEBase with facet normals.

```julia
using FiniteElementMethod, Ferrite, ModelingToolkit, LinearAlgebra, LinearSolve
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
`MaxwellModel`. Absorbing boundaries and ports are not yet part of the `PDESystem` path.

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
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```
