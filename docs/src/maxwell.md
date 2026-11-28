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

## Resonant modes

[`maxwell_eigenmodes`](@ref) finds the modes nearest a target frequency by shift-invert Arnoldi, with the
PEC dofs eliminated and the `ω = 0` gradient null space of the edge elements projected out, so any target
works, including one below the lowest mode. It needs `using KrylovKit`. A lossy material gives complex
`ω`; the quality factor is `Q = Re(ω) / (2 Im(ω))`. Only ordinary passive materials are supported: `real(εr)` and `μr` must be positive (the gradient projection needs a definite mass form), otherwise an `ArgumentError` names the cell set. An air-filled 100 × 50 × 75 mm box has its TE101 mode
at 2.498 GHz:

```julia
using FiniteElementMethod, Ferrite, KrylovKit
grid = generate_grid(Tetrahedron, (12, 6, 9), Vec(0.0, 0.0, 0.0), Vec(100.0, 50.0, 75.0))
model = MaxwellModel(grid; pec = ["left", "right", "bottom", "top", "front", "back"], length_scale = 1.0e-3)
modes = maxwell_eigenmodes(model; nev = 4, target_frequency = 2.0e9)
modes.frequency, modes.Q                          # Hz; Q is Inf without loss
field_values(modes.modes[1], [Vec(50.0, 25.0, 37.5)]).E
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
maxwell_eigenmodes
FiniteElementMethod.maxwell_rhs
FiniteElementMethod.add_term!
FiniteElementMethod.EPS0
FiniteElementMethod.MU0
FiniteElementMethod.C0
```
