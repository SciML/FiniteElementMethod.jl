# Discretization

`FEMDiscretization`, `Galerkin` and `SUPG` are exported, as are the PDEBase names a model needs
(`MeshDomain`, `Region`, `boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's
`Lagrange` and reference shapes.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
FiniteElementMethod.SUPG
FiniteElementMethod.assemble_sources!
```

## Multi-field systems, unequal orders and regions

Each dependent variable is one field with its own interpolation, given in `spaces`; fields
without an entry get `Lagrange{RefShape, 1}()`. All fields share one quadrature rule, of order
`2p` for the highest order `p` unless `quadrature` is given. A Taylor–Hood Stokes system pairs
a quadratic velocity with a linear pressure:

```julia
@variables x y u1(..) u2(..) p(..)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))
continuity = 0 ~ Dx(u1(t, x, y)) + Dy(u2(t, x, y))
eqs = [
    Dt(u1(t, x, y)) ~ Δ(u1(t, x, y)) - Dx(p(t, x, y)) + f1,
    Dt(u2(t, x, y)) ~ Δ(u2(t, x, y)) - Dy(p(t, x, y)) + f2,
    continuity,
]
P2, P1 = Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 1}()
disc = FEMDiscretization(Ω, t; spaces = Dict(u1 => P2, u2 => P2, p => P1), pairing = Dict(continuity => p))
```

The continuity equation has no time derivative and no pressure, so `pairing` names the field
it is tested with. The pressure rows are algebraic, and with Dirichlet velocity on the whole
boundary the pressure is fixed only up to a constant: give it a Dirichlet condition on one
small piece (a single facet set). Because the velocity rows of the DAE do not determine the
pressure at the initial time (an index-2 system), start from a velocity that is discretely
divergence free, such as zero, and solve with `DFBDF`.

A field can live on part of the mesh: `regions = Dict(w => Region(Ω, :name))` restricts `w`
to the cell set `name`, through a Ferrite `SubDofHandler`. Outside its region the field has no
dofs, its equation is not assembled, it enters the other equations as zero, and the solution
reads `NaN` there (`sol[w(t, x, y)]` at vertices outside the region, `sol(t, x, y; dv)` at
points outside it). Its boundary conditions must be on pieces whose facets belong to the
region's cells; on the rest of the region's boundary, such as an interface with the other
cells, it has the natural zero-flux condition.

## Streamline-upwind Petrov–Galerkin

`test_space = SUPG(τ)` tests every equation with `v + τ b·∇v` on every cell, where `b` is the
advection velocity read from the equation (the coefficient of `∇u`), or `SUPG(τ; b = [b1, b2])`.
`τ` is a number, an expression, or a function `τ(hK, b)` of the cell diameter `hK` and `b`:

```julia
eq = Dt(u(t, x)) ~ ε * Dx(Dx(u(t, x))) - Dx(u(t, x))
τ(hK, b) = (Pe = abs(b[1]) * hK / (2ε); hK / (2abs(b[1])) * (coth(Pe) - 1 / Pe))
disc = FEMDiscretization(Ω, t; test_space = SUPG(τ))
```

The stabilisation is consistent: the strong residual, time derivative and source included, is
tested with `τ b·∇v`, which needs the shape-function Hessians, so the `CellValues` are built
with them. Limits: linear terms only, `τ` and `b` must be numbers or expressions in the
coordinates, time and `hK` (not parameters of the `PDESystem`), and boundary terms stay
Galerkin.
