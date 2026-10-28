# Getting started

This tutorial solves the 2D heat equation on an L-shaped Gmsh mesh with a time-dependent
Dirichlet condition, an insulated (Neumann) wall and a Robin outlet. The code is the API
example test of the package, `test/shared/frontend_tests/api_example.jl`, so it runs as
written. It needs FiniteElementMethod.jl, ModelingToolkit.jl, DomainSets.jl, FerriteGmsh.jl,
Gmsh.jl and OrdinaryDiffEqBDF.jl.

!!! warning "Coordinates come from the domain pairing, not from the mesh"
    A mesh carries no coordinate names. `(x, y) ∈ Ω` below is the only thing that names them:
    `x` is mesh coordinate 1 and `y` is mesh coordinate 2, and the same order is used by
    `sol[x]`, `sol[y]` and `sol(t, x, y)`. See [Coordinate binding](@ref).

## The mesh

The L-shape is the square ``[0, 2]^2`` with the upper-right quadrant ``(1, 2]^2`` removed. It
is built with the Gmsh API; each Gmsh physical group of dimension 1 becomes a Ferrite facet
set, and each facet set is a boundary piece.

```julia
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

function lshape_grid(; h = 0.25)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("lshape")
        corners = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        points = [gmsh.model.geo.addPoint(x, y, 0, h) for (x, y) in corners]
        lines = [gmsh.model.geo.addLine(points[i], points[mod1(i + 1, 6)]) for i in 1:6]
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [lines[6]], -1, "inlet")
        gmsh.model.addPhysicalGroup(1, [lines[2]], -1, "outlet")
        gmsh.model.addPhysicalGroup(1, lines[[1, 3, 4, 5]], -1, "wall")
        gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")
        gmsh.model.mesh.generate(2)
        return FerriteGmsh.togrid()
    finally
        Gmsh.finalize()
    end
end
```

The inlet is the edge ``x = 0``, the outlet is the edge ``x = 2`` and the wall is every other
edge. A mesh stored in a file is read the same way, with `FerriteGmsh.togrid("plate.msh")`.

## The model

```julia
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using ModelingToolkit: t_nounits as t

@parameters k α
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

grid = lshape_grid()
Ω = MeshDomain(grid)
Γ_in, Γ_wall, Γ_out = boundarypiece.(Ω, (:inlet, :wall, :outlet))
Dn_wall, Dn_out = NormalDerivative(Γ_wall), NormalDerivative(Γ_out)

eqs = [Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y))) + 1]
bcs = [
    u(t, Γ_in) ~ sin(t),
    Dn_wall(u(t, Γ_wall)) ~ 0,
    k * Dn_out(u(t, Γ_out)) + α * u(t, Γ_out) ~ α * 0.3,
    u(0, x, y) ~ 0,
]
domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)]
@named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y)], [k => 2.0, α => 5.0])
```

Line by line:

  - `@variables x y u(..)` declares the coordinates and the dependent variable as for
    MethodOfLines. `x` and `y` get their meaning only from the domain pairing further down.
  - `MeshDomain(grid)` wraps the `Ferrite.Grid` as a domain. It holds the mesh and nothing
    else: no coordinate names and no geometry beyond the mesh's facet sets.
  - `boundarypiece.(Ω, names)` returns the boundary pieces named by the facet sets `"inlet"`,
    `"wall"` and `"outlet"`. A name that is not a facet set of the mesh is an error listing the
    valid names. `boundarypieces(Ω)` returns every piece.
  - `NormalDerivative(Γ)` is the derivative along the outward unit normal of the piece `Γ`. It
    is an operator like `Differential(x)`.
  - The equation is written in strong form, as for MethodOfLines. The package derives the
    Galerkin weak form from it; you do not write test functions or integrals.
  - `u(t, Γ_in) ~ sin(t)` is a time-dependent Dirichlet condition. `u(t, Γ)` is the trace of `u`
    on `Γ`: the piece replaces the spatial arguments and time stays in its declared position
    (see [Trace syntax](@ref)).
  - `Dn_wall(u(t, Γ_wall)) ~ 0` is a Neumann condition: the normal derivative on the wall is
    zero, so the wall is insulated. A piece with no condition at all would get the same
    zero-flux condition, with a warning.
  - `k * Dn_out(u(t, Γ_out)) + α * u(t, Γ_out) ~ α * 0.3` is a Robin condition. The conormal
    coefficient `k` of the flux `k ∇u` is written explicitly.
  - `u(0, x, y) ~ 0` is the initial condition, written as for MethodOfLines.
  - `(x, y) ∈ Ω` binds `x` to mesh coordinate 1 and `y` to mesh coordinate 2. The tuple must
    have one entry per mesh coordinate.
  - Parameters need numeric values in the `PDESystem`. They are substituted when the
    operators are assembled, so `remake` with new parameter values does not reassemble the
    operators.

## Discretize and solve

```julia
disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 1}()))
prob = DAEProblem(pdesys, disc)
sol = solve(prob, DFBDF())
```

  - `FEMDiscretization(Ω, t; spaces)` chooses the finite-element space of each dependent
    variable, here continuous piecewise-linear Lagrange elements on triangles. `spaces` may be
    omitted; the default is linear Lagrange on the mesh's cell shape.
  - `DAEProblem(pdesys, disc)` (or `discretize(pdesys, disc)`) assembles the mass matrix `M`,
    the stiffness matrix `K` and the load `F(t)` with Ferrite and returns the `DAEProblem`
    `M du + K u = F(t)`, with one array unknown per field. The Dirichlet values are algebraic
    rows `u_d = g_d(t)`. Nothing calls `mtkcompile`, and anything outside the supported classes
    raises an error here, before any solver runs.
  - Any DAE solver applies. `DFBDF()` is a good default.

!!! note "Exact Dirichlet rows"
    With the default solver settings the Dirichlet rows `u(t, Γ) ~ g(t)` hold to about the
    solver tolerance, not to round-off. OrdinaryDiffEq's BDF methods reuse the Newton
    Jacobian across step-size changes and rescale the Newton step ("relaxed Newton"), which
    leaves the algebraic rows off by about the tolerance. To hold them to round-off, refresh
    the Jacobian on every step-size change:

    ```julia
    using OrdinaryDiffEqBDF: NLNewton
    sol = solve(prob, DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0)))
    ```

    In the API example test the default solve keeps `sol[u(t, Γ_in)]` within `1e-5` of
    `sin.(sol.t)`, and the solve above keeps it within `1e-12`.

## Use the solution

```julia
sol[u(t, x, y)]                    # nt × nnodes: one column per mesh node
sol[x], sol[y]                     # node coordinates; column i is at (sol[x][i], sol[y][i])
sol(0.5, 0.2, 0.3; dv = u(t, x, y))  # value at t = 0.5, (x, y) = (0.2, 0.3)
sol(0.5, 1.5, 1.5; dv = u(t, x, y))  # NaN: the point is in the missing quadrant
sol[u(t, Γ_in)]                    # trace on the inlet, one column per inlet node
sol[Dn_out(u(t, Γ_out))]           # facet-mean normal derivative on the outlet
write_vtk(sol, "heat")             # heat_0001.vtu, … and heat.pvd for ParaView
```

These shapes differ from a MethodOfLines solution; [Solution shapes that differ from MethodOfLines](@ref)
lists every difference.
