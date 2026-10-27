# Discretization

`FEMDiscretization` is exported, as are the PDEBase names a model needs (`MeshDomain`,
`boundarypiece`, `boundarypieces`, `NormalDerivative`) and Ferrite's `Lagrange` and reference
shapes.

## Curved meshes

The geometry of every cell comes from the mesh: the space, the assembly (facet normals
included) and the solution interface map each cell with the geometric interpolation of its
Ferrite cell type. A second-order mesh therefore gives curved cells with no option to set.
To keep P2 at its full rate on a curved domain, mesh it at order 2 so the mid-side nodes lie
on the true boundary:

```julia
using FerriteGmsh, Gmsh

Gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
outer, inner = gmsh.model.occ.addCircle(0, 0, 0, 1.0), gmsh.model.occ.addCircle(0, 0, 0, 0.5)
surface = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([outer]), gmsh.model.occ.addCurveLoop([inner])])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(1, [inner], -1, "inner")
gmsh.model.addPhysicalGroup(1, [outer], -1, "outer")
gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)         # QuadraticTriangle cells, mid-side nodes on the circles
grid = FerriteGmsh.togrid()
Gmsh.finalize()

Ω = MeshDomain(grid)
Γi, Γo = boundarypiece.(Ω, (:inner, :outer))
disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
```

Conditions on the curved pieces are written as on any other piece, with `NormalDerivative(Γ)`
the derivative along the outward normal of the curved facets. With straight-sided cells
(`setOrder(1)`) the boundary is a polygon and P2 drops from L2 rate 3 to 2 once boundary data
or normals of the true curve enter; `test/shared/frontend_tests/curved.jl` measures both on
this annulus. Use a mesh generator that does not leave nodes outside every cell, such as the
centre point of Gmsh's built-in-kernel circle arcs: `sol[u]` is `NaN` at such nodes.

```@docs
FEMDiscretization
FiniteElementMethod.SciMLBase.DAEProblem(::FiniteElementMethod.ModelingToolkit.PDESystem, ::FEMDiscretization)
FiniteElementMethod.AbstractTestSpace
FiniteElementMethod.Galerkin
FiniteElementMethod.assemble_sources!
```
