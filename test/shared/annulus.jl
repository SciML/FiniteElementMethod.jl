using Ferrite: Ferrite
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

"""
    annulus_grid(h; order = 2, r = 0.5, R = 1.0)

Triangle mesh of the annulus `r ≤ |x| ≤ R` with mesh size `h`, built with the Gmsh API from
OpenCASCADE circles and converted with `FerriteGmsh.togrid`. With `order = 2` Gmsh places the mid-side
nodes on the circles and the grid has `QuadraticTriangle` cells; with `order = 1` the same
vertices make straight-sided `Triangle`s. Physical groups of dimension 1: `"inner"` (`|x| = r`)
and `"outer"` (`|x| = R`); dimension 2: `"domain"`.
"""
function annulus_grid(h; order = 2, r = 0.5, R = 1.0)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("annulus")
        # OpenCASCADE full circles: Gmsh's built-in kernel needs a centre point for its arcs, which
        # FerriteGmsh would keep as a node outside every cell.
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        outer = gmsh.model.occ.addCircle(0, 0, 0, R)
        inner = gmsh.model.occ.addCircle(0, 0, 0, r)
        surface = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([outer]), gmsh.model.occ.addCurveLoop([inner])])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(1, [inner], -1, "inner")
        gmsh.model.addPhysicalGroup(1, [outer], -1, "outer")
        gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        return FerriteGmsh.togrid()
    finally
        Gmsh.finalize()
    end
end
