using Ferrite: Ferrite
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

"""
    lshape_grid(; h = 0.25)

Triangle mesh of the L-shape `[0, 2]² \\ (1, 2]²` (the upper-right quadrant is missing), built
with the Gmsh API and converted with `FerriteGmsh.togrid`. Physical groups of dimension 1:
`"inlet"` (`x = 0`), `"outlet"` (`x = 2`) and `"wall"` (every other edge); dimension 2:
`"domain"`; dimension 0: `"corner"` at the re-entrant corner `(1, 1)`, which FerriteGmsh drops.
"""
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
        gmsh.model.addPhysicalGroup(0, [points[4]], -1, "corner")
        gmsh.model.mesh.generate(2)
        return FerriteGmsh.togrid()
    finally
        Gmsh.finalize()
    end
end
