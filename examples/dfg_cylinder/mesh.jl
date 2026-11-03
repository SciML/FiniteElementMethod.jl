using Ferrite: Ferrite
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

"""
    dfg_channel_grid(; h = 0.04, hc = h / 6, length = 2.2, order = 2)

Quadrilateral mesh of the DFG/Schäfer–Turek 2D channel `[0, length] × [0, 0.41]` minus the cylinder of
radius 0.05 centred at `(0.2, 0.2)`, built with the Gmsh API and converted with `FerriteGmsh.togrid`.
The element size is `hc` on the cylinder, grows linearly to `h` at distance 0.25 from it and is `h`
elsewhere; halving both is one uniform refinement. `order = 2` gives `QuadraticQuadrilateral` cells whose
edge midpoints lie on the cylinder, so the cylinder is curved for an isoparametric Q2 field; `order = 1`
gives straight-edged `Quadrilateral`s. `length = 1.1` is the shortened channel of Ferrite's
`ns_vs_diffeq.jl` tutorial.

Facet sets: `"inlet"` (`x = 0`), `"outlet"` (`x = length`), `"walls"` (`y = 0` and `y = 0.41`) and
`"cylinder"`.
"""
function dfg_channel_grid(; h = 0.04, hc = h / 6, length = 2.2, order = 2)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dfg")
        rect = gmsh.model.occ.addRectangle(0, 0, 0, length, 0.41)
        disk = gmsh.model.occ.addDisk(0.2, 0.2, 0, 0.05, 0.05)
        gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        curves = [tag for (_, tag) in gmsh.model.getEntities(1)]
        # Classify the boundary curves by their bounding boxes rather than by OCC's tag numbering.
        on(tag, test) = test(gmsh.model.getBoundingBox(1, tag)...)
        tol = 1.0e-6
        inlet = filter(c -> on(c, (x0, y0, z0, x1, y1, z1) -> x1 < tol), curves)
        outlet = filter(c -> on(c, (x0, y0, z0, x1, y1, z1) -> x0 > length - tol), curves)
        walls = filter(c -> on(c, (x0, y0, z0, x1, y1, z1) -> y1 < tol || y0 > 0.41 - tol), curves)
        cylinder = setdiff(curves, inlet, outlet, walls)
        gmsh.model.addPhysicalGroup(1, inlet, -1, "inlet")
        gmsh.model.addPhysicalGroup(1, outlet, -1, "outlet")
        gmsh.model.addPhysicalGroup(1, walls, -1, "walls")
        gmsh.model.addPhysicalGroup(1, cylinder, -1, "cylinder")
        gmsh.model.addPhysicalGroup(2, [tag for (_, tag) in surfaces], -1, "fluid")

        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", cylinder)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        # Subdivision halves every edge, so the triangulation is made at twice the final size.
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", 2hc)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 2h)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.25)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # Frontal-Delaunay for quads and Blossom recombination; the subdivision step splits any triangle
        # Blossom leaves into three quadrilaterals, so the mesh is all quadrilaterals at every size.
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        grid = FerriteGmsh.togrid()
        celltype = Ferrite.getcelltype(grid)
        expected = order == 2 ? Ferrite.QuadraticQuadrilateral : Ferrite.Quadrilateral
        celltype === expected || error("Gmsh left non-quadrilateral cells; the cell type is $celltype")
        return grid
    finally
        Gmsh.finalize()
    end
end
