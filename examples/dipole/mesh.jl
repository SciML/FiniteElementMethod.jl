using Ferrite: Ferrite
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

"""
    antenna_grid(; outer_radius, huygens_radius, h, h_fine = h, strip = nothing, source_radius = nothing,
                 growth = 0.3)

Graded tetrahedral mesh of the ball of radius `outer_radius` about the origin, built with the Gmsh OCC API.
Physical groups, which become the facet and cell sets of the grid:

  - `"outer"`: the outer sphere (for the absorbing boundary);
  - `"huygens"`: the closed interior sphere of radius `huygens_radius`;
  - with `strip = (; length, width, gap)`: a zero-thickness strip dipole in the plane `y = 0`, along `z`,
    `|x| ≤ width/2`, `|z| ≤ length/2`, as the interior facet sets `"strip"` (the two arms, for `pec`) and
    `"gap"` (the rectangle `|z| ≤ gap/2` between them, for a `LumpedPort`);
  - with `source_radius = a`: the cell set `"source"`, the ball of radius `a`.

The element size is `h_fine` on the strip and in the source ball, grows by `growth` per unit distance from
them (a size `h_fine + growth d` at distance `d`) and is `h` elsewhere.
"""
function antenna_grid(;
        outer_radius, huygens_radius, h, h_fine = h, strip = nothing, source_radius = nothing, growth = 0.3
    )
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("antenna")
        occ = gmsh.model.occ
        tools = [(3, occ.addSphere(0, 0, 0, huygens_radius))]
        source_radius === nothing || push!(tools, (3, occ.addSphere(0, 0, 0, source_radius)))
        if strip !== nothing
            (; length, width, gap) = strip
            arm = (length - gap) / 2
            # Rectangles in the plane z = 0, turned into y = 0 by a quarter turn about x: (x, y, 0) -> (x, 0, y).
            rects = [
                occ.addRectangle(-width / 2, gap / 2, 0, width, arm),
                occ.addRectangle(-width / 2, -length / 2, 0, width, arm),
                occ.addRectangle(-width / 2, -gap / 2, 0, width, gap),
            ]
            occ.rotate([(2, r) for r in rects], 0, 0, 0, 1, 0, 0, π / 2)
            append!(tools, [(2, r) for r in rects])
        end
        occ.fragment([(3, occ.addSphere(0, 0, 0, outer_radius))], tools)
        occ.synchronize()

        # Classify the surfaces by their bounding boxes rather than by OCC's tag numbering.
        surfaces = [tag for (_, tag) in gmsh.model.getEntities(2)]
        box(tag) = gmsh.model.getBoundingBox(2, tag)
        extent(tag) = maximum(abs, box(tag))
        tol = 1.0e-4 * outer_radius
        sphere(r) = filter(t -> abs(extent(t) - r) < 1.0e-2 * r && abs(box(t)[2]) > tol, surfaces)
        gmsh.model.addPhysicalGroup(2, sphere(outer_radius), -1, "outer")
        gmsh.model.addPhysicalGroup(2, sphere(huygens_radius), -1, "huygens")
        fine_curves = Int[]
        if strip !== nothing
            flat = filter(t -> abs(box(t)[2]) < tol && abs(box(t)[5]) < tol, surfaces)
            gap_surface = filter(t -> box(t)[6] < strip.gap / 2 + tol && box(t)[3] > -strip.gap / 2 - tol, flat)
            gmsh.model.addPhysicalGroup(2, setdiff(flat, gap_surface), -1, "strip")
            gmsh.model.addPhysicalGroup(2, gap_surface, -1, "gap")
            for t in flat
                append!(fine_curves, abs.(last.(gmsh.model.getBoundary([(2, t)], false, false))))
            end
        end
        if source_radius !== nothing
            volumes = [tag for (_, tag) in gmsh.model.getEntities(3)]
            ball = filter(t -> maximum(abs, gmsh.model.getBoundingBox(3, t)) < 1.1 * source_radius, volumes)
            gmsh.model.addPhysicalGroup(3, ball, -1, "source")
        end
        gmsh.model.addPhysicalGroup(3, [tag for (_, tag) in gmsh.model.getEntities(3)], -1, "domain")

        fields = Int[]
        # Size h_fine within `near` of the field `distance`, growing linearly to h.
        function refine(distance, near)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_fine)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", near)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", near + (h - h_fine) / growth)
            return push!(fields, threshold)
        end
        if !isempty(fine_curves)
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "CurvesList", unique(fine_curves))
            gmsh.model.mesh.field.setNumber(distance, "Sampling", 400)
            refine(distance, 0.0)
        end
        if source_radius !== nothing
            radius = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(radius, "F", "Sqrt(x^2 + y^2 + z^2)")
            refine(radius, source_radius)
        end
        if isempty(fields)
            constant = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(constant, "F", string(h))
            push!(fields, constant)
        end
        background = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(3)
        return gmsh_grid()
    finally
        Gmsh.finalize()
    end
end

"""
    gmsh_grid()

The `Ferrite.Grid` of the current Gmsh model, as `FerriteGmsh.togrid()` makes it (facet sets from the physical
groups of dimension 2, listing every cell next to each facet; cell sets from those of dimension 3), but finding
the facets through a hash of their nodes, in time linear in the mesh size: `togrid` compares every facet with
every cell (about 17 s at 1.8 × 10⁵ cells, against 0.2 s here).
"""
function gmsh_grid()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    # getNodes lists the nodes by entity, not by tag.
    tags, coordinates = gmsh.model.mesh.getNodes()
    nodes = Vector{Ferrite.Node{3, Float64}}(undef, length(tags))
    for (i, tag) in pairs(tags)
        nodes[tag] = Ferrite.Node(Ferrite.Vec{3}(Tuple(coordinates[(3i - 2):(3i)])))
    end
    cells, element_tags = FerriteGmsh.toelements(3)
    FerriteGmsh.reorient!(cells, nodes)
    cellsets = FerriteGmsh.tocellsets(3, element_tags)
    facets_of = Dict{Vector{Int}, Vector{Ferrite.FacetIndex}}()
    for (i, cell) in pairs(cells), (k, facet) in pairs(Ferrite.facets(cell))
        push!(get!(facets_of, sort!(collect(facet)), Ferrite.FacetIndex[]), Ferrite.FacetIndex(i, k))
    end
    facetsets = Dict{String, Set{Ferrite.FacetIndex}}()
    for (name, triangles) in FerriteGmsh.toboundary(2)
        facetsets[name] = Set(f for t in triangles for f in facets_of[sort!(collect(t))])
    end
    return Ferrite.Grid(cells, nodes; facetsets, cellsets)
end
