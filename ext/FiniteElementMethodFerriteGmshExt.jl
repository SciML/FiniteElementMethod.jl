module FiniteElementMethodFerriteGmshExt

using FiniteElementMethod: FiniteElementMethod, MeshDomain
using FerriteGmsh: FerriteGmsh

const gmsh = FerriteGmsh.gmsh

# The names of the physical groups of dimension `dim - 1` in the file, as FerriteGmsh.togrid reads them.
function _boundary_group_names(path)
    should_finalize = FerriteGmsh.Gmsh.initialize()
    try
        gmsh.open(path)
        facedim = Int(gmsh.model.getDimension()) - 1
        return [gmsh.model.getPhysicalName(d, tag) for (d, tag) in gmsh.model.getPhysicalGroups(facedim)]
    finally
        should_finalize && FerriteGmsh.Gmsh.finalize()
    end
end

function FiniteElementMethod.gmsh_domain(path::AbstractString)
    file = String(path)
    isfile(file) || throw(ArgumentError("gmsh_domain: no such file $(repr(file))"))
    names = _boundary_group_names(file)
    isempty(names) && throw(
        ArgumentError("gmsh_domain: $(repr(file)) has no physical groups of boundary dimension to become boundary pieces; name them with `gmsh.model.addPhysicalGroup`")
    )
    (any(isempty, names) || !allunique(names)) && throw(
        ArgumentError("gmsh_domain: $(repr(file)) has boundary physical groups that are unnamed or share a name, and FerriteGmsh keys boundary pieces by name; every physical group must be named, in a .geo file `Physical Curve(\"name\") = {...}`")
    )
    return MeshDomain(FerriteGmsh.togrid(file))
end

end # module
