module FiniteElementMethodFerriteGmshExt

using FiniteElementMethod: FiniteElementMethod, MeshDomain
using FerriteGmsh: FerriteGmsh

const gmsh = FerriteGmsh.gmsh

# Opens the file as a new model, checks the names of its boundary physical groups and converts that model.
# FerriteGmsh.togrid(file) would open it a second time. A Gmsh session that the caller started stays as it
# was, with its current model; a session started here is finalized, also when the conversion throws.
function _read_grid(file)
    should_finalize = FerriteGmsh.Gmsh.initialize()
    previous = gmsh.model.getCurrent()
    try
        gmsh.model.add("FiniteElementMethod gmsh_domain") # Gmsh.open loads into an empty current model, so the caller's is never reused
        gmsh.open(file)
        facedim = Int(gmsh.model.getDimension()) - 1
        names = [gmsh.model.getPhysicalName(d, tag) for (d, tag) in gmsh.model.getPhysicalGroups(facedim)]
        isempty(names) && throw(
            ArgumentError("gmsh_domain: $(repr(file)) has no physical groups of boundary dimension to become boundary pieces; name them with `gmsh.model.addPhysicalGroup`")
        )
        (any(isempty, names) || !allunique(names)) && throw(
            ArgumentError("gmsh_domain: $(repr(file)) has boundary physical groups that are unnamed or share a name, and FerriteGmsh keys boundary pieces by name; every physical group must be named, in a .geo file `Physical Curve(\"name\") = {...}`")
        )
        return FerriteGmsh.togrid()
    finally
        if should_finalize
            FerriteGmsh.Gmsh.finalize()
        else
            gmsh.model.remove()
            isempty(previous) || gmsh.model.setCurrent(previous)
        end
    end
end

function FiniteElementMethod.gmsh_domain(path::AbstractString)
    file = String(path)
    isfile(file) || throw(ArgumentError("gmsh_domain: no such file $(repr(file))"))
    return MeshDomain(_read_grid(file))
end

end # module
