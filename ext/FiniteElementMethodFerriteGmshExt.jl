module FiniteElementMethodFerriteGmshExt

using FiniteElementMethod: FiniteElementMethod, MeshDomain
using Ferrite: Ferrite
using FerriteGmsh: FerriteGmsh

function FiniteElementMethod.gmsh_domain(path::AbstractString)
    isfile(path) || throw(ArgumentError("gmsh_domain: no such file $(repr(path))"))
    grid = FerriteGmsh.togrid(path)
    isempty(grid.facetsets) && throw(
        ArgumentError("gmsh_domain: $(repr(path)) has no physical groups of dimension $(Ferrite.getspatialdim(grid) - 1) to become boundary pieces; name them with `gmsh.model.addPhysicalGroup`")
    )
    return MeshDomain(grid)
end

end # module
