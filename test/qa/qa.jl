using SciMLTesting, FiniteElementMethod, Test
using JET

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative`, and `Dual`, `Partials`, `value` and `partials` for custom rules, as its API
# but declares none of them `public`;
# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
# `Ferrite.vertices`, `reference_facets` and `getrefdim` (the corner nodes of a cell, of its facets, and the
# dimension of its reference shape) identify the facets of a submesh and have no public spelling in Ferrite 1.
# ModelingToolkit re-exports these ModelingToolkitBase names; on Julia 1.10 `Base.which` reports
# ModelingToolkitBase as their owner, on 1.11+ ModelingToolkit, the module we depend on.
const LTS_MTK_OWNER = VERSION < v"1.11" ? (:ProblemTypeCtx, :t_nounits) : ()

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    # `curl` gets a method for symbolic vectors, so that it is the one `curl` with Ferrite loaded too.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid, FiniteElementMethod.Ferrite.curl])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :FieldWise, :derivative, :Dual, :Partials, :value, :partials, :dirichlet_facetdof_indices,
                :vertices, :reference_facets, :getrefdim,
            ),
        ),
        # `curl` is owned by Tensors, which Ferrite re-exports and which is not a dependency.
        all_explicit_imports_via_owners = (; ignore = (LTS_MTK_OWNER..., :curl)),
        all_qualified_accesses_via_owners = (; ignore = (LTS_MTK_OWNER..., :curl)),
    ),
    # The user API of a FEM model: PDEBase's mesh vocabulary and weighted integration by parts, and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :Region, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron, :curl, :Weighted,
    ),
)
