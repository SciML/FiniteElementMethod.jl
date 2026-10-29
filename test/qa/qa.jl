using SciMLTesting, FiniteElementMethod, Test
using JET

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`;
# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
# ModelingToolkit re-exports these ModelingToolkitBase names; on Julia 1.10 `Base.which` reports
# ModelingToolkitBase as their owner, on 1.11+ ModelingToolkit, the module we depend on.
const LTS_MTK_OWNER = VERSION < v"1.11" ? (:ProblemTypeCtx, :t_nounits) : ()

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (:FieldWise, :derivative, :dirichlet_facetdof_indices),
        ),
        all_explicit_imports_via_owners = (; ignore = LTS_MTK_OWNER),
        all_qualified_accesses_via_owners = (; ignore = LTS_MTK_OWNER),
    ),
    # The user API of a FEM model: PDEBase's mesh vocabulary and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :Region, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron,
    ),
)
