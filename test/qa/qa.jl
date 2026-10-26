using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`;
# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
# Tracking: drop NONPUBLIC_PDEBASE once PDEBase declares StrongForm's `route_natural`,
# `essential_condition` and `pair_equations` public.
const NONPUBLIC_PDEBASE = (:route_natural, :essential_condition, :pair_equations)

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (:FieldWise, :derivative, :dirichlet_facetdof_indices, NONPUBLIC_PDEBASE...),
        ),
    ),
    # The user API of a FEM model: PDEBase's mesh vocabulary and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron,
    ),
)
