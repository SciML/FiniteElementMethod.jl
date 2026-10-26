using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`;
# `Ferrite.facets` (the node ids of a cell's facets) has no public spelling in Ferrite 1.
# `PDEBase.StrongForm`, `make_pdesys_compatible` and StrongForm's boundary routing
# (`route_natural`, `essential_condition`, `pair_equations`) are not yet declared public upstream.
const NONPUBLIC_PDEBASE = (:make_pdesys_compatible, :route_natural, :essential_condition, :pair_equations)

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (; ignore = (:FieldWise, :derivative, :facets, NONPUBLIC_PDEBASE...)),
        all_explicit_imports_are_public = (; ignore = (:StrongForm,)),
    ),
    # The user API of a FEM model: PDEBase's mesh vocabulary and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron,
    ),
)
