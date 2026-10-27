using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`;
# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
# Tracking: drop NONPUBLIC_PDEBASE once the PDEBase compat floor is the M0 release (public
# `make_pdesys_compatible` and `pair_equations`) and declares `parse_bcs` public; ExplicitImports
# also scans the gated source on PDEBase 0.1.39.
const NONPUBLIC_PDEBASE = (:make_pdesys_compatible, :parse_bcs, :pair_equations)

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (:FieldWise, :derivative, :dirichlet_facetdof_indices, NONPUBLIC_PDEBASE...),
        ),
    ),
    # FEMDiscretization is rendered on the Discretization page, which docs/make.jl copies into
    # docs/src from docs/discretization.md only when the front end is available.
    api_docs_kwargs = (; rendered_ignore = (:FEMDiscretization,)),
    # The user API of a FEM model: PDEBase's mesh vocabulary and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron,
    ),
)
