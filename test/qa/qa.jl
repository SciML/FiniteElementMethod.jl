using SciMLTesting, FiniteElementMethod, Test
using JET

# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1; ForwardDiff
# documents `derivative` as its API but does not declare it `public`.
# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:dirichlet_facetdof_indices, :derivative, :FieldWise))),
)
