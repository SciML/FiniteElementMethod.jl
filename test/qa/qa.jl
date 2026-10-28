using SciMLTesting, FiniteElementMethod, Test
using JET

# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:dirichlet_facetdof_indices,))),
)
