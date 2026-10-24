using SciMLTesting, FiniteElementMethod

# `Ferrite.facets` (the node ids of a cell's facets) has no public spelling in Ferrite 1.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:facets,))),
)
