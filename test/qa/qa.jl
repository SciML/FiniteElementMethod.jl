using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`;
# `Ferrite.facets` (the node ids of a cell's facets) has no public spelling in Ferrite 1.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:FieldWise, :derivative, :facets))),
)
