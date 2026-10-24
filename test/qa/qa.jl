using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative` as its API but does not declare it `public`.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:FieldWise, :derivative)))
)
