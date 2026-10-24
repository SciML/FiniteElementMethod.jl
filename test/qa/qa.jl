using SciMLTesting, FiniteElementMethod

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:FieldWise,)))
)
