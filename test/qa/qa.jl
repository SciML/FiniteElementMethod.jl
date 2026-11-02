using SciMLTesting, FiniteElementMethod, Test
using JET

# ForwardDiff documents `derivative` as its API but does not declare it `public`.
# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`.
run_qa(
    FiniteElementMethod;
    ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:derivative, :FieldWise)))
)
