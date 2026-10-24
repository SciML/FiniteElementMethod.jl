using SciMLTesting, FiniteElementMethod

# ForwardDiff documents `derivative` as its API but does not declare it `public`.
run_qa(FiniteElementMethod; ei_kwargs = (; all_qualified_accesses_are_public = (; ignore = (:derivative,))))
