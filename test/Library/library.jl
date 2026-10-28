# Spec §10.5. Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "shared", "library_tests", "library.jl"))
else
    @testset "library (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier and StrongForm boundary routing"
    end
end
