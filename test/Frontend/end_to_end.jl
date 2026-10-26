# Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "shared", "frontend_tests", "end_to_end.jl"))
else
    @testset "end_to_end (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier and StrongForm boundary routing"
    end
end
