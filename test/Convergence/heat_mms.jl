# Spec §10.3. Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "shared", "convergence_tests", "heat_mms.jl"))
else
    @testset "heat_mms (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier and StrongForm boundary routing"
    end
end
