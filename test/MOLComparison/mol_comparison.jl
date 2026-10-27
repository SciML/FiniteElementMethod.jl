# Spec §10.4. Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "shared", "molcomparison_tests", "mol_comparison.jl"))
else
    @testset "mol_comparison (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier and StrongForm boundary routing"
    end
end
