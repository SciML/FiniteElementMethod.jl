# Needs the PDEBase mesh-discretization layer, unreleased as of PDEBase 0.1.39.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "shared", "frontend_tests", "steady.jl"))
else
    @testset "steady (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier, StrongForm boundary routing or pairing-ordered mesh coordinates (PDEBase#158)"
    end
end
