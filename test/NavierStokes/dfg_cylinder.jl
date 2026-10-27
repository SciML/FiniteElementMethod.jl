# The DFG 2D-1 cylinder example (spec §11.5): the PDESystem path against the benchmark's acceptance intervals
# and against the hand-written Ferrite solver on the same mesh.
using FiniteElementMethod, Test
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "..", "examples", "dfg_cylinder", "dfg_cylinder.jl"))
    include(joinpath(@__DIR__, "..", "..", "examples", "dfg_cylinder", "reference.jl"))

    @testset "DFG 2D-1 flow around a cylinder" begin
        h, hc = 0.04, 0.0025
        r = dfg_solve(; h, hc)
        @info "DFG 2D-1" r.ndofs r.build_s r.solve_s r.coefficients
        @test SciMLBase.successful_retcode(r.sol)
        stopped = solve(r.prob, DFG_ALG; abstol = differential_abstol(r.prob, 1.0e-7), reltol = 1.0e-7, maxiters = 3)
        @test_throws "the DFG solve stopped at t = " check_solved(stopped)
        c = r.coefficients
        for k in (:cD, :cL, :Δp)
            lo, hi = DFG_2D1_BANDS[k]
            @test lo <= c[k] <= hi
        end

        # The hand-written steady solution on the same mesh: same dof numbering after FieldWise renumbering.
        dh_ref, x_ref, _ = dfg_reference_steady(dfg_channel_grid(; h, hc); pressure = :strong)
        ref = dfg_coefficients(dh_ref, x_ref; ν = 1.0e-3, Umean = 0.2)
        for k in (:cD, :cL, :Δp)
            @test c[k] ≈ ref[k] rtol = 1.0e-5
        end
    end
else
    @testset "dfg_cylinder (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier, StrongForm boundary routing or pairing-ordered mesh coordinates (PDEBase#158)"
    end
end
