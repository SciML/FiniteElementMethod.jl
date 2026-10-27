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
        # Terminated is a success code, so the final time is checked too.
        halt = SciMLBase.DiscreteCallback((u, t, integrator) -> t > 1.0, SciMLBase.terminate!)
        terminated = solve(r.prob, DFG_ALG; abstol = differential_abstol(r.prob, 1.0e-7), reltol = 1.0e-7, callback = halt)
        @test terminated.retcode == SciMLBase.ReturnCode.Terminated
        @test_throws "the DFG solve stopped at t = " check_solved(terminated)
        c = r.coefficients
        # The force functional gets the integrator's final derivative, which vanishes at the steady state and
        # satisfies the DAE there, not the slope between the two saved end points (about |u|/t_end).
        dv = r.prob.differential_vars
        x_end = r.sol.original_sol.u[end]
        @test maximum(abs, r.dxdt[dv]) <= 1.0e-6
        residual = zeros(length(x_end))
        r.prob.f(residual, r.dxdt, x_end, r.prob.p, last(r.prob.tspan))
        @test maximum(abs, residual[dv]) <= 1.0e-9
        steady = dfg_coefficients(r.sol.disc_data.space.dh, x_end; ν = 1.0e-3, Umean = 0.2)
        for k in (:cD, :cL)
            @test c[k] ≈ steady[k] rtol = 1.0e-8
        end
        for k in (:cD, :cL, :Δp)
            lo, hi = DFG_2D1_BANDS[k]
            @test lo <= c[k] <= hi
        end

        # The hand-written steady solution on the same mesh: same dof numbering after FieldWise renumbering.
        grid = dfg_channel_grid(; h, hc)
        @test_throws "Newton's method did not converge in 1 iterations" dfg_reference_steady(grid; maxiter = 1)
        dh_ref, x_ref, _ = dfg_reference_steady(grid; pressure = :strong)
        ref = dfg_coefficients(dh_ref, x_ref; ν = 1.0e-3, Umean = 0.2)
        for k in (:cD, :cL, :Δp)
            @test c[k] ≈ ref[k] rtol = 1.0e-7
        end
    end
else
    @testset "dfg_cylinder (skipped)" begin
        @test_skip "the installed PDEBase lacks the mesh classifier, StrongForm boundary routing or pairing-ordered mesh coordinates (PDEBase#158)"
    end
end
