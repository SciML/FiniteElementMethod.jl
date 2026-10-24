using FiniteElementMethod, Test
using FiniteElementMethod: FEMOperators, FEMOp, FEMLoad, numeric_dae_problem
using ModelingToolkit: ModelingToolkit
using OrdinaryDiffEqBDF: DFBDF
using SciMLBase: SciMLBase, ReturnCode, remake, solve
using SciMLBase.SymbolicIndexingInterface: getp
using SparseArrays, LinearAlgebra

include(joinpath(@__DIR__, "helpers", "heat2d.jl"))

struct TestMetadata <: SciMLBase.AbstractDiscretizationMetadata{Val(true)}
    metadata::Base.RefValue{Any}
end

const tol = (; abstol = 1.0e-10, reltol = 1.0e-10)

heat_operators(fe; kw...) = FEMOperators(fe.M, fe.K, fe.F, [:u => 1:length(fe.F)]; prescribed = fe.prescribed, kw...)

@testset verbose = true "numeric DAE path" begin
    @testset "numeric path matches a hand-written DAEProblem, N = $((n + 1)^2)" for n in (9, 31, 99)
        N = (n + 1)^2
        fe = heat2d(n)
        ops = heat_operators(fe)
        u0 = zeros(N)
        tbuild = @elapsed prob = numeric_dae_problem(ops, u0, (0.0, 1.0))
        @info "numeric_dae_problem construction" N tbuild
        tsolve = @elapsed sol = solve(prob, DFBDF(); tol...)
        ref = solve(hand_daeproblem(fe.M, fe.K, fe.F, fe.prescribed, u0, (0.0, 1.0)), DFBDF(); tol...)
        @info "solve" N tsolve
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.u[end] - ref.u[end]) <= 1.0e-9

        sys = prob.f.sys
        u = only(SciMLBase.SymbolicIndexingInterface.variable_symbols(sys))
        @test sol[u] == sol.u
        @test sol[u[5]] == getindex.(sol.u, 5)
        @test getp(sys, sys.Kop)(prob.p).A == ops.K
        @test getp(sys, sys.Fop)(prob.p) === ops.load

        n == 9 || continue
        # Parameter remake: swap in a reassembled stiffness with the same pattern.
        K2 = copy(fe.K)
        nonzeros(K2) .*= 2
        ops2 = FEMOperators(fe.M, K2, fe.F, [:u => 1:N]; prescribed = fe.prescribed)
        prob2 = remake(prob; p = [sys.Kop => FEMOp(ops2.K)], du0 = zeros(N))
        @test getp(sys, sys.Kop)(prob2.p).A == ops2.K
        sol2 = solve(prob2, DFBDF(); tol...)
        ref2 = solve(hand_daeproblem(fe.M, 2 .* fe.K, fe.F, fe.prescribed, u0, (0.0, 1.0)), DFBDF(); tol...)
        @test maximum(abs, sol2.u[end] - ref2.u[end]) <= 1.0e-9

        # Numeric u0 remake (du0 left inconsistent; Brown initialisation repairs it).
        u0b = [d in fe.prescribed ? 0.0 : 0.1 for d in 1:N]
        sol3 = solve(remake(prob; u0 = u0b), DFBDF(); tol...)
        ref3 = solve(hand_daeproblem(fe.M, fe.K, fe.F, fe.prescribed, u0b, (0.0, 1.0)), DFBDF(); tol...)
        @test maximum(abs, sol3.u[end] - ref3.u[end]) <= 1.0e-8

        # Upstream blocker 9: SciMLBase remake.jl `fill_vars` keys `idx_to_vsym` by a range for an
        # array unknown, so symbolic u0 maps throw. When this starts passing, flip it to a real test.
        @test_throws KeyError remake(prob; u0 = [u => fill(0.2, N)])
        @test_throws KeyError remake(prob; u0 = [u[5] => 0.3])
    end

    fe = heat2d(9)
    N = length(fe.F)
    ops = heat_operators(fe)
    ref = solve(hand_daeproblem(fe.M, fe.K, fe.F, fe.prescribed, zeros(N), (0.0, 1.0)), DFBDF(); tol...)
    err(sol) = maximum(abs, sol.u[end] - ref.u[end])

    @testset "jac_prototype shares the pattern of M and K" begin
        prob = numeric_dae_problem(ops, zeros(N), (0.0, 1.0))
        J = prob.f.jac_prototype
        @test J.colptr == ops.M.colptr == ops.K.colptr
        @test J.rowval == ops.M.rowval == ops.K.rowval
        @test !any(ops.differential_vars[fe.prescribed])
        @test count(ops.differential_vars) == N - length(fe.prescribed)
        Jv = copy(J)
        prob.f.jac(Jv, prob.du0, prob.u0, prob.p, 0.5, 0.0)
        @test Jv == ops.K + 0.5 * ops.M
        @test all(Jv[d, :] == sparsevec([d], [1.0], N) for d in fe.prescribed)
    end

    @testset "inconsistent u0 on a Dirichlet dof" begin
        u0bad = zeros(N)
        u0bad[fe.prescribed] .= 0.5
        prob = numeric_dae_problem(ops, u0bad, (0.0, 1.0))
        sol = solve(prob, DFBDF())
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.u[1][fe.prescribed]) == 0
        @test err(sol) < 1.0e-3
        # Expected upstream behaviour: with every dof marked differential, Brown initialisation may
        # change only du, so the violated row u_d - g_d = 0.5 cannot be repaired.
        sol_nomask = solve(remake(prob; differential_vars = trues(N)), DFBDF())
        @test sol_nomask.retcode == ReturnCode.InitialFailure
    end

    @testset "inconsistent du0 = 0 with the sparse Jacobian" begin
        consistent = solve(numeric_dae_problem(ops, zeros(N), (0.0, 1.0)), DFBDF())
        prob = numeric_dae_problem(ops, zeros(N), (0.0, 1.0); du0 = zeros(N))
        @test prob.f.jac_prototype isa SparseMatrixCSC
        sol = solve(prob, DFBDF())
        @test SciMLBase.successful_retcode(sol)
        @test isapprox(err(sol), err(consistent); rtol = 1.0e-2)
    end

    @testset "time-dependent Dirichlet data g = sin(t)" begin
        opsg = heat_operators(fe; dirichlet_values = (out, t) -> fill!(out, sin(t)))
        @test opsg.load.timedep
        sol = solve(numeric_dae_problem(opsg, zeros(N), (0.0, 1.0)), DFBDF())
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol.u[end][fe.prescribed] .- sin(sol.t[end])) <= 1.0e-12
    end

    @testset "algebraic block: 0 = Δv + exp(-t) sin(πx)" begin
        cp = coupled2d(9)
        opsc = FEMOperators(cp.M, cp.K, cp.load!, cp.fields; prescribed = cp.prescribed)
        ru, rv = last.(cp.fields)
        @test !any(opsc.differential_vars[rv])
        @test opsc.differential_vars[ru] == [d ∉ cp.prescribed for d in ru]
        prob = numeric_dae_problem(opsc, zeros(length(rv) + length(ru)), (0.0, 1.0))
        sys = prob.f.sys
        us = SciMLBase.SymbolicIndexingInterface.variable_symbols(sys)
        @test length(us) == 2
        sol = solve(prob, DFBDF())
        @test SciMLBase.successful_retcode(sol)
        # The algebraic block holds at the final time.
        Kvv, Kvu = opsc.K[rv, rv], opsc.K[rv, ru]
        F1 = cp.load!(zeros(length(sol.u[end])), nothing, sol.t[end])
        F1[cp.prescribed] .= 0
        @test norm(Kvv * sol.u[end][rv] + Kvu * sol.u[end][ru] - F1[rv]) < 1.0e-8
        @test sol[us[2]][end] == sol.u[end][rv]
    end

    @testset "discretization metadata becomes problem_type" begin
        md = TestMetadata(Ref{Any}(nothing))
        prob = numeric_dae_problem(ops, zeros(N), (0.0, 1.0), md)
        @test prob.problem_type === md
        @test md.metadata[] === prob.f.sys
        @test ModelingToolkit.getmetadata(prob.f.sys, ModelingToolkit.ProblemTypeCtx, nothing) === md
        @test remake(prob; differential_vars = trues(N)).problem_type === md
    end

    @testset "non-finite initial condition names the coordinate" begin
        u0 = [sin(hypot(x...)) / hypot(x...) for x in fe.coords]
        bad = findfirst(!isfinite, u0)
        @test fe.coords[bad] == (0.0, 0.0)
        e = try
            numeric_dae_problem(ops, u0, (0.0, 1.0); coordinate = d -> fe.coords[d])
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("(0.0, 0.0)", e.msg)
        @test occursin("not finite", e.msg)
    end
end
