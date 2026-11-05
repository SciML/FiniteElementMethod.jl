# Accuracy of the sparse direct solvers on the Newton matrix of the axisymmetric swirl problem.
#
#     julia --project=docs examples/axisym_swirl/linear_residual.jl extract c t_stop file
#     julia --project=docs examples/axisym_swirl/linear_residual.jl solve file umfpack|klu
#
# `extract` steps the integrator to `t_stop`, then writes the Newton matrix W = dF/du + γ dF/d(du) at that state
# (γ is the one DFBDF built its own W with) and the residual right-hand side of the next step to `file`. `solve`
# factorizes the saved matrix once with the named solver, in a process of its own so that the peak memory is that
# solver's, and prints the factorization time and the relative residual ‖W x - b‖ / ‖b‖.
using LinearAlgebra, SparseArrays, Serialization, Printf
using LinearSolve: LinearSolve, LinearProblem, KLUFactorization, init
using SciMLBase: SciMLBase, step!, solve!

if ARGS[1] == "extract"
    include(joinpath(@__DIR__, "axisym_swirl.jl"))
    c, t_stop, file = parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), ARGS[4]
    prob = axisym_problem(axisym_grid(; c, tau_end = 1.0e-2); tend = 0.99)
    integ = SciMLBase.init(prob, AXISYM_ALG; reltol = 1.0e-8, save_everystep = false)
    while integ.t < t_stop
        step!(integ)
    end
    stats = integ.stats
    @printf("t = %.4g, dt = %.3g, steps %d, Newton iterations %d, Newton failures %d\n", integ.t, integ.dt, stats.naccept, stats.nnonliniter, stats.nnonlinconvfail)
    γ = integ.cache.nlsolver.cache.W_γdt
    u, du, p, tt = copy(integ.u), copy(integ.du), integ.p, integ.t
    W = copy(prob.f.jac_prototype)
    prob.f.jac(W, du, u, p, γ, tt)
    residual = zeros(length(u))
    prob.f.f(residual, du, u, p, tt + integ.dt)
    serialize(file, (; W, b = -residual, γ, t = tt, naccept = stats.naccept, newton = stats.nnonliniter))
    @printf("saved %s: %d unknowns, %d nonzeros\n", file, size(W, 1), nnz(W))
else
    W, b = let m = deserialize(ARGS[2])
        m.W, m.b
    end
    solver = ARGS[3] == "klu" ? KLUFactorization() : LinearSolve.UMFPACKFactorization()
    cache = init(LinearProblem(W, b), solver)
    factor_s = @elapsed x = solve!(cache).u
    @printf(
        "%s: %d unknowns, factor+solve %.1f s, relative residual %.3e, peak memory %.2f GiB\n",
        ARGS[3], size(W, 1), factor_s, norm(W * x - b) / norm(b), Sys.maxrss() / 2^30
    )
end
