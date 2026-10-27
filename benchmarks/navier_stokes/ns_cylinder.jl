# Benchmark (d), spec §11.5: wall-clock of the Navier–Stokes cylinder flow through the PDESystem front end against
# Ferrite's `ns_vs_diffeq.jl` tutorial on the same mesh and settings, and the DFG 2D-1 drag and lift.
#
#     julia --project=benchmarks/navier_stokes -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmarks/navier_stokes benchmarks/navier_stokes/ns_cylinder.jl
#
# Part 1, wall-clock: the tutorial's problem (ν = 1e-3, peak inflow min(1.5t, 1.5), so Re = 100 once ramped, to
# T = 6, abstol = 1e-4, reltol = 1e-5, output every 0.1) on `dfg_channel_grid(h = 0.05, length = 1.1)`, the
# tutorial's shortened channel. Ferrite: the tutorial's Rodas5P on the mass-matrix ODE. FiniteElementMethod: the
# PDESystem of `examples/dfg_cylinder` with the same ramp, discretized to a DAEProblem and solved with DFBDF, the
# pressure and Dirichlet rows out of error control (the tutorial's norm leaves out the Dirichlet dofs). Both
# are timed after a warm-up solve to t = 0.05 in the same process, so neither includes compilation; the
# discretization time of the PDESystem is reported separately.
# The two solutions are compared at t = 1 (end of the ramp) and t = T at 20 points behind the cylinder.
# Part 2, accuracy: 2D-1 (Re = 20) on the full channel at the meshes `BENCHMARK_NS_MESHES`, `h:hc` pairs of
# `dfg_channel_grid` sizes (default 0.04:0.0025,0.02:0.00333,0.01:0.00167).
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "ferrite_ns_vs_diffeq.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "dfg_cylinder", "dfg_cylinder.jl"))
using Printf

const MESHES = [Tuple(parse.(Float64, split(m, ":"))) for m in split(get(ENV, "BENCHMARK_NS_MESHES", "0.04:0.0025,0.02:0.00333,0.01:0.00167"), ",")]
const T_END = parse(Float64, get(ENV, "BENCHMARK_NS_T", "6.0"))

function fem_tutorial(grid; T)
    build_s = @elapsed prob = dfg_problem(grid; tend = T, ν_ = 1.0e-3, Umax_ = 1.5, ramp = min(t, 1))
    abstol = differential_abstol(prob, 1.0e-4)
    solve_s = @elapsed sol = check_solved(solve(prob, DFG_ALG; abstol, reltol = 1.0e-5, saveat = 0.1, tstops = [1.0]))
    return (; sol, build_s, solve_s, ndofs = length(prob.u0), steps = sol.original_sol.stats.naccept + sol.original_sol.stats.nreject)
end

function run()
    grid = dfg_channel_grid(; h = 0.05, length = 1.1)
    ferrite_ns_vs_diffeq(grid; T = 0.05)
    ferrite = ferrite_ns_vs_diffeq(grid; T = T_END)
    fem_tutorial(grid; T = 0.05)
    fem = fem_tutorial(grid; T = T_END)
    # Horizontal velocity of both solutions at points behind the cylinder.
    points = [Ferrite.Vec(x_, y_) for x_ in range(0.4, 1.0; length = 5) for y_ in range(0.05, 0.36; length = 4)]
    ph = Ferrite.PointEvalHandler(grid, points)
    agreement = Dict{String, Any}()
    for τ in (1.0, T_END)
        u_ferrite = first.(Ferrite.evaluate_at_points(ph, ferrite.dh, ferrite.sol(τ), :v))
        u_fem = Ferrite.evaluate_at_points(ph, fem.sol.disc_data.space.dh, fem.sol.original_sol(τ), :u)
        agreement["max_abs_u_difference_t$(τ)"] = maximum(abs, u_ferrite - u_fem)
        agreement["max_abs_u_t$(τ)"] = maximum(abs, u_ferrite)
    end
    println("velocity agreement: ", agreement)
    @printf(
        "tutorial problem to T = %g: Ferrite %.1f s (%d dofs, %d steps, %s); FiniteElementMethod %.1f s (%d dofs, %d steps, %s), discretization %.1f s\n",
        T_END, ferrite.solve_s, ferrite.ndofs, ferrite.steps, ferrite.sol.retcode, fem.solve_s, fem.ndofs, fem.steps, fem.sol.retcode, fem.build_s
    )

    rows = Dict{String, Any}[]
    for (h, hc) in MESHES
        r = dfg_solve(; h, hc)
        c = r.coefficients
        push!(
            rows, Dict{String, Any}(
                "h" => h, "hc" => hc, "ndofs" => r.ndofs, "build_s" => r.build_s, "solve_s" => r.solve_s,
                "cD" => c.cD, "cL" => c.cL, "dp" => c.Δp,
                "cD_relerr" => c.cD / DFG_2D1.cD - 1, "cL_relerr" => c.cL / DFG_2D1.cL - 1, "dp_relerr" => c.Δp / DFG_2D1.Δp - 1,
            )
        )
        @printf(
            "2D-1, h = %g: %d dofs, cD = %.6f (%.2e), cL = %.6f (%.2e), Δp = %.6f (%.2e), solve %.1f s\n",
            h, r.ndofs, c.cD, c.cD / DFG_2D1.cD - 1, c.cL, c.cL / DFG_2D1.cL - 1, c.Δp, c.Δp / DFG_2D1.Δp - 1, r.solve_s
        )
    end
    return write_results(
        "ns_cylinder", Dict(
            "benchmark" => "(d) Navier–Stokes flow past a cylinder: wall-clock against Ferrite's ns_vs_diffeq tutorial, and DFG 2D-1 drag and lift",
            "wallclock" => Dict(
                "description" => "ns_vs_diffeq settings (ν = 1e-3, inflow min(1.5t, 1.5), abstol 1e-4, reltol 1e-5, saveat 0.1) to T = $T_END on dfg_channel_grid(h = 0.05, length = 1.1), Q2/Q1 Taylor–Hood, after a warm-up solve",
                "cells" => Ferrite.getncells(grid),
                "ferrite_solver" => "Rodas5P (AutoFiniteDiff, Ferrite step limiter, UMFPACK) on the mass-matrix ODE",
                "ferrite_s" => ferrite.solve_s, "ferrite_ndofs" => ferrite.ndofs, "ferrite_steps" => ferrite.steps,
                "fem_solver" => "DFBDF on the DAEProblem of the PDESystem",
                "fem_s" => fem.solve_s, "fem_ndofs" => fem.ndofs, "fem_steps" => fem.steps, "fem_discretize_s" => fem.build_s,
                "ratio" => fem.solve_s / ferrite.solve_s,
                "agreement" => agreement,
            ),
            "dfg_2d1" => Dict(
                "reference" => Dict("cD" => DFG_2D1.cD, "cL" => DFG_2D1.cL, "dp" => DFG_2D1.Δp),
                "bands" => Dict("cD" => collect(DFG_2D1_BANDS.cD), "cL" => collect(DFG_2D1_BANDS.cL), "dp" => collect(DFG_2D1_BANDS.Δp)),
                "rows" => rows,
            ),
        )
    )
end

run()
