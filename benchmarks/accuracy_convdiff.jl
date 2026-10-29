# Benchmark (a), spec §11.5: accuracy per degree of freedom against MethodOfLines on PDESystemLibrary's
# `convdiff2`, ∂ₜf = k ∂²f/∂z² + v ∂f/∂z on 0 ≤ z ≤ 5 with k = 1, v = −1, f(t, 0) = 0, f(t, 5) = 1 and f(0, z) = 0
# inside. The library's reference series is NaN everywhere (spec §11.2, L2), so the error is taken at t = 30 against
# the steady solution (1 − e^{−vz/k}) / (1 − e^{−vL/k}); the slowest mode decays like e^{−0.645 t}, so f is steady
# to below 1e-8 there. MethodOfLines discretizes the library system itself; FiniteElementMethod gets the same
# equation on a P1 Line mesh with the same nodes, written with boundary pieces instead of endpoint arguments.
include(joinpath(@__DIR__, "common.jl"))
using FiniteElementMethod, MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, PDESystemLibrary, Printf
using Ferrite: Ferrite
using OrdinaryDiffEqBDF: NLNewton
using ModelingToolkit: t_nounits as t
using Symbolics: Num

const K, V, L, TEND = 1.0, -1.0, 5.0, 30.0
const SIZES = parse.(Int, split(get(ENV, "BENCHMARK_CONVDIFF_SIZES", "16,32,64,128"), ","))
const TOL = (; reltol = 1.0e-9, abstol = 1.0e-11)
steady(z) = (1 - exp(-V * z / K)) / (1 - exp(-V * L / K))

function mol_error(n)
    sys = PDESystemLibrary.all_systems[findfirst(s -> string(s.name) == "convdiff2", PDESystemLibrary.all_systems)]
    z, tt = sys.ivs[2], sys.ivs[1]
    prob = discretize(sys, MOLFiniteDifference([z => L / n], tt))
    alg = prob isa SciMLBase.DAEProblem ? DFBDF() : FBDF()
    sol = solve(prob, alg; TOL...)
    SciMLBase.successful_retcode(sol) || error("MethodOfLines convdiff2 n = $n: $(sol.retcode)")
    values = sol[sys.dvs[1]][end, :]
    solver = "$(nameof(typeof(alg)))() on the $(nameof(typeof(prob))) from MethodOfLines; $(tolerances())"
    return maximum(abs, values .- steady.(range(0, L, n + 1))), n + 1, solver
end

tolerances() = "reltol = $(TOL.reltol), abstol = $(TOL.abstol)"
const FEM_SOLVER = "DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0)) on the DAEProblem from FEMDiscretization; $(tolerances())"

@variables z f(..)
@parameters k v
const Dt, Dz = Differential(t), Differential(z)

function fem_error(n)
    Ω = MeshDomain(FiniteElementMethod.rectangle_grid(Ferrite.Line, (n,); lo = (0.0,), hi = (L,)))
    lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
    eq = Dt(f(t, z)) ~ k * Dz(Dz(f(t, z))) + v * Dz(f(t, z))
    bcs = [f(t, lo) ~ 0.0, f(t, hi) ~ 1.0, f(0, z) ~ 0.0]
    @named pdesys = PDESystem([eq], bcs, [z ∈ Ω, t ∈ Interval(0.0, TEND)], [t, z], [f(t, z)], [k => K, v => V])
    prob = discretize(pdesys, FEMDiscretization(Ω, t))
    # A fresh W at every step keeps the algebraic Dirichlet rows exact (test/Frontend/api_example.jl).
    sol = solve(prob, DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0)); TOL...)
    SciMLBase.successful_retcode(sol) || error("FEM convdiff2 n = $n: $(sol.retcode)")
    values = sol[f(t, z)][end, :]
    nodes = sol[Num(only(sol.disc_data.x̄))]
    return maximum(abs, values .- steady.(nodes)), length(values), FEM_SOLVER
end

rates(errors) = [NaN; [log2(errors[i - 1] / errors[i]) for i in 2:length(errors)]]

function run()
    rows = Dict{String, Any}[]
    fem, mol = [fem_error(n) for n in SIZES], [mol_error(n) for n in SIZES]
    fem_rates, mol_rates = rates(first.(fem)), rates(first.(mol))
    for (i, n) in enumerate(SIZES)
        push!(
            rows, Dict(
                "cells" => n, "fem_ndofs" => fem[i][2], "mol_ndofs" => mol[i][2],
                "fem_error" => fem[i][1], "mol_error" => mol[i][1],
                "fem_rate" => fem_rates[i], "mol_rate" => mol_rates[i],
            )
        )
        @printf(
            "n = %4d  FEM P1 %4d dofs  error %.3e  rate %5.2f   |   MethodOfLines %4d points  error %.3e  rate %5.2f\n",
            n, fem[i][2], fem[i][1], fem_rates[i], mol[i][2], mol[i][1], mol_rates[i]
        )
    end
    return write_results(
        "accuracy_convdiff", Dict(
            "benchmark" => "(a) accuracy per degree of freedom, convdiff2",
            "description" => "max nodal error at t = 30 against the steady solution; FEM P1 Galerkin vs MethodOfLines (default upwind advection), same nodes",
            "fem_solver" => only(unique(last.(fem))),
            "mol_solver" => join(unique(last.(mol)), "; "),
            "rows" => rows,
        )
    )
end

run()
