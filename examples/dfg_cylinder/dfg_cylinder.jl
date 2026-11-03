# The DFG/Schäfer–Turek 2D flow around a cylinder (Schäfer & Turek 1996, case 2D-1, Re = 20), written as a
# strong-form ModelingToolkit PDESystem and discretized with Q2/Q1 Taylor–Hood elements by FEMDiscretization.
#
#     julia --project=benchmarks/navier_stokes examples/dfg_cylinder/dfg_cylinder.jl [h]
#
# The incompressible Navier–Stokes equations are written with scalar velocity components u, v and pressure p.
# Formulation choices, each forced by the front end and checked against the hand-written Ferrite solver of
# `reference.jl` (do-nothing outflow, weak pressure gradient), which it matches on every mesh tried:
#
#   - The pressure gradient is a first-order term, so PDEBase's strong-form analysis keeps it strong (tested with
#     V, not integrated by parts) and the natural outflow condition is ν∂ₙu = 0 rather than the do-nothing
#     ν∂ₙu − p n = 0. The pressure level is then fixed by p = 0 on the outlet. On five meshes from 5k to
#     149k dofs, `dfg_reference_steady` gives the same cD, cL and Δp to seven digits with either formulation
#     (`pressure = :weak` or `:strong`).
#   - The pressure has no equation of its own: the continuity equation is paired with it explicitly, and the
#     system is a DAE of index 2. Its initial state is consistent without index reduction because the flow starts
#     at rest and the inflow is ramped up by 1 − exp(−t²), whose value and rate vanish at t = 0. The zero
#     velocity is discretely divergence free, and p needs no initial condition: it is algebraic, starts at zero,
#     and every residual row vanishes at t = 0. `CheckInit()` therefore only verifies that start. The default
#     initialiser would instead solve the algebraic rows for the algebraic variables, and the continuity rows do
#     not contain p, so its Newton matrix is singular in the pressure columns: there is nothing for it to fix
#     and nothing it could determine.
#   - The pressure, an index-2 algebraic variable, and the Dirichlet rows are left out of DFBDF's error control (an
#     absolute tolerance of 1e8 on them): the local error of an index-2 variable does not shrink with the step, so
#     with it DFBDF cuts the step to nothing. The Dirichlet rows are exact anyway with a W refreshed every step.
#   - 2D-1 is steady: the DAE is integrated until the flow has settled, and the coefficients are evaluated at the
#     final time with the time-derivative term of the force functional included.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using OrdinaryDiffEqBDF: NLNewton
using LinearSolve: UMFPACKFactorization
using ModelingToolkit: t_nounits as t
using SciMLBase: SciMLBase, CheckInit, init, solve!
using Ferrite: Ferrite

include(joinpath(@__DIR__, "mesh.jl"))
include(joinpath(@__DIR__, "forces.jl"))

"DFG 2D-1 reference values (Nabh 1998, John & Matthies 2001), quoted by the FEATFLOW benchmark pages."
const DFG_2D1 = (cD = 5.57953523384, cL = 0.010618948146, Δp = 0.11752016697)
"Acceptance intervals of the benchmark paper (Schäfer & Turek 1996, table 4): the tolerance this example meets."
const DFG_2D1_BANDS = (cD = (5.57, 5.59), cL = (0.0104, 0.011), Δp = (0.1172, 0.1176))

@parameters ν Umax
@variables x y u(..) v(..) p(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

"""
    dfg_problem(grid; tend = 30.0, ν_ = 1e-3, Umax_ = 0.3, ramp = 1 - exp(-t^2), kwargs...)

The `DAEProblem` of the DFG channel `grid` (from `dfg_channel_grid`) with peak inflow `Umax_` times `ramp`, an
expression in `t`; `kwargs` are passed to `FEMDiscretization`.
"""
function dfg_problem(grid; tend = 30.0, ν_ = 1.0e-3, Umax_ = 0.3, ramp = 1 - exp(-t^2), kwargs...)
    Ω = MeshDomain(grid)
    Γin, Γout, Γwall, Γcyl = boundarypiece.(Ω, (:inlet, :outlet, :walls, :cylinder))
    U, V, P = u(t, x, y), v(t, x, y), p(t, x, y)
    eqs = [
        Dt(U) ~ Dx(ν * Dx(U)) + Dy(ν * Dy(U)) - U * Dx(U) - V * Dy(U) - Dx(P),
        Dt(V) ~ Dx(ν * Dx(V)) + Dy(ν * Dy(V)) - U * Dx(V) - V * Dy(V) - Dy(P),
        0 ~ Dx(U) + Dy(V),
    ]
    inflow = ramp * 4Umax * y * (0.41 - y) / 0.41^2
    bcs = [
        u(t, Γin) ~ inflow, v(t, Γin) ~ 0,
        u(t, Γwall) ~ 0, v(t, Γwall) ~ 0,
        u(t, Γcyl) ~ 0, v(t, Γcyl) ~ 0,
        p(t, Γout) ~ 0,
        u(0, x, y) ~ 0, v(0, x, y) ~ 0,
    ]
    domains = [(x, y) ∈ Ω, t ∈ Interval(0.0, tend)]
    @named dfg = PDESystem(eqs, bcs, domains, [t, x, y], [U, V, P], [ν => ν_, Umax => Umax_])
    Q2, Q1 = Lagrange{RefQuadrilateral, 2}(), Lagrange{RefQuadrilateral, 1}()
    disc = FEMDiscretization(
        Ω, t; spaces = Dict(u => Q2, v => Q2, p => Q1), pairing = Dict(eqs[3] => p), natural = [Γout],
        initializealg = CheckInit(), kwargs...
    )
    return DAEProblem(dfg, disc)
end

"Absolute tolerances: `abstol` on the differential variables, none (1e8) on the algebraic ones."
differential_abstol(prob, abstol) = [d ? abstol : 1.0e8 for d in prob.differential_vars]

# UMFPACK by name: LinearSolve 5.18's default sparse LU (supernodal) throws "snlu! requires the same sparsity
# pattern as the analyzed matrix" on these Newton matrices, as it does on Ferrite's own ns_vs_diffeq tutorial.
# Naming it also removes the default's silent fall-back to sparse QR when a Newton matrix is singular, which
# for this index-2 system took tens of GB: a singular matrix now fails the step, and `check_solved` raises.
const DFG_ALG = DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0), linsolve = UMFPACKFactorization())

"Raise unless `sol` succeeded and reached its final time (a callback can stop it early with `Terminated`, a success code); a DAE solve that stopped early must not report coefficients."
function check_solved(sol)
    tend = last(sol.original_sol.prob.tspan)
    SciMLBase.successful_retcode(sol) && sol.t[end] == tend && return sol
    return error(
        "the DFG solve stopped at t = $(sol.t[end]) of $tend with retcode $(sol.retcode); with the pressure in the error control or a singular Newton matrix DFBDF cuts its step to nothing"
    )
end

"""
    dfg_solve(; h = 0.04, hc = 0.0025, tend = 60.0, alg = DFG_ALG, abstol = 1e-7, reltol = 1e-7)

Mesh, discretize and solve 2D-1; returns `(; sol, prob, coefficients, dxdt, ndofs, build_s, solve_s)` with the
coefficients `(cD, cL, Δp)` at `tend`. `dxdt` is the integrator's derivative at `tend`, used in the `∂ₜu` term of
the force functional: with `save_everystep = false` the saved solution has only its two end points, and its
derivative is the slope between them, not the derivative at `tend`.
"""
function dfg_solve(; h = 0.04, hc = 0.0025, tend = 60.0, alg = DFG_ALG, abstol = 1.0e-7, reltol = 1.0e-7)
    grid = dfg_channel_grid(; h, hc)
    build_s = @elapsed prob = dfg_problem(grid; tend)
    integrator = init(prob, alg; abstol = differential_abstol(prob, abstol), reltol, save_everystep = false)
    solve_s = @elapsed solve!(integrator)
    sol = check_solved(SciMLBase.wrap_sol(integrator.sol))
    dh = sol.disc_data.space.dh
    xend = sol.original_sol.u[end]
    dxdt = copy(integrator.du)
    coefficients = dfg_coefficients(dh, xend; ν = 1.0e-3, Umean = 0.2, dxdt)
    return (; sol, prob, coefficients, dxdt, ndofs = length(prob.u0), build_s, solve_s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    h = isempty(ARGS) ? 0.06 : parse(Float64, ARGS[1])
    r = dfg_solve(; h, hc = h / 6)
    c = r.coefficients
    println("2D-1, h = $h: $(r.ndofs) dofs, build $(round(r.build_s; digits = 1)) s, solve $(round(r.solve_s; digits = 1)) s")
    for k in (:cD, :cL, :Δp)
        lo, hi = DFG_2D1_BANDS[k]
        println("  $k = $(c[k]) (reference $(DFG_2D1[k]), relative error $(c[k] / DFG_2D1[k] - 1), band [$lo, $hi]: ", lo <= c[k] <= hi ? "pass" : "FAIL", ")")
    end
end
