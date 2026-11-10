# Axisymmetric Navier–Stokes with swirl in the half plane (r, z): a manufactured vortex core that collapses
# self-similarly toward the origin, solved with Taylor–Hood P2/P1 triangles on a mesh graded toward the origin.
#
#     julia --project=docs examples/axisym_swirl/axisym_swirl.jl [c] [tau_end]
#
# This is our own discretization driven by our own manufactured field (`manufactured.jl`); it is not a
# reproduction of any blowup result. The pieces, each explained on the documentation page:
#
#   - Each momentum equation is written in the textbook strong form of the (r, z) system and integrated by parts
#     with `Weighted(r)`, the axisymmetric measure r dr dz. The continuity equation is written already weighted,
#     `0 ~ Dr(r ur) + r Dz(uz)`, and paired with the pressure.
#   - u_r = u_theta = 0 on the axis; u_z carries the natural condition there (`natural = [Γaxis]`). The far
#     boundaries carry the exact velocity and the exact pressure is pinned on `outer`.
#   - The mesh comes from `mesh.geo` through the Gmsh API; the size is c times the distance to the origin,
#     between hmin = c sqrt(tau_end) / 2 and hmax = 5c.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using OrdinaryDiffEqBDF: NLNewton
using LinearSolve: UMFPACKFactorization
using PDEBase.StrongForm: Weighted
using SciMLBase: SciMLBase, CheckInit, solve
using Ferrite: Ferrite, Lagrange, RefTriangle
using FerriteGmsh: FerriteGmsh
using Gmsh: Gmsh, gmsh

include(joinpath(@__DIR__, "manufactured.jl"))

const AXISYM_MESH = joinpath(@__DIR__, "mesh.geo")

@variables ur(..) ut(..) uz(..) p(..)

"""
    axisym_grid(; c, tau_end, R = 8.0, hmax = 5c)

The graded triangle mesh of `mesh.geo` on `r ∈ [0, R]`, `z ∈ [-R, R]` as a `Ferrite.Grid` with facet sets `axis`,
`outer`, `top` and `bottom`. The element size is `c` times the distance to the origin, clamped to
`[c sqrt(tau_end) / 2, hmax]`, so halving `c` halves the size everywhere. Parameters reach the `.geo` file through
`gmsh.onelab.setNumber` before `gmsh.open`.
"""
function axisym_grid(; c, tau_end, R = 8.0, hmax = 5c)
    Gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        for (name, value) in (("R", R), ("c", c), ("hmin", c * sqrt(tau_end) / 2), ("hmax", hmax), ("order", 1))
            gmsh.onelab.setNumber("Parameters/$name", [Float64(value)])
        end
        gmsh.open(AXISYM_MESH)
        gmsh.model.mesh.generate(2)
        return FerriteGmsh.togrid()
    finally
        Gmsh.finalize()
    end
end

"""
    axisym_problem(grid; tend, abstol = 1e-8)

The `DAEProblem` of the axisymmetric swirl system on `grid` from t = 0 to `tend`. `abstol` is the absolute
tolerance of the differential rows; the pressure and Dirichlet rows are left out of the error control.
"""
function axisym_problem(grid; tend, abstol = 1.0e-8)
    Ω = MeshDomain(grid)
    Γaxis, Γout, Γtop, Γbottom = boundarypiece.(Ω, (:axis, :outer, :top, :bottom))
    UR, UT, UZ, P = ur(t, r, z), ut(t, r, z), uz(t, r, z), p(t, r, z)
    eqs = [
        Dt(UR) + UR * Dr(UR) + UZ * Dz(UR) - UT^2 / r + Dr(P) ~ laplacian(UR) - UR / r^2 + force_r(t, r, z),
        Dt(UT) + UR * Dr(UT) + UZ * Dz(UT) + UR * UT / r ~ laplacian(UT) - UT / r^2 + force_t(t, r, z),
        Dt(UZ) + UR * Dr(UZ) + UZ * Dz(UZ) + Dz(P) ~ laplacian(UZ) + force_z(t, r, z),
        0 ~ Dr(r * UR) + r * Dz(UZ),
    ]
    far = (Γout, Γtop, Γbottom)
    bcs = [
        [ur(t, Γ) ~ EXACT[:ur] for Γ in far]; [ut(t, Γ) ~ EXACT[:ut] for Γ in far];
        [uz(t, Γ) ~ EXACT[:uz] for Γ in far];
        ur(t, Γaxis) ~ 0; ut(t, Γaxis) ~ 0;
        p(t, Γout) ~ EXACT[:p];
        ur(0, r, z) ~ 0; ut(0, r, z) ~ 0; uz(0, r, z) ~ 0
    ]
    @named axisym = PDESystem(eqs, bcs, [(r, z) ∈ Ω, t ∈ Interval(0.0, tend)], [t, r, z], [UR, UT, UZ, P], [])
    P2, P1 = Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 1}()
    disc = FEMDiscretization(
        Ω, t; spaces = Dict(ur => P2, ut => P2, uz => P2, p => P1), pairing = Dict(eqs[4] => p),
        ibp = Dict(eqs[i] => Weighted(r) for i in 1:3), natural = [Γaxis], initializealg = CheckInit(), abstol
    )
    return discretize(axisym, disc)
end

"""
    weighted_errors(sol, tt, exact)

The r-weighted L² errors `sqrt(∫ r (u_h - u)² dr dz)` and norms `sqrt(∫ r u² dr dz)` at time `tt` of each field in
`exact`, a `Dict` from field name to a function `(t, r, z) -> value`, as two dictionaries `(err, nrm)`. Integrated
with an order-6 triangle rule on the solver's dense output.
"""
function weighted_errors(sol, tt, exact)
    dh = sol.disc_data.space.dh
    U = sol.original_sol(tt)
    sdh = only(dh.subdofhandlers)
    geometry = Ferrite.geometric_interpolation(Ferrite.getcelltype(dh.grid))
    rule = Ferrite.QuadratureRule{Ferrite.RefTriangle}(6)
    values = Dict(f => Ferrite.CellValues(rule, Ferrite.getfieldinterpolation(sdh, f), geometry) for f in keys(exact))
    err2, norm2 = Dict(f => 0.0 for f in keys(exact)), Dict(f => 0.0 for f in keys(exact))
    for cell in Ferrite.CellIterator(dh)
        dofs, xs = Ferrite.celldofs(cell), Ferrite.getcoordinates(cell)
        for (f, exact_f) in exact
            cv = values[f]
            Ferrite.reinit!(cv, xs)
            local_dofs = U[dofs[Ferrite.dof_range(sdh, f)]]
            for q in 1:Ferrite.getnquadpoints(cv)
                X = Ferrite.spatial_coordinate(cv, q, xs)
                weight = Ferrite.getdetJdV(cv, q) * X[1]
                value = exact_f(tt, X[1], X[2])
                err2[f] += (Ferrite.function_value(cv, q, local_dofs) - value)^2 * weight
                norm2[f] += value^2 * weight
            end
        end
    end
    return Dict(f => sqrt(err2[f]) for f in keys(exact)), Dict(f => sqrt(norm2[f]) for f in keys(exact))
end

"""
    axisym_errors(sol, tt)

Relative r-weighted L² errors of the swirl solution at `tt` against `EXACT_FUNCTIONS`, as `(velocity, pressure)`;
the velocity error combines the three components.
"""
function axisym_errors(sol, tt)
    err, nrm = weighted_errors(sol, tt, EXACT_FUNCTIONS)
    velocity = (:ur, :ut, :uz)
    return sqrt(sum(err[f]^2 for f in velocity) / sum(nrm[f]^2 for f in velocity)), err[:p] / nrm[:p]
end

# A fresh Newton matrix every step (`new_W_dt_cutoff = 0`): with the default cutoff the pressure error at t = 0.5 is
# 22% larger at c = 0.05 (the velocity is unchanged) for a solve that is 16% faster.
const AXISYM_ALG = DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0), linsolve = UMFPACKFactorization())

"""
    axisym_solve(; c = 0.1, tau_end = 1e-2, tol = 1e-8, alg = AXISYM_ALG, report = (0.5, 0.9, 0.99))

Mesh, discretize and solve to `t = 1 - tau_end`, and return the solution with the unknown count, the cell count,
the wall times and the relative errors `(t, velocity, pressure)` at the `report` times that were reached.
"""
function axisym_solve(; c = 0.1, tau_end = 1.0e-2, tol = 1.0e-8, alg = AXISYM_ALG, report = (0.5, 0.9, 0.99))
    tend = 1 - tau_end
    mesh_s = @elapsed grid = axisym_grid(; c, tau_end)
    build_s = @elapsed prob = axisym_problem(grid; tend, abstol = tol)
    saveat = [tt for tt in report if tt <= tend]
    solve_s = @elapsed sol = solve(prob, alg; reltol = tol, saveat)
    errors = [(tt, axisym_errors(sol, tt)...) for tt in saveat]
    return (; sol, prob, ncells = Ferrite.getncells(grid), ndofs = length(prob.u0), mesh_s, build_s, solve_s, errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    c = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.1
    tau_end = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0e-2
    run = axisym_solve(; c, tau_end)
    println("c = $c, tau_end = $tau_end: $(run.ncells) cells, $(run.ndofs) unknowns, solve $(round(run.solve_s; digits = 1)) s")
    for (tt, velocity, pressure) in run.errors
        println("  t = $tt: velocity error $velocity, pressure error $pressure")
    end
end
