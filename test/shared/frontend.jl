# Helpers for end-to-end front-end tests: PDESystem -> DAEProblem -> solve -> nodal error against
# an exact solution.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using OrdinaryDiffEqBDF: NLNewton
using Ferrite: Ferrite
using SciMLBase: SciMLBase
using Symbolics: Num

# A fresh W at every step keeps the algebraic Dirichlet rows exact (see test/Frontend/api_example.jl).
const EXACT_ROWS = DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0))

errmsg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

"""
    fem_solve(eqs, bcs, Ω, coords, t, dvs; ps = [], tspan = (0.0, 1.0), alg = EXACT_ROWS, disc_kwargs...)

Build the `PDESystem` on `coords ∈ Ω`, discretize it with `FEMDiscretization(Ω, t; disc_kwargs...)`
and solve it at tight tolerances. Returns `(sol, prob)`.
"""
function fem_solve(
        eqs, bcs, Ω, coords, t, dvs; ps = [], tspan = (0.0, 1.0), alg = EXACT_ROWS,
        solve_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10), disc_kwargs...
    )
    domains = [coords ∈ Ω, t ∈ Interval(tspan...)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, coords...], dvs, ps)
    prob = discretize(pdesys, FEMDiscretization(Ω, t; disc_kwargs...))
    return solve(prob, alg; solve_kwargs...), prob
end

"Largest nodal error of `dv` against `exact(t, x...)` over every saved time and mesh vertex."
function nodal_error(sol, dv, exact)
    vals = sol[dv]
    xs = [sol[Num(x)] for x in sol.disc_data.x̄]
    return maximum(abs(vals[k, i] - exact(sol.t[k], getindex.(xs, i)...)) for k in eachindex(sol.t), i in axes(vals, 2))
end
