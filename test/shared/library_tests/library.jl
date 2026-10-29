# Spec §10.5, M2 layer "Linear numeric": PDESystemLibrary 0.1.10's linear systems heat_1d1, pdesys,
# sph, convdiff2 and convdiff4, solved end to end at n = 16, 32, 64 (and 128 for convdiff4).
# The library writes MethodOfLines endpoint conditions on interval domains; each system is
# converted to trace conditions on the named pieces of a 1D mesh of the same interval (the inverse
# of PDEBase.to_box_bcs, which must take the trace form back to the library's conditions exactly).
# Exact solutions come from the symbolic `sys.analytic`, never `analytic_func` (L1). Library
# defects (spec §11.2, L1–L7) are `@test_broken` with their identifier, never worked around.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using PDESystemLibrary: PDESystemLibrary
using PDEBase: PDEBase, substitute_traces
using Ferrite: Line
using SymbolicUtils: SymbolicUtils, iscall, operation, arguments

include(joinpath(@__DIR__, "..", "frontend.jl"))

const MIN_RATE = 1.9
librarysystem(name) = only(filter(s -> nameof(s) == name, PDESystemLibrary.all_systems))

spatial_interval(sys, x) = only(d.domain for d in sys.domain if isequal(d.variables, x))
time_interval(sys, t) = only(d.domain for d in sys.domain if isequal(d.variables, t))

# Every term of `ex` (unwrapped) that satisfies `pred`.
function subterms(pred, ex, found = [])
    ex = SymbolicUtils.unwrap_const(Symbolics.unwrap(ex))
    pred(ex) && push!(found, ex)
    iscall(ex) && foreach(a -> subterms(pred, a, found), arguments(ex))
    return found
end

"""
    trace_bcs(sys, Ω)

The conditions of the 1D library system `sys` (time `t`, space `x`) with every endpoint
condition rewritten on the pieces `:x1_lo`, `:x1_hi` of `Ω`: `u(t, a)` becomes `u(t, Γ)` and
`Dx(u(t, a))` becomes `∓Dn_Γ(u(t, Γ))`. Initial conditions are kept.
"""
function trace_bcs(sys, Ω)
    t, x = sys.ivs
    I = spatial_interval(sys, x)
    ends = ((DomainSets.infimum(I), boundarypiece(Ω, :x1_lo), -1), (DomainSets.supremum(I), boundarypiece(Ω, :x1_hi), 1))
    dvops = [operation(Symbolics.unwrap(u)) for u in sys.dvs]
    function endpoint(ex)
        iscall(ex) && any(isequal(operation(ex)), dvops) || return nothing
        a = SymbolicUtils.unwrap_const(arguments(ex)[2])
        return a isa Number ? a : nothing
    end
    rules = Dict()
    for bc in sys.bcs
        for call in subterms(ex -> endpoint(ex) isa Number, bc.lhs - bc.rhs)
            a = endpoint(call)
            (_, Γ, sign) = only(e for e in ends if isapprox(e[1], a; atol = 8eps(e[1] + 1)))
            trace = operation(call)(t, Γ)
            rules[call] = trace
            rules[Differential(x)(call)] = sign * NormalDerivative(Γ)(trace)
        end
    end
    return [substitute_traces(bc, rules) for bc in sys.bcs]
end

params(sys) = sys.ps isa AbstractVector ? sys.ps : []

"The library system on the domain `Ω` with trace conditions, time span `(t0, tend)`."
function trace_system(sys, Ω, tend)
    t, x = sys.ivs
    t0 = DomainSets.infimum(time_interval(sys, t))
    domains = [(Ω isa MeshDomain ? (x,) : x) ∈ Ω, t ∈ Interval(t0, tend)]
    return PDESystem(sys.eqs, trace_bcs(sys, Ω), domains, sys.ivs, sys.dvs, params(sys); name = nameof(sys))
end
mesh_domain(sys, n) = (I = spatial_interval(sys, sys.ivs[2]); MeshDomain(rectangle_grid(Line, (n,); lo = (DomainSets.infimum(I),), hi = (DomainSets.supremum(I),))))

"`sys.analytic` with the parameter values substituted, as a function `(t, x)`."
function symbolic_exact(sys)
    rhs = only(sys.analytic).rhs
    rhs = substitute(rhs, Dict(first(p) => last(p) for p in params(sys)))
    f = Symbolics.build_function(rhs, sys.ivs...; expression = Val(false))
    I = spatial_interval(sys, sys.ivs[2])
    lo, hi = DomainSets.infimum(I), DomainSets.supremum(I)
    # sph's sin(r)/r has a removable singularity at r = 0: take the limit from inside the interval
    return (t_, x_) -> (v = f(t_, x_); isfinite(v) ? v : f(t_, x_ + 1.0e-9 * (hi - lo) * sign((lo + hi) / 2 - x_)))
end

# convdiff: k f'' + v f' = 0 with f(0) = 0, f(L) = 1, reached to below 1e-8 by t = 30 (slowest
# decay rates 0.645 for convdiff2, 4.55 for convdiff4), used because the library's reference is
# NaN everywhere (L2).
convdiff_steady(k, v, L) = (_, z) -> (1 - exp(-v * z / k)) / (1 - exp(-v * L / k))

const CASES = [
    (name = :heat_1d1, tend = 0.05, ns = (16, 32, 64), err64 = 2.2e-4), # L7: compare at t = 0.05, not t_end = 1
    (name = :pdesys, tend = 1.0, ns = (16, 32, 64), err64 = 7.39e-5),
    (name = :sph, tend = 1.0, ns = (16, 32, 64), err64 = 2.43e-7),
    (name = :convdiff2, tend = 30.0, ns = (16, 32, 64), err64 = 1.78e-4, exact = convdiff_steady(1.0, -1.0, 5.0)),
    (name = :convdiff4, tend = 30.0, ns = (16, 32, 64, 128), err64 = 2.99e-2, exact = convdiff_steady(0.5, 3.0, 10.0)),
]

analytic_func_evaluates(sys, args...) = try
    all(isfinite, (f(Float64[last(p) for p in params(sys)], args...) for f in values(sys.analytic_func)))
catch
    false
end

"Solve the converted system at every n of `case`; the nodal errors at `case.tend` against `exact`."
function nodal_errors(case, sys, exact)
    t = sys.ivs[1]
    return map(case.ns) do n
        Ω = mesh_domain(sys, n)
        prob = discretize(trace_system(sys, Ω, case.tend), FEMDiscretization(Ω, t; saveat = [case.tend]))
        sol = solve(prob, EXACT_ROWS; abstol = 1.0e-11, reltol = 1.0e-9)
        @test SciMLBase.successful_retcode(sol)
        @test sol.t[end] == case.tend
        nodal_error(sol, sys.dvs[1], exact)
    end
end

@testset verbose = true "PDESystemLibrary linear numeric" begin
    @testset "$(case.name)" for case in CASES
        sys = librarysystem(case.name)
        I = spatial_interval(sys, sys.ivs[2])

        @testset "lowering round trip" begin
            @test all(isequal.(PDEBase.to_box_bcs(trace_system(sys, I, case.tend)).bcs, sys.bcs))
        end

        exact = haskey(case, :exact) ? case.exact : symbolic_exact(sys)
        if case.name === :sph
            # L8 (new, spec §11.2 numbering continued): sph's initial condition sin(r)/r is NaN at
            # r = 0, a node of every mesh of [0, 1], and the front end rejects non-finite initial data.
            Ω = mesh_domain(sys, 16)
            @test_throws "u(0, r) ~ sin(r) / r is not finite at the node coordinate (0.0,)" discretize(
                trace_system(sys, Ω, case.tend), FEMDiscretization(Ω, sys.ivs[1])
            )
            @test_broken try
                errs = nodal_errors(case, sys, exact)
                log2(errs[end - 1] / errs[end]) >= MIN_RATE && errs[end] <= 1.5 * case.err64
            catch e
                e isa ArgumentError || rethrow()
                false
            end
            continue
        end
        errs = nodal_errors(case, sys, exact)
        rate = log2(errs[end - 1] / errs[end])
        err64 = errs[findfirst(==(64), case.ns)]
        @info "Library: $(case.name)" ns = case.ns errs rate err64 reference = case.err64
        @test rate >= MIN_RATE
        @test err64 <= 1.5 * case.err64
    end

    @testset "library defects" begin
        # L1 (analytic_func throwing for Pair parameters) no longer reproduces with the
        # ModelingToolkitBase 1.78; the exact solutions still come from
        # sys.analytic as spec §10.5 requires.
        @test analytic_func_evaluates(librarysystem(:heat_1d1), 0.05, 0.3)
        # L2: the convdiff references are NaN everywhere.
        @test_broken analytic_func_evaluates(librarysystem(:convdiff2), 30.0, 2.5)
        @test_broken analytic_func_evaluates(librarysystem(:convdiff4), 30.0, 5.0)
    end
end
