# Precompile workload: a seam-free case (i) chain, weak form to assembled operators, on a tiny grid.

"""
    workload_chain(k = 2.0; nel = 2)

Build and assemble `∂ₜu = ∇·(k∇u) − u³ + xy + 1` with `k∇u·n = k·P₁` on `right`, from a hand-built strong-form
analysis (no PDEBase) on an `nel × nel` triangle grid: `weak_form`, `emit_kernels`, `assemble_linear` and one
`residual!`. The precompile workload runs it; a different `k` generates different kernels, which is how the cold-latency
gate measures time to first assembly without reusing the workload's own kernels.
"""
function workload_chain(k = 2.0; nel = 2)
    var(name, n) = [Symbolics.variable(name, i) for i in 1:n]
    U, G, Ud, X, N, P = var(:U, 1), var(:G, 2), var(:Ud, 1), var(:X, 2), var(:N, 2), var(:P, 1)
    slots = (; U, G, Ud, H = nothing, X, T = Symbolics.variable(:T), N, P)
    analysis = (;
        slots, nfields = 1, dim = 2, mass = [Ud[1]], flux = [[-k * G[1], -k * G[2]]],
        remainder = [U[1]^3 - X[1] * X[2] - 1], natural = [(; piece = "right", eq = 1, Φ = -k * P[1])],
    )
    kernels = emit_kernels(weak_form(analysis))
    grid = Ferrite.generate_grid(Ferrite.Triangle, (nel, nel))
    ip = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.RefTriangle}(2), ip)
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{Ferrite.RefTriangle}(2), ip)
    ctx = AssemblyContext(dh, [:u], cv, fv, Dict("right" => "right"))
    p = [1.5]
    linear = assemble_linear(ctx, kernels, p)
    n = Ferrite.ndofs(dh)
    r = residual!(zeros(n), NonlinearOperator(ctx, kernels), ones(n), p, 0.0)
    return (; linear..., r)
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        workload_chain()
    end
end

"""
    workload_solution(; nel = 2)

Lower a heat equation on an `nel × nel` triangle grid, wrap two hand-built saved states of it in a
`PDETimeSeriesSolution` (a `DAESolution` of SciMLBase's own constructor, with linear interpolation, so no solver
package is needed) and read it: a field, a projected gradient, and a scalar and a gridded point query. The precompile
workload runs it, so that the first read of a solution does not infer the solution wrapper from scratch.
The solution type is not the one a solver returns (no algorithm, linear interpolation), so only the reads that do not
depend on it are fully covered; the first read of a solver's solution still specializes the rest.
"""
function workload_solution(; nel = 2)
    @variables x y u(..)
    t = ModelingToolkit.t_nounits
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    Ω = MeshDomain(rectangle_grid(Ferrite.Triangle, (nel, nel)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + 1
    bcs = [[u(t, Γ) ~ 0 for Γ in sides]; u(0, x, y) ~ 0]
    domains = [(x, y) ∈ Ω, t ∈ DomainSets.Interval(0.0, 1.0)]
    system = ModelingToolkit.PDESystem([heat], bcs, domains, [t, x, y], [u(t, x, y)], []; name = :workload_solution)
    prob = SciMLBase.discretize(system, FEMDiscretization(Ω, t))
    ts = [0.0, 1.0]
    us = [copy(prob.u0), copy(prob.u0)]
    interp = SciMLBase.LinearInterpolation(ts, us)
    sol = SciMLBase.build_solution(prob, nothing, ts, us; interp, dense = true, retcode = SciMLBase.ReturnCode.Success)
    wrapped = SciMLBase.PDETimeSeriesSolution(sol, prob.problem_type)
    field = u(t, x, y)
    wrapped[field]
    wrapped[Dx(field)]
    wrapped(0.5, 0.3, 0.4; dv = field)
    wrapped(ts, [0.3, 0.4], [0.2]; dv = field)
    return nothing
end

PrecompileTools.@compile_workload begin
    workload_solution()
end
