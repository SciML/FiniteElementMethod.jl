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
