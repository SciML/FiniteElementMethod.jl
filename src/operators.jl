# Nonlinear operator and linear-operator cache with the reassembly policy (spec §8.1–8.3, red-cell m5).

"""
    NonlinearOperator(ctx, kernels; ch = nothing, pattern = allocate_matrix(dh, ch))

The `:nonlinear` kernels of `kernels` on `ctx`, with preallocated residual `r`, Jacobian `J` (sparsity from
`allocate_matrix(dh, ch)`, or `pattern`) and element scratch. Use [`residual!`](@ref), [`jacobian!`](@ref) and
[`residual_jacobian!`](@ref); `op(r, u, p, t)` is `residual!(r, op, u, p, t)`.
"""
struct NonlinearOperator{C <: AssemblyContext, V <: Tuple, G <: Tuple, JT, T <: AbstractFloat}
    ctx::C
    volume::V
    groups::G
    r::Vector{T}
    J::JT
    Je::Matrix{T}
    re::Vector{T}
    u_elem::Vector{T}
end

function NonlinearOperator(ctx::AssemblyContext, kernels; ch = nothing, pattern = allocate_pattern(ctx, ch))
    ne = _max_ndofs(ctx.dh)
    volume, groups = _split(ctx, kernels, (:nonlinear,))
    r = zeros(Ferrite.ndofs(ctx.dh))
    return NonlinearOperator(ctx, volume, groups, r, copy(pattern), zeros(ne, ne), zeros(ne), zeros(ne))
end

function _nonlinear!(r, aJ, op::NonlinearOperator, u, p, t)
    fill!(r, 0)
    _assemble!(NonlinearOp(aJ, r, op.Je, op.re, u, op.u_elem), op.ctx, op.volume, op.groups, p, t)
    return r
end

"""
    residual!(r, op::NonlinearOperator, u, p, t)

Overwrite `r` with the nonlinear residual `r(u)` at `u`, without the Jacobian.
"""
residual!(r, op::NonlinearOperator, u, p, t) = _nonlinear!(r, nothing, op, u, p, t)

"""
    jacobian!(J, op::NonlinearOperator, u, p, t)

Overwrite `J` (with the sparsity of `op.J`) with `∂r/∂u` at `u`; the residual goes to `op.r`.
"""
function jacobian!(J, op::NonlinearOperator, u, p, t)
    _nonlinear!(op.r, Ferrite.start_assemble(J), op, u, p, t)
    return J
end

"""
    residual_jacobian!(r, J, op::NonlinearOperator, u, p, t)

Overwrite `r` and `J` in one pass over the cells.
"""
function residual_jacobian!(r, J, op::NonlinearOperator, u, p, t)
    _nonlinear!(r, Ferrite.start_assemble(J), op, u, p, t)
    return r, J
end

(op::NonlinearOperator)(r, u, p, t) = residual!(r, op, u, p, t)

# Ferrite releases before `task_local_copy` (1.7.0) only have `copy`; newer ones deprecate it.
const _values_copy = isdefined(Ferrite, :task_local_copy) ? Ferrite.task_local_copy : copy

# `reinit!` mutates CellValues and FacetValues, so each load evaluation gets its own copies and
# concurrent solves of one problem's remakes share no mutable state.
function _fresh_values(ctx::AssemblyContext{NF, DH, CV, FV}) where {NF, DH, CV, FV}
    return AssemblyContext{NF, DH, CV, FV}(
        ctx.dh, ctx.fields, map(_values_copy, ctx.cellvalues), map(_values_copy, ctx.facetvalues), ctx.facetsets, ctx.facetcells
    )
end

# A copy of `op` that shares no mutable state with it: new buffers, Jacobian and values objects.
function independent_copy(op::NonlinearOperator)
    return NonlinearOperator(_fresh_values(op.ctx), op.volume, op.groups, zero(op.r), copy(op.J), zero(op.Je), zero(op.re), zero(op.u_elem))
end

const LINEAR_TARGET = (mass = :M, bilinear = :K, source = :F)

"""
    LinearCache(ctx, wf, kernels, p; t = 0.0, ch = nothing)

`M`, `K` and `F` of `kernels = emit_kernels(wf)`, assembled at `(p, t)`, with the kinds whose terms are `timedep`
or `paramdep`. [`update!`](@ref) reassembles only the stale ones; `assemblies` counts assemblies per matrix.
"""
mutable struct LinearCache{C <: AssemblyContext, KS, MT, P}
    const ctx::C
    const kernels::KS
    const M::MT
    const K::MT
    const F::Vector{Float64}
    const timedep::Set{Symbol}
    const paramdep::Set{Symbol}
    const assemblies::Dict{Symbol, Int}
    p::P
    t::Float64
end

function LinearCache(ctx::AssemblyContext, wf::WeakForm, kernels, p; t = 0.0, ch = nothing)
    terms = [wf.volume; wf.facet]
    length(fused_terms(wf)) == length(kernels) || throw(ArgumentError("kernels must be emit_kernels(wf), one per equation, kind and piece of wf"))
    timedep = Set{Symbol}(term.kind for term in terms if term.timedep)
    paramdep = Set{Symbol}(term.kind for term in terms if term.paramdep)
    K = allocate_pattern(ctx, ch)
    M = copy(K)
    F = zeros(Ferrite.ndofs(ctx.dh))
    assemble_linear!(M, K, F, ctx, kernels, p; t)
    assemblies = Dict(:M => 1, :K => 1, :F => 1)
    return LinearCache(ctx, kernels, M, K, F, timedep, paramdep, assemblies, copy(p), Float64(t))
end

"""
    update!(cache::LinearCache, p, t)

Reassemble each of `M`, `K` and `F` whose terms depend on `t` (when `t` changed) or on `p` (when `p` changed).
"""
function update!(cache::LinearCache, p, t)
    tchanged = t != cache.t
    pchanged = !isequal(p, cache.p)
    stale = Tuple(
        kind for kind in LINEAR_KINDS
            if (tchanged && kind in cache.timedep) || (pchanged && kind in cache.paramdep)
    )
    if !isempty(stale)
        assemble_linear!(cache.M, cache.K, cache.F, cache.ctx, cache.kernels, p; t, kinds = stale)
        for kind in stale
            cache.assemblies[LINEAR_TARGET[kind]] += 1
        end
    end
    cache.p = copy(p)
    cache.t = t
    return cache
end
