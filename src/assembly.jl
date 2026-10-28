# Generic Ferrite assembly driver for per-quadrature-point kernels (spec §6.5, §6.6).
# One pass over the cells: each cell is reinitialised once, every volume kernel and every facet kernel whose
# piece contains one of the cell's facets adds into the same element scratch, which is assembled once.
# Kernels are iterated as a tuple through recursive calls, so each kernel's loop is compiled for its own type.

"""
    AssemblyContext(dh, fields, cv, fv, pieces = Dict())

What the driver needs to assemble kernels: the `DofHandler`, its field names in equation/field index order, the
`CellValues` and `FacetValues` shared by all fields, and `pieces`, a map from boundary piece to its facet set,
given as a facet-set name of the grid or as a set of `FacetIndex`. Each facet set is copied at construction, so
changing the caller's set or the grid's facet sets afterwards does not affect the context; build a new one instead.
"""
struct AssemblyContext{NF, DH <: Ferrite.DofHandler, CV, FV}
    dh::DH
    fields::Vector{Symbol}
    cv::CV
    fv::FV
    facetsets::Dict{Any, Any}
    facetcells::BitSet   # cells with a facet in some piece; the others skip the per-facet membership tests
end

function AssemblyContext(dh::Ferrite.DofHandler, fields, cv, fv, pieces = Dict())
    fields = collect(Symbol, fields)
    nb = Ferrite.getnbasefunctions(cv)
    for f in fields
        length(Ferrite.dof_range(dh, f)) == nb ||
            throw(ArgumentError("field $f has $(length(Ferrite.dof_range(dh, f))) dofs per cell, but the CellValues have $nb base functions; all fields must share one interpolation"))
    end
    facetsets = Dict{Any, Any}()
    for (piece, set) in pieces
        facetsets[piece] = Set{Ferrite.FacetIndex}(set isa AbstractString ? _named_facetset(dh.grid, piece, set) : set)
    end
    ncells = Ferrite.getncells(dh.grid)
    facetcells = BitSet(fi[1] for set in values(facetsets) for fi in set if 1 ≤ fi[1] ≤ ncells)
    return AssemblyContext{length(fields), typeof(dh), typeof(cv), typeof(fv)}(dh, fields, cv, fv, facetsets, facetcells)
end

function _named_facetset(grid, piece, name)
    try
        return Ferrite.getfacetset(grid, name)
    catch err
        err isa KeyError && error("piece $piece: the grid has no facet set named \"$name\"")
        rethrow()
    end
end

allocate_pattern(ctx::AssemblyContext, ::Nothing) = Ferrite.allocate_matrix(ctx.dh)
allocate_pattern(ctx::AssemblyContext, ch) = Ferrite.allocate_matrix(ctx.dh, ch)

"""
    assemble_linear(ctx, kernels, p; t = 0.0, ch = nothing) -> (; M, K, F)

Assemble the `:mass`, `:bilinear` and `:source` kernels into `M`, `K` and `F = −∫source`, sharing one sparsity
pattern (from `allocate_matrix(dh, ch)` when a `ConstraintHandler` is given). Kernels with `coeffs` are contracted
with the shape functions once per quadrature point; mass and bilinear kernels without are evaluated with a unit
trial function, which is exact because they are linear in the state slots.
"""
function assemble_linear(ctx::AssemblyContext, kernels, p; t = 0.0, ch = nothing)
    K = allocate_pattern(ctx, ch)
    M = copy(K)
    F = zeros(Ferrite.ndofs(ctx.dh))
    assemble_linear!(M, K, F, ctx, kernels, p; t)
    return (; M, K, F)
end

const LINEAR_KINDS = (:mass, :bilinear, :source)

"""
    assemble_linear!(M, K, F, ctx, kernels, p; t = 0.0, kinds = (:mass, :bilinear, :source))

Assemble in place the kernels of `kinds` only: `M` is overwritten when `:mass` is in `kinds`, `K` when `:bilinear`
is, `F` when `:source` is, and the others are left as they are.
"""
function assemble_linear!(M, K, F, ctx::AssemblyContext{NF}, kernels, p; t = 0.0, kinds = LINEAR_KINDS) where {NF}
    ne = NF * Ferrite.getnbasefunctions(ctx.cv)
    aM = Ferrite.start_assemble(M; fillzero = :mass in kinds)
    aK = Ferrite.start_assemble(K; fillzero = :bilinear in kinds)
    :source in kinds && fill!(F, 0)
    op = LinearOp(aM, aK, F, zeros(ne, ne), zeros(ne, ne), zeros(ne))
    _assemble!(op, ctx, _split(ctx, kernels, kinds)..., p, t)
    return M, K, F
end

"""
    assemble_nonlinear(ctx, kernels, u, p; t = 0.0, ch = nothing) -> (; r, J)

Assemble the `:nonlinear` kernels at state `u` into the residual `r(u)` and its Jacobian `J = ∂r/∂u` in one pass,
the Jacobian from each kernel's `tangent`.
"""
function assemble_nonlinear(ctx::AssemblyContext, kernels, u, p; t = 0.0, ch = nothing)
    J = allocate_pattern(ctx, ch)
    r = zeros(Ferrite.ndofs(ctx.dh))
    assemble_nonlinear!(r, J, ctx, kernels, u, p; t)
    return (; r, J)
end

function assemble_nonlinear!(r, J, ctx::AssemblyContext{NF}, kernels, u, p; t = 0.0) where {NF}
    ne = NF * Ferrite.getnbasefunctions(ctx.cv)
    fill!(r, 0)
    op = NonlinearOp(Ferrite.start_assemble(J), r, zeros(ne, ne), zeros(ne), u, zeros(ne))
    _assemble!(op, ctx, _split(ctx, kernels, (:nonlinear,))..., p, t)
    return r, J
end

struct LinearOp{AM, AK, FV <: AbstractVector{Float64}}
    aM::AM
    aK::AK
    F::FV
    Me::Matrix{Float64}
    Ke::Matrix{Float64}
    Fe::Vector{Float64}
end

# `aJ === nothing` assembles the residual only.
struct NonlinearOp{AJ, R <: AbstractVector{Float64}, U}
    aJ::AJ
    r::R
    Je::Matrix{Float64}
    re::Vector{Float64}
    u::U
    u_elem::Vector{Float64}
end

"Volume kernels of `kinds` as a tuple, and one `(facet set, kernel tuple, piece)` group per facet piece."
function _split(ctx::AssemblyContext{NF}, kernels, kinds) where {NF}
    for k in kernels
        k.kind in (:mass, :bilinear, :source, :nonlinear) || error("kernel kind $(repr(k.kind)) is not one of :mass, :bilinear, :source, :nonlinear")
        1 ≤ k.eq ≤ NF || error("kernel equation $(k.eq) is outside 1:$NF")
        k.kind === :nonlinear && k.tangent === nothing && error("nonlinear kernel of equation $(k.eq) has no tangent")
        k.f === nothing && (k.coeffs === nothing || k.kind === :nonlinear) &&
            throw(ArgumentError("$(k.kind) kernel of equation $(k.eq) has no f: pass f, or coeffs for a linear kind"))
    end
    selected = [k for k in kernels if k.kind in kinds]
    volume = _fuse([k for k in selected if k.piece === nothing])
    pieces = unique(k.piece for k in selected if k.piece !== nothing)
    groups = map(pieces) do piece
        haskey(ctx.facetsets, piece) || error("no facet set is given for piece $piece")
        (ctx.facetsets[piece], _fuse([k for k in selected if isequal(k.piece, piece)]), piece)
    end
    return volume, Tuple(groups)
end

"""
The linear kernels of one equation that have `coeffs`, at most one per kind, evaluated together in a single
loop over quadrature points and shape functions; `nothing` for an absent kind.
"""
struct LinearBlock{CM, CK, CF}
    eq::Int
    mass::CM
    bilinear::CK
    source::CF
end

# The linear kernels with `coeffs` grouped into `LinearBlock`s, one per equation unless an equation has two kernels
# of one kind; every other kernel is kept as it is.
function _fuse(kernels)
    fusible(k) = k.kind in LINEAR_KINDS && k.coeffs !== nothing
    out = Any[k for k in kernels if !fusible(k)]
    blocks = Tuple{Int, Dict{Symbol, Any}}[]
    for k in kernels
        fusible(k) || continue
        i = findfirst(((eq, slots),) -> eq == k.eq && slots[k.kind] === nothing, blocks)
        i === nothing && (push!(blocks, (k.eq, Dict{Symbol, Any}(kind => nothing for kind in LINEAR_KINDS))); i = length(blocks))
        blocks[i][2][k.kind] = k.coeffs
    end
    for (eq, c) in blocks
        push!(out, LinearBlock(eq, c[:mass], c[:bilinear], c[:source]))
    end
    return Tuple(out)
end

# Function barrier: `volume` and `groups` are concrete tuples from here on.
function _assemble!(op, ctx::AssemblyContext{NF}, volume::Tuple, groups::Tuple, p, t) where {NF}
    (; dh, cv, fv) = ctx
    ranges = ntuple(f -> Ferrite.dof_range(dh, ctx.fields[f]), Val(NF))
    counts = zeros(Int, length(groups))
    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        _start_cell!(op, Ferrite.celldofs(cell))
        xs = Ferrite.getcoordinates(cell)
        _kernels!(op, volume, cv, xs, ranges, p, t)
        if !isempty(groups) && Ferrite.cellid(cell) in ctx.facetcells
            for lf in 1:Ferrite.nfacets(cell)
                _facet_groups!(op, groups, counts, 1, Ferrite.FacetIndex(Ferrite.cellid(cell), lf), fv, xs, ranges, p, t)
            end
        end
        _finish_cell!(op, Ferrite.celldofs(cell))
    end
    for (g, (set, _, piece)) in enumerate(groups)
        counts[g] == length(set) ||
            error("piece $piece: assembled $(counts[g]) facets, but its facet set has $(length(set)); the set must hold FacetIndex entries of this grid")
    end
    return nothing
end

_facet_groups!(op, ::Tuple{}, counts, g, fi, fv, xs, ranges, p, t) = nothing
function _facet_groups!(op, groups::Tuple, counts, g, fi, fv, xs, ranges, p, t)
    set, kernels, _ = first(groups)
    if fi in set
        Ferrite.reinit!(fv, xs, fi[2])
        counts[g] += 1
        _kernels!(op, kernels, fv, xs, ranges, p, t)
    end
    return _facet_groups!(op, Base.tail(groups), counts, g + 1, fi, fv, xs, ranges, p, t)
end

_kernels!(op, ::Tuple{}, vals, xs, ranges, p, t) = nothing
function _kernels!(op, kernels::Tuple, vals, xs, ranges, p, t)
    _kernel!(op, first(kernels), vals, xs, ranges, p, t)
    return _kernels!(op, Base.tail(kernels), vals, xs, ranges, p, t)
end

_normal(vals::Ferrite.AbstractFacetValues, q, X) = Ferrite.getnormal(vals, q)
_normal(vals, q, X) = zero(X)

_unit(::Val{NF}, b, φ) where {NF} = ntuple(f -> ifelse(f == b, φ, zero(φ)), Val(NF))
function _unit_gradient(::Val{NF}, b, ∇φ::Ferrite.Vec{DIM, T}) where {NF, DIM, T}
    return ntuple(m -> ifelse(cld(m, DIM) == b, ∇φ[mod1(m, DIM)], zero(T)), Val(NF * DIM))
end

function _start_cell!(op::LinearOp, dofs)
    fill!(op.Me, 0)
    fill!(op.Ke, 0)
    fill!(op.Fe, 0)
    return nothing
end

function _finish_cell!(op::LinearOp, dofs)
    Ferrite.assemble!(op.aK, dofs, op.Ke)
    Ferrite.assemble!(op.F, dofs, op.Fe)
    Ferrite.assemble!(op.aM, dofs, op.Me)
    return nothing
end

# The contraction below indexes the `coeffs` tuple only with compile-time constants, so `StructuralZero` entries
# vanish from the generated code and blocks that are structurally zero are never written.

# Test-function basis (φ, ∂₁φ, …, ∂_dim φ) of base function `i`, matching the `w = [V; W]` order of `coeffs`.
@inline _test_basis(vals, q, i) = (Ferrite.shape_value(vals, q, i), _ntuple(Ferrite.shape_gradient(vals, q, i))...)
@inline _ntuple(v::Ferrite.Vec{D}) where {D} = ntuple(d -> v[d], Val(D))

@inline _sum(t::Tuple{Any}) = t[1]
@inline _sum(t::Tuple) = t[1] + _sum(Base.tail(t))

# Σ_w c[offset + w] ψ[w], and per trial slot s the trial coefficient Σ_w c[(s - 1) * NW + w] ψ[w].
@inline _test_contract(c, ψ::NTuple{NW}, offset = 0) where {NW} = _sum(ntuple(@inline(w -> c[offset + w] * ψ[w]), Val(NW)))
@inline function _trial_coeffs(c, ψ::NTuple{NW}, ::Val{NS}) where {NW, NS}
    return ntuple(@inline(s -> _test_contract(c, ψ, (s - 1) * NW)), Val(NS))
end

@inline _static_foreach(f, ::Val{N}) where {N} = (ntuple(@inline(b -> (f(b); nothing)), Val(N)); nothing)

@inline _add!(Ae, row, col, value, dΩ) = (Ae[row, col] += value * dΩ; nothing)
_add!(Ae, row, col, ::StructuralZero, dΩ) = nothing

# Row `row` of a mass block, trial slots Ud: Σ_j tt[b] φ_j for every field b; nothing for an absent block.
_mass_row!(Ae, row, ranges::NTuple{NF}, ::Nothing, vals, q, nb, dΩ) where {NF} = nothing
function _mass_row!(Ae, row, ranges::NTuple{NF}, tt, vals, q, nb, dΩ) where {NF}
    for j in 1:nb
        φ = Ferrite.shape_value(vals, q, j)
        _static_foreach(@inline(b -> _add!(Ae, row, ranges[b][j], tt[b] * φ, dΩ)), Val(NF))
    end
    return nothing
end

# Row `row` of a state block, trial slots [U; G]: Σ_j (tt[U_b] φ_j + Σ_d tt[G_bd] ∂_d φ_j) for every field b.
_state_row!(Ae, row, ranges::NTuple{NF}, ::Nothing, vals, q, nb, dΩ) where {NF} = nothing
function _state_row!(Ae, row, ranges::NTuple{NF}, tt, vals, q, nb, dΩ) where {NF}
    for j in 1:nb
        χ = _test_basis(vals, q, j)
        DIM = length(χ) - 1
        _static_foreach(Val(NF)) do b
            @inline
            value = tt[b] * χ[1] + _sum(ntuple(@inline(d -> tt[NF + (b - 1) * DIM + d] * χ[1 + d]), Val(DIM)))
            _add!(Ae, row, ranges[b][j], value, dΩ)
        end
    end
    return nothing
end

_coeffs(::Nothing, args...) = nothing
_coeffs(c, args...) = c(args...)
_trial_coeffs(::Nothing, ψ::NTuple{NW}, ::Val{NS}) where {NW, NS} = nothing
_test_contract(::Nothing, ψ::NTuple{NW}) where {NW} = StructuralZero()

function _kernel!(op::LinearOp, blk::LinearBlock, vals, xs, ranges::NTuple{NF}, p, t) where {NF}
    nb = Ferrite.getnbasefunctions(vals)
    rows = ranges[blk.eq]
    for q in 1:Ferrite.getnquadpoints(vals)
        dΩ = Ferrite.getdetJdV(vals, q)
        X = Ferrite.spatial_coordinate(vals, q, xs)
        N = _normal(vals, q, X)
        cM, cK, cF = _coeffs(blk.mass, X, t, N, p), _coeffs(blk.bilinear, X, t, N, p), _coeffs(blk.source, X, t, N, p)
        for i in 1:nb
            ψ = _test_basis(vals, q, i)
            op.Fe[rows[i]] -= _test_contract(cF, ψ) * dΩ
            _mass_row!(op.Me, rows[i], ranges, _trial_coeffs(cM, ψ, Val(NF)), vals, q, nb, dΩ)
            _state_row!(op.Ke, rows[i], ranges, _trial_coeffs(cK, ψ, Val(NF * length(ψ))), vals, q, nb, dΩ)
        end
    end
    return nothing
end

# A kernel without `coeffs`: evaluate `f` with a unit trial function, exact because these kinds are linear in the
# state.
function _kernel!(op::LinearOp, k::KernelSpec, vals, xs, ranges::NTuple{NF}, p, t) where {NF}
    nb = Ferrite.getnbasefunctions(vals)
    rows = ranges[k.eq]
    mass = k.kind === :mass
    Ae = mass ? op.Me : op.Ke
    U0 = ntuple(_ -> 0.0, Val(NF))
    for q in 1:Ferrite.getnquadpoints(vals)
        dΩ = Ferrite.getdetJdV(vals, q)
        X = Ferrite.spatial_coordinate(vals, q, xs)
        N = _normal(vals, q, X)
        G0 = _unit_gradient(Val(NF), 0, zero(X))
        for i in 1:nb
            V, W = Ferrite.shape_value(vals, q, i), Ferrite.shape_gradient(vals, q, i)
            if k.kind === :source
                op.Fe[rows[i]] -= k.f(U0, G0, U0, V, W, X, t, N, p) * dΩ
                continue
            end
            for b in 1:NF, j in 1:nb
                φ, ∇φ = Ferrite.shape_value(vals, q, j), Ferrite.shape_gradient(vals, q, j)
                unit = _unit(Val(NF), b, φ)
                value = mass ? k.f(U0, G0, unit, V, W, X, t, N, p) :
                    k.f(unit, _unit_gradient(Val(NF), b, ∇φ), U0, V, W, X, t, N, p)
                Ae[rows[i], ranges[b][j]] += value * dΩ
            end
        end
    end
    return nothing
end

function _start_cell!(op::NonlinearOp, dofs)
    fill!(op.Je, 0)
    fill!(op.re, 0)
    for (a, d) in enumerate(dofs)
        op.u_elem[a] = op.u[d]
    end
    return nothing
end

function _finish_cell!(op::NonlinearOp, dofs)
    op.aJ === nothing || Ferrite.assemble!(op.aJ, dofs, op.Je)
    Ferrite.assemble!(op.r, dofs, op.re)
    return nothing
end

function _kernel!(op::NonlinearOp, k::KernelSpec, vals, xs, ranges::NTuple{NF}, p, t) where {NF}
    nb = Ferrite.getnbasefunctions(vals)
    rows = ranges[k.eq]
    u_elem = op.u_elem
    Ud0 = ntuple(_ -> 0.0, Val(NF))
    for q in 1:Ferrite.getnquadpoints(vals)
        dΩ = Ferrite.getdetJdV(vals, q)
        X = Ferrite.spatial_coordinate(vals, q, xs)
        N = _normal(vals, q, X)
        DIM = length(X)
        U = ntuple(f -> Ferrite.function_value(vals, q, u_elem, ranges[f]), Val(NF))
        grads = ntuple(f -> Ferrite.function_gradient(vals, q, u_elem, ranges[f]), Val(NF))
        G = ntuple(m -> grads[cld(m, DIM)][mod1(m, DIM)], Val(NF * length(X)))
        for i in 1:nb
            V, W = Ferrite.shape_value(vals, q, i), Ferrite.shape_gradient(vals, q, i)
            op.re[rows[i]] += k.f(U, G, Ud0, V, W, X, t, N, p) * dΩ
            op.aJ === nothing && continue
            # promote: a generated tangent returns literal `0` (Int) for constant-zero entries, and indexing a
            # mixed tuple with a runtime index allocates
            dI = promote(k.tangent(U, G, Ud0, V, W, X, t, N, p)...)
            for b in 1:NF, j in 1:nb
                φ, ∇φ = Ferrite.shape_value(vals, q, j), Ferrite.shape_gradient(vals, q, j)
                value = dI[b] * φ
                for d in 1:DIM
                    value += dI[NF + (b - 1) * DIM + d] * ∇φ[d]
                end
                op.Je[rows[i], ranges[b][j]] += value * dΩ
            end
        end
    end
    return nothing
end
