# Generic Ferrite assembly driver for per-quadrature-point kernels (spec §6.5, §6.6).
# One pass over the cells of each SubDofHandler: each cell is reinitialised once, every volume kernel and every
# facet kernel whose piece contains one of the cell's facets adds into the same element scratch, which is assembled
# once. Kernels are iterated as a tuple through recursive calls, so each kernel's loop is compiled for its own type.
# Each field has its own values (spec §6.2): equation `e` is tested with the shape functions of field `e`, and
# the trial functions of field `b` come from the values of field `b`, so fields of different order couple.

"""
    AssemblyContext(dh, fields, cellvalues, facetvalues, pieces = Dict())

What the driver needs to assemble kernels: the `DofHandler`, its field names in equation/field index order, the
`CellValues` and `FacetValues` of each field, and `pieces`, a map from boundary piece to its facet set, given as a
facet-set name of the grid or as a set of `FacetIndex`.

`cellvalues` and `facetvalues` are each a `NamedTuple` keyed by field name, a tuple in field order, or a single
values object shared by all fields. The values of all fields must use one quadrature rule (the same points in the
same order, and weights), since test and trial functions of different fields are paired by quadrature point. A field may be missing
from some `SubDofHandler`s of `dh` (a field on a region): there it has no dofs, its equation is not assembled and
its state slots are zero.

Each facet set is copied at construction, so changing the caller's set or the grid's facet sets afterwards does not
affect the context; build a new one instead.
"""
struct AssemblyContext{NF, DH <: Ferrite.DofHandler, CV <: NamedTuple, FV <: NamedTuple}
    dh::DH
    fields::Vector{Symbol}
    cellvalues::CV
    facetvalues::FV
    facetsets::Dict{Any, Any}
    facetcells::BitSet   # cells with a facet in some piece; the others skip the per-facet membership tests
end

function AssemblyContext(dh::Ferrite.DofHandler, fields, cv, fv, pieces = Dict())
    fields = collect(Symbol, fields)
    names = Tuple(fields)
    cellvalues, facetvalues = _per_field(cv, names, "cell"), _per_field(fv, names, "facet")
    for f in fields
        any(sdh -> f in sdh.field_names, dh.subdofhandlers) ||
            throw(ArgumentError("field $f is not a field of the DofHandler, whose fields are $(dh.field_names)"))
    end
    for sdh in dh.subdofhandlers, f in sdh.field_names
        f in names || continue
        n = length(Ferrite.dof_range(sdh, f))
        for (what, vs) in (("CellValues", cellvalues), ("FacetValues", facetvalues))
            nb = Ferrite.getnbasefunctions(vs[f])
            n == nb || throw(ArgumentError("field $f has $n dofs per cell, but its $what have $nb base functions"))
        end
    end
    _check_rules("CellValues", names, map(cv -> (cv.qr,), cellvalues))
    _check_rules("FacetValues", names, map(fv -> Tuple(fv.fqr.facet_rules), facetvalues))
    facetsets = Dict{Any, Any}()
    for (piece, set) in pieces
        facetsets[piece] = Set{Ferrite.FacetIndex}(set isa AbstractString ? _named_facetset(dh.grid, piece, set) : set)
    end
    ncells = Ferrite.getncells(dh.grid)
    facetcells = BitSet(fi[1] for set in values(facetsets) for fi in set if 1 ≤ fi[1] ≤ ncells)
    NF = length(fields)
    return AssemblyContext{NF, typeof(dh), typeof(cellvalues), typeof(facetvalues)}(dh, fields, cellvalues, facetvalues, facetsets, facetcells)
end

# Test and trial functions of different fields are paired by quadrature point, so every field's values must use the
# same rule: the same reference points in the same order, and the same weights (per local facet for facet rules).
# The rules are read from the values' `qr`/`fqr` fields, which Ferrite has no public accessor for.
function _check_rules(what, names, rules)
    first_rules = first(rules)
    for (f, field_rules) in zip(names, rules)
        for (k, (a, b)) in enumerate(zip(first_rules, field_rules))
            a.points == b.points && a.weights == b.weights && continue
            on = what == "FacetValues" ? " on local facet $k" : ""
            throw(ArgumentError("the $what of fields $(first(names)) and $f use different quadrature rules$on; all fields must share one rule: the same points, in the same order, with the same weights"))
        end
        length(first_rules) == length(field_rules) ||
            throw(ArgumentError("the $what of fields $(first(names)) and $f have rules for $(length(first_rules)) and $(length(field_rules)) facets"))
    end
    return nothing
end

# Values keyed by field: a NamedTuple in field order from a NamedTuple, a tuple, or one shared object.
function _per_field(v::NamedTuple, names, what)
    for f in names
        haskey(v, f) || throw(ArgumentError("no $what values given for field $f; the $what values have keys $(keys(v))"))
    end
    return NamedTuple{names}(map(f -> v[f], names))
end
function _per_field(v::Tuple, names, what)
    length(v) == length(names) ||
        throw(ArgumentError("$(length(v)) $what values given for the $(length(names)) fields $(collect(names))"))
    return NamedTuple{names}(v)
end
_per_field(v, names, what) = NamedTuple{names}(map(_ -> v, names))

# The dofs of `field` in the cells of `sdh`, empty where the field is absent.
_field_range(sdh, field) = field in sdh.field_names ? Ferrite.dof_range(sdh, field) : (1:0)
_max_ndofs(dh) = maximum(Ferrite.ndofs_per_cell, dh.subdofhandlers)

function _named_facetset(grid, piece, name)
    try
        return Ferrite.getfacetset(grid, name)
    catch err
        err isa KeyError && throw(ArgumentError("piece $piece: the grid has no facet set named \"$name\""))
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
function assemble_linear!(M, K, F, ctx::AssemblyContext, kernels, p; t = 0.0, kinds = LINEAR_KINDS)
    ne = _max_ndofs(ctx.dh)
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

function assemble_nonlinear!(r, J, ctx::AssemblyContext, kernels, u, p; t = 0.0)
    ne = _max_ndofs(ctx.dh)
    fill!(r, 0)
    op = NonlinearOp(Ferrite.start_assemble(J), r, zeros(ne, ne), zeros(ne), u, zeros(ne))
    _assemble!(op, ctx, _split(ctx, kernels, (:nonlinear,))..., p, t)
    return r, J
end

struct LinearOp{AM, AK, FV <: AbstractVector{Float64}, ME <: AbstractMatrix{Float64}, FE <: AbstractVector{Float64}}
    aM::AM
    aK::AK
    F::FV
    Me::ME
    Ke::ME
    Fe::FE
end

# `aJ === nothing` assembles the residual only.
struct NonlinearOp{AJ, R <: AbstractVector{Float64}, U, JE <: AbstractMatrix{Float64}, RE <: AbstractVector{Float64}}
    aJ::AJ
    r::R
    Je::JE
    re::RE
    u::U
    u_elem::RE
end

# The op with element scratch of `n` dofs: the buffers themselves when they have that size, else views into them.
_scratch(A::AbstractMatrix, n) = size(A, 1) == n ? A : view(A, 1:n, 1:n)
_scratch(a::AbstractVector, n) = length(a) == n ? a : view(a, 1:n)
_sized(op::LinearOp, n) = LinearOp(op.aM, op.aK, op.F, _scratch(op.Me, n), _scratch(op.Ke, n), _scratch(op.Fe, n))
_sized(op::NonlinearOp, n) = NonlinearOp(op.aJ, op.r, _scratch(op.Je, n), _scratch(op.re, n), op.u, _scratch(op.u_elem, n))

"Volume kernels of `kinds` as a tuple, and one `(facet set, kernel tuple, piece)` group per facet piece."
function _split(ctx::AssemblyContext{NF}, kernels, kinds) where {NF}
    for k in kernels
        k.kind in (:mass, :bilinear, :source, :nonlinear) || throw(ArgumentError("kernel kind $(repr(k.kind)) is not one of :mass, :bilinear, :source, :nonlinear"))
        1 ≤ k.eq ≤ NF || throw(ArgumentError("kernel equation $(k.eq) is outside 1:$NF, the fields of the AssemblyContext"))
        k.kind === :nonlinear && k.tangent === nothing && throw(ArgumentError("nonlinear kernel of equation $(k.eq) has no tangent"))
        k.f === nothing && (k.coeffs === nothing || k.kind === :nonlinear) &&
            throw(ArgumentError("$(k.kind) kernel of equation $(k.eq) has no f: pass f, or coeffs for a linear kind"))
    end
    selected = [k for k in kernels if k.kind in kinds]
    any(_needs_hessians, selected) && _check_hessians(ctx, selected)
    volume = _fuse([k for k in selected if k.piece === nothing])
    pieces = unique(k.piece for k in selected if k.piece !== nothing)
    groups = map(pieces) do piece
        haskey(ctx.facetsets, piece) || throw(ArgumentError("no facet set is given for piece $piece: add it to the pieces of the AssemblyContext"))
        (ctx.facetsets[piece], _fuse([k for k in selected if isequal(k.piece, piece)]), piece)
    end
    return volume, Tuple(groups)
end

# Bilinear `CellCoeffs` contract the shape-function Hessians of the fields in `coeffs.fields`, which Ferrite has only
# when those fields' values were built with `update_hessians = true` (otherwise `shape_hessian` has no method).
_needs_hessians(k) = k.kind === :bilinear && k.coeffs isa CellCoeffs
function _check_hessians(ctx, kernels)
    for k in kernels
        _needs_hessians(k) || continue
        values, what = k.piece === nothing ? (ctx.cellvalues, "CellValues") : (ctx.facetvalues, "FacetValues")
        for b in k.coeffs.fields
            f = ctx.fields[b]
            try
                Ferrite.shape_hessian(values[f], 1, 1)
            catch err
                err isa MethodError || rethrow()
                throw(ArgumentError("a stabilised (SUPG) kernel needs the shape-function Hessians of field $f: build its $what with update_hessians = true"))
            end
        end
    end
    return nothing
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

function _assemble!(op, ctx::AssemblyContext{NF}, volume::Tuple, groups::Tuple, p, t) where {NF}
    counts = zeros(Int, length(groups))
    # each values object once, also when fields share it
    cvs, fvs = values(ctx.cellvalues), values(ctx.facetvalues)
    unique_cvs, unique_fvs = Tuple(unique(objectid, cvs)), Tuple(unique(objectid, fvs))
    for sdh in ctx.dh.subdofhandlers
        ranges = ntuple(f -> _field_range(sdh, ctx.fields[f]), Val(NF))
        sop = _sized(op, Ferrite.ndofs_per_cell(sdh))
        _assemble_cells!(sop, ctx, sdh, cvs, fvs, unique_cvs, unique_fvs, ranges, volume, groups, counts, p, t)
    end
    for (g, (set, _, piece)) in enumerate(groups)
        counts[g] == length(set) ||
            throw(ArgumentError("piece $piece: assembled $(counts[g]) facets, but its facet set has $(length(set)); the set must hold FacetIndex entries of this grid"))
    end
    return nothing
end

# Function barrier: the kernels, values and ranges are concrete from here on.
function _assemble_cells!(op, ctx, sdh, cvs, fvs, unique_cvs, unique_fvs, ranges, volume, groups, counts, p, t)
    for cell in Ferrite.CellIterator(sdh)
        _reinit_all!(unique_cvs, cell)
        _start_cell!(op, Ferrite.celldofs(cell))
        xs = Ferrite.getcoordinates(cell)
        _kernels!(op, volume, cvs, xs, ranges, p, t)
        if !isempty(groups) && Ferrite.cellid(cell) in ctx.facetcells
            for lf in 1:Ferrite.nfacets(cell)
                fi = Ferrite.FacetIndex(Ferrite.cellid(cell), lf)
                _facet_groups!(op, groups, counts, 1, fi, fvs, unique_fvs, xs, ranges, p, t)
            end
        end
        _finish_cell!(op, Ferrite.celldofs(cell))
    end
    return nothing
end

# The loops over fields and values below recurse over tuples and `Val`s with explicit arguments instead of taking
# closures: a closure capturing arrays is heap-allocated whenever the compiler declines to inline it, which costs
# an allocation per quadrature point and shape function.
_reinit_all!(::Tuple{}, args...) = nothing
@inline function _reinit_all!(vs::Tuple, args...)
    Ferrite.reinit!(first(vs), args...)
    return _reinit_all!(Base.tail(vs), args...)
end

# `block!(Val(b), args...)` for b = 1, …, B in order.
@inline _fields!(block!::F, ::Val{0}, args...) where {F} = nothing
@inline function _fields!(block!::F, ::Val{B}, args...) where {F, B}
    _fields!(block!, Val(B - 1), args...)
    block!(Val(B), args...)
    return nothing
end

_facet_groups!(op, ::Tuple{}, counts, g, fi, fvs, unique_fvs, xs, ranges, p, t) = nothing
function _facet_groups!(op, groups::Tuple, counts, g, fi, fvs, unique_fvs, xs, ranges, p, t)
    set, kernels, _ = first(groups)
    if fi in set
        _reinit_all!(unique_fvs, xs, fi[2])
        counts[g] += 1
        _kernels!(op, kernels, fvs, xs, ranges, p, t)
    end
    return _facet_groups!(op, Base.tail(groups), counts, g + 1, fi, fvs, unique_fvs, xs, ranges, p, t)
end

_kernels!(op, ::Tuple{}, vals, xs, ranges, p, t) = nothing
function _kernels!(op, kernels::Tuple, vals, xs, ranges, p, t)
    _fields!(_kernel_on_eq!, Val(length(ranges)), op, first(kernels), vals, xs, ranges, p, t)
    return _kernels!(op, Base.tail(kernels), vals, xs, ranges, p, t)
end

# The kernel's equation `E` as a compile-time constant, so `vals[E]` is concrete when the fields' values differ.
@inline function _kernel_on_eq!(::Val{E}, op, k, vals, xs, ranges, p, t) where {E}
    k.eq == E && _kernel!(op, k, Val(E), vals, xs, ranges, p, t)
    return nothing
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


@inline _add!(Ae, row, col, value, dΩ) = (Ae[row, col] += value * dΩ; nothing)
_add!(Ae, row, col, ::StructuralZero, dΩ) = nothing

# Row `row` of a mass block, trial slots Ud: Σ_j tt[b] φ_j over the trial functions of every field b; nothing for
# an absent block.
_mass_row!(Ae, row, ranges::NTuple{NF}, ::Nothing, vals, q, dΩ) where {NF} = nothing
function _mass_row!(Ae, row, ranges::NTuple{NF}, tt, vals, q, dΩ) where {NF}
    return _fields!(_mass_block!, Val(NF), Ae, row, ranges, tt, vals, q, dΩ)
end
@inline function _mass_block!(::Val{B}, Ae, row, ranges, tt, vals, q, dΩ) where {B}
    return _mass_block!(Ae, row, ranges[B], tt[B], vals[B], q, dΩ)
end
_mass_block!(Ae, row, cols, ::StructuralZero, vb, q, dΩ) = nothing
function _mass_block!(Ae, row, cols, coef, vb, q, dΩ)
    for j in eachindex(cols)
        _add!(Ae, row, cols[j], coef * Ferrite.shape_value(vb, q, j), dΩ)
    end
    return nothing
end

# Row `row` of a state block, trial slots [U; G]: Σ_j (tt[U_b] φ_j + Σ_d tt[G_bd] ∂_d φ_j) over the trial functions
# of every field b.
_state_row!(Ae, row, ranges::NTuple{NF}, ::Nothing, vals, q, dΩ) where {NF} = nothing
function _state_row!(Ae, row, ranges::NTuple{NF}, tt::NTuple{NS, Any}, vals, q, dΩ) where {NF, NS}
    return _fields!(_state_block!, Val(NF), Ae, row, ranges, tt, vals, q, dΩ, Val(NS ÷ NF - 1))
end
@inline function _state_block!(::Val{B}, Ae, row, ranges::NTuple{NF}, tt, vals, q, dΩ, ::Val{DIM}) where {B, NF, DIM}
    tG = ntuple(@inline(d -> tt[NF + (B - 1) * DIM + d]), Val(DIM))
    return _state_block!(Ae, row, ranges[B], tt[B], tG, vals[B], q, dΩ)
end
_state_block!(Ae, row, cols, ::StructuralZero, ::Tuple{Vararg{StructuralZero}}, vb, q, dΩ) = nothing
function _state_block!(Ae, row, cols, tU, tG::NTuple{DIM, Any}, vb, q, dΩ) where {DIM}
    for j in eachindex(cols)
        χ = _test_basis(vb, q, j)
        value = tU * χ[1] + _sum(ntuple(@inline(d -> tG[d] * χ[1 + d]), Val(DIM)))
        _add!(Ae, row, cols[j], value, dΩ)
    end
    return nothing
end

# The block's bilinear coefficient function selects the variant: `CellCoeffs` adds the Hessian trial slots, every
# other one is the [U; G] row above.
_state_row!(Ae, row, ranges, tt, vals, q, dΩ, c, ::Val) = _state_row!(Ae, row, ranges, tt, vals, q, dΩ)
function _state_row!(Ae, row, ranges::NTuple{NF}, tt, vals, q, dΩ, ::CellCoeffs, dim::Val) where {NF}
    return _fields!(_hessian_state_block!, Val(NF), Ae, row, ranges, tt, vals, q, dΩ, dim)
end

# Trial slots [U; G; H] of field B: adds Σ_{d,e} tt[H_Bde] ∂_d∂_e φ_j with field B's shape Hessians, H field-major in
# the slot order of `H`; a field whose Hessian coefficients are all structural zeros never evaluates its Hessians.
@inline function _hessian_state_block!(::Val{B}, Ae, row, ranges::NTuple{NF}, tt, vals, q, dΩ, ::Val{DIM}) where {B, NF, DIM}
    tG = ntuple(@inline(d -> tt[NF + (B - 1) * DIM + d]), Val(DIM))
    tH = ntuple(@inline(m -> tt[NF * (1 + DIM) + (B - 1) * DIM^2 + m]), Val(DIM^2))
    return _hessian_block!(Ae, row, ranges[B], tt[B], tG, tH, vals[B], q, dΩ)
end
function _hessian_block!(Ae, row, cols, tU, tG::NTuple{DIM, Any}, ::Tuple{Vararg{StructuralZero}}, vb, q, dΩ) where {DIM}
    return _state_block!(Ae, row, cols, tU, tG, vb, q, dΩ)
end
function _hessian_block!(Ae, row, cols, tU, tG::NTuple{DIM, Any}, tH, vb, q, dΩ) where {DIM}
    for j in eachindex(cols)
        χ = _test_basis(vb, q, j)
        Hφ = Ferrite.shape_hessian(vb, q, j)
        value = tU * χ[1] + _sum(ntuple(@inline(d -> tG[d] * χ[1 + d]), Val(DIM))) +
            _sum(ntuple(@inline(m -> tH[m] * Hφ[cld(m, DIM), mod1(m, DIM)]), Val(DIM^2)))
        _add!(Ae, row, cols[j], value, dΩ)
    end
    return nothing
end

_coeffs(::Nothing, X, t, N, p, hK) = nothing
_coeffs(c, X, t, N, p, hK) = c(X, t, N, p)
_coeffs(c::CellCoeffs, X, t, N, p, hK) = c.f(X, t, N, p, hK)

# The element size `hK` (the cell diameter) when a coefficient function of the block needs it; otherwise a constant,
# so the Galerkin blocks compute nothing.
_needs_size(c) = c isa CellCoeffs
_cell_size(blk, xs) = _needs_size(blk.mass) || _needs_size(blk.bilinear) || _needs_size(blk.source) ? _diameter(xs) : 0.0
function _diameter(xs)
    h = 0.0
    for i in eachindex(xs), j in (i + 1):lastindex(xs)
        h = max(h, sqrt(sum(abs2, xs[i] - xs[j])))
    end
    return h
end

# Trial slots of a bilinear block: `[U; G]`, or `[U; G; H]` for `CellCoeffs`, with `NW = 1 + dim` test slots.
_state_count(c, ::Val{NF}, ::NTuple{NW}) where {NF, NW} = Val(NF * NW)
_state_count(::CellCoeffs, ::Val{NF}, ::NTuple{NW}) where {NF, NW} = Val(NF * (NW + (NW - 1)^2))
_trial_coeffs(::Nothing, ψ::NTuple{NW}, ::Val{NS}) where {NW, NS} = nothing
_test_contract(::Nothing, ψ::NTuple{NW}) where {NW} = StructuralZero()

function _kernel!(op::LinearOp, blk::LinearBlock, ::Val{E}, vals, xs, ranges::NTuple{NF}, p, t) where {E, NF}
    rows = ranges[E]
    isempty(rows) && return nothing
    ve = vals[E]
    hK = _cell_size(blk, xs)
    for q in 1:Ferrite.getnquadpoints(ve)
        dΩ = Ferrite.getdetJdV(ve, q)
        X = Ferrite.spatial_coordinate(ve, q, xs)
        N = _normal(ve, q, X)
        cM, cK, cF = _coeffs(blk.mass, X, t, N, p, hK), _coeffs(blk.bilinear, X, t, N, p, hK), _coeffs(blk.source, X, t, N, p, hK)
        for i in eachindex(rows)
            ψ = _test_basis(ve, q, i)
            op.Fe[rows[i]] -= _test_contract(cF, ψ) * dΩ
            _mass_row!(op.Me, rows[i], ranges, _trial_coeffs(cM, ψ, Val(NF)), vals, q, dΩ)
            tt = _trial_coeffs(cK, ψ, _state_count(blk.bilinear, Val(NF), ψ))
            _state_row!(op.Ke, rows[i], ranges, tt, vals, q, dΩ, blk.bilinear, Val(length(ψ) - 1))
        end
    end
    return nothing
end

# A kernel without `coeffs`: evaluate `f` with a unit trial function, exact because these kinds are linear in the
# state.
function _kernel!(op::LinearOp, k::KernelSpec, ::Val{E}, vals, xs, ranges::NTuple{NF}, p, t) where {E, NF}
    rows = ranges[E]
    isempty(rows) && return nothing
    ve = vals[E]
    mass = k.kind === :mass
    Ae = mass ? op.Me : op.Ke
    U0 = ntuple(_ -> 0.0, Val(NF))
    for q in 1:Ferrite.getnquadpoints(ve)
        dΩ = Ferrite.getdetJdV(ve, q)
        X = Ferrite.spatial_coordinate(ve, q, xs)
        N = _normal(ve, q, X)
        G0 = _unit_gradient(Val(NF), 0, zero(X))
        for i in eachindex(rows)
            V, W = Ferrite.shape_value(ve, q, i), Ferrite.shape_gradient(ve, q, i)
            if k.kind === :source
                op.Fe[rows[i]] -= k.f(U0, G0, U0, V, W, X, t, N, p) * dΩ
                continue
            end
            _fields!(_unit_trial_block!, Val(NF), Ae, rows[i], ranges, vals, k, mass, (V, W, X, t, N, p), q, dΩ)
        end
    end
    return nothing
end

@inline function _unit_trial_block!(::Val{B}, Ae, row, ranges::NTuple{NF}, vals, k, mass, (V, W, X, t, N, p), q, dΩ) where {B, NF}
    vb, cols = vals[B], ranges[B]
    U0 = ntuple(_ -> 0.0, Val(NF))
    G0 = _unit_gradient(Val(NF), 0, zero(X))
    for j in eachindex(cols)
        φ, ∇φ = Ferrite.shape_value(vb, q, j), Ferrite.shape_gradient(vb, q, j)
        unit = _unit(Val(NF), B, φ)
        value = mass ? k.f(U0, G0, unit, V, W, X, t, N, p) :
            k.f(unit, _unit_gradient(Val(NF), B, ∇φ), U0, V, W, X, t, N, p)
        Ae[row, cols[j]] += value * dΩ
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

# State of each field at a quadrature point: zero where the field has no dofs.
_field_value(vf, q, u, range) = isempty(range) ? 0.0 : Ferrite.function_value(vf, q, u, range)
_field_gradient(vf, q, u, range, X) = isempty(range) ? zero(X) : Ferrite.function_gradient(vf, q, u, range)
_field_values(::Tuple{}, q, u, ::Tuple{}) = ()
@inline function _field_values(vals::Tuple, q, u, ranges::Tuple)
    return (_field_value(first(vals), q, u, first(ranges)), _field_values(Base.tail(vals), q, u, Base.tail(ranges))...)
end
_field_gradients(::Tuple{}, q, u, ::Tuple{}, X) = ()
@inline function _field_gradients(vals::Tuple, q, u, ranges::Tuple, X)
    return (_field_gradient(first(vals), q, u, first(ranges), X), _field_gradients(Base.tail(vals), q, u, Base.tail(ranges), X)...)
end

function _kernel!(op::NonlinearOp, k::KernelSpec, ::Val{E}, vals, xs, ranges::NTuple{NF}, p, t) where {E, NF}
    rows = ranges[E]
    isempty(rows) && return nothing
    ve = vals[E]
    u_elem = op.u_elem
    Ud0 = ntuple(_ -> 0.0, Val(NF))
    for q in 1:Ferrite.getnquadpoints(ve)
        dΩ = Ferrite.getdetJdV(ve, q)
        X = Ferrite.spatial_coordinate(ve, q, xs)
        N = _normal(ve, q, X)
        DIM = length(X)
        U = _field_values(vals, q, u_elem, ranges)
        grads = _field_gradients(vals, q, u_elem, ranges, X)
        G = ntuple(m -> grads[cld(m, DIM)][mod1(m, DIM)], Val(NF * length(X)))
        for i in eachindex(rows)
            V, W = Ferrite.shape_value(ve, q, i), Ferrite.shape_gradient(ve, q, i)
            op.re[rows[i]] += k.f(U, G, Ud0, V, W, X, t, N, p) * dΩ
            op.aJ === nothing && continue
            # promote: a hand-built tangent may return literal `0` (Int) entries, and indexing a mixed tuple with a
            # runtime index allocates
            dI = promote(k.tangent(U, G, Ud0, V, W, X, t, N, p)...)
            _fields!(_tangent_block!, Val(NF), op.Je, rows[i], ranges, vals, dI, q, dΩ)
        end
    end
    return nothing
end

@inline function _tangent_block!(::Val{B}, Je, row, ranges::NTuple{NF}, vals, dI, q, dΩ) where {B, NF}
    vb, cols = vals[B], ranges[B]
    for j in eachindex(cols)
        φ, ∇φ = Ferrite.shape_value(vb, q, j), Ferrite.shape_gradient(vb, q, j)
        DIM = length(∇φ)
        value = dI[B] * φ
        for d in 1:DIM
            value += dI[NF + (B - 1) * DIM + d] * ∇φ[d]
        end
        Je[row, cols[j]] += value * dΩ
    end
    return nothing
end
