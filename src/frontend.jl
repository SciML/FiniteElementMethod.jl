# The PDESystem front end (spec §4.6, §5, §7): PDEBase classifies the conditions and analyses the
# strong form, this file slots the trace conditions on mesh pieces, projects with Galerkin, assembles
# with m1's driver and lowers to the numeric DAE path.
#
# This driver calls PDEBase's steps (make_pdesys_compatible, VariableMap, interface_errors,
# parse_bcs, split_ics_bcs, StrongForm.analyse) itself rather than through the generic
# `symbolic_discretize`, whose remaining hooks build an MTK system this numeric path does not use.
# Tracking: move to PDEBase's `symbolic_discretize` with hook methods once a PDEBase release has
# the mesh `parse_bcs` (PDEBase#152) and piece slotting in StrongForm.

const _PERIODIC_M4 = "periodic conditions need milestone M4 (Ferrite PeriodicDirichlet)"

_unwrap_arg(a) = SymbolicUtils.unwrap_const(unwrap(a))

function _walk(f, ex)
    ex = _unwrap_arg(ex)
    f(ex)
    SymbolicUtils.iscall(ex) && foreach(a -> _walk(f, a), SymbolicUtils.arguments(ex))
    return nothing
end

# Pieces that the traces in `ex` are taken on.
function _trace_pieces(ex)
    pieces = []
    _walk(ex) do ex
        SymbolicUtils.iscall(ex) || return
        for a in SymbolicUtils.arguments(ex)
            a = _unwrap_arg(a)
            a isa PDEBase.BoundaryPiece && push!(pieces, a)
        end
    end
    return unique(pieces)
end

function _condition_pieces(bcs)
    pieces = []
    for eq in bcs
        append!(pieces, _trace_pieces(eq.lhs - eq.rhs))
        _walk(eq.lhs - eq.rhs) do ex
            SymbolicUtils.iscall(ex) && SymbolicUtils.operation(ex) isa PDEBase.NormalDerivative &&
                push!(pieces, SymbolicUtils.operation(ex).piece)
        end
    end
    return unique(pieces)
end

function _check_domains(pdesys, disc::FEMDiscretization)
    pairings = pdesys.domain
    i = findfirst(d -> d.domain isa PDEBase.MeshDomain, pairings)
    i === nothing && throw(
        ArgumentError(
            "the PDESystem has no spatial domain of the form (x, y) ∈ MeshDomain(mesh); " *
                "FEMDiscretization discretizes on a mesh"
        )
    )
    pairings[i].domain === disc.domain ||
        throw(ArgumentError("the PDESystem's MeshDomain is not the domain of the FEMDiscretization"))
    t = disc.time
    t === nothing && return nothing
    j = findfirst(d -> isequal(unwrap(d.variables), unwrap(t)), pairings)
    (j === nothing || !(pairings[j].domain isa DomainSets.AbstractInterval)) &&
        throw(ArgumentError("time $t must be paired with an interval, as in $t ∈ Interval(0.0, 1.0)"))
    return nothing
end

function PDEBase.interface_errors(pdesys::ModelingToolkit.PDESystem, v::PDEBase.VariableMap, disc::FEMDiscretization)
    _check_domains(pdesys, disc)
    grid = disc.domain.mesh
    if !isconcretetype(Ferrite.getcelltype(grid))
        unset = [u for u in PDEBase.depvars(v) if !haskey(disc.spaces, _field_op(u))]
        isempty(unset) || throw(
            ArgumentError(
                "the mesh has mixed cell types, so every dependent variable needs a space in " *
                    "`spaces`; $(join(unset, ", ")) has none"
            )
        )
        throw(ArgumentError("meshes with mixed cell types land in M4"))
    end
    check_facetsets(grid, [Γ.name for Γ in _condition_pieces(pdesys.bcs)])
    return nothing
end

# Initial conditions, and `piece => equation` for every boundary condition in input order.
# PDEBase's mesh `parse_bcs` files the conditions by field and piece; FEM restores the input
# order because the condition listed last wins at a dof shared by two Dirichlet pieces.
# Tracking: use the boundary map's order directly once PDEBase keeps essential conditions in input order.
function _split_conditions(bcs, v, disc)
    for eq in bcs
        length(_trace_pieces(eq.lhs - eq.rhs)) > 1 && throw(ArgumentError("$(_PERIODIC_M4): $eq"))
    end
    orders = Dict(x => PDEBase.d_orders(x, bcs) for x in PDEBase.all_ivs(v))
    boundarymap = PDEBase.parse_bcs(bcs, v, orders, disc)
    ics, bcmap = PDEBase.split_ics_bcs(boundarymap, v, disc)
    position(eq) = findfirst(b -> isequal(b, eq), bcs)
    piecebcs = Pair{Any, Equation}[b.piece => b.eq for bykey in values(bcmap) for bs in values(bykey) for b in bs]
    sort!(piecebcs; by = pb -> position(last(pb)))
    return Equation[b.eq for b in ics], unique(piecebcs)
end

_field_index(ctx, ex) = SymbolicUtils.iscall(ex) ?
    findfirst(op -> isequal(op, SymbolicUtils.operation(ex)), ctx.ops) : nothing

_depends(ex, syms) = any(s -> any(isequal(unwrap(s)), Symbolics.get_variables(ex)), syms)

function _has_op(ex, T)
    found = false
    _walk(e -> (found |= SymbolicUtils.iscall(e) && SymbolicUtils.operation(e) isa T), ex)
    return found
end

# Trace `u_f(t, Γ)` => `U[f]`, `Dn_Γ(u_f(t, Γ))` => `N·G_f` and `Dx_d(u_f(t, Γ))` => `G_f[d]`.
# Tracking: delete `_trace_rules!` and `_slot_piece_condition` once PDEBase.StrongForm slots
# piece traces and `NormalDerivative` (strongform-3 `boundary.jl`, "A NormalDerivative(Γ) method").
function _trace_rules!(rules, ex, ctx, slots, eq)
    SymbolicUtils.iscall(ex) || return rules
    dim = length(ctx.sivs)
    op = SymbolicUtils.operation(ex)
    if op isa PDEBase.NormalDerivative || op isa Differential
        inner = _unwrap_arg(only(SymbolicUtils.arguments(ex)))
        f = _field_index(ctx, inner)
        if f === nothing
            isempty(_trace_pieces(inner)) && return rules
            throw(ArgumentError("in boundary condition $eq, $(Num(ex)) differentiates an expression of traces; apply the derivative to the trace itself"))
        end
        if op isa PDEBase.NormalDerivative
            rules[ex] = sum(slots.N[d] * slots.G[(f - 1) * dim + d] for d in 1:dim)
            return rules
        end
        d = findfirst(x -> isequal(x, op.x), ctx.sivs)
        d === nothing && throw(ArgumentError("time derivative of a trace in boundary condition $eq is not supported"))
        op.order == 1 ||
            throw(ArgumentError("derivative of order $(op.order) of a trace in boundary condition $eq is not supported"))
        rules[ex] = slots.G[(f - 1) * dim + d]
        return rules
    end
    f = _field_index(ctx, ex)
    f === nothing || return (rules[ex] = slots.U[f]; rules)
    foreach(a -> _trace_rules!(rules, _unwrap_arg(a), ctx, slots, eq), SymbolicUtils.arguments(ex))
    return rules
end

_leaf_rules(ctx, slots) = Dict{Any, Any}(
    (x => slots.X[d] for (d, x) in enumerate(ctx.sivs))...,
    (ctx.t === nothing ? () : (ctx.t => slots.T,))...,
    (p => slots.P[j] for (j, p) in enumerate(ctx.params))...,
)

# Boundary residual in slots. Traces are replaced before anything is derivative-expanded, because
# `expand_derivatives(Dx(u(t, Γ)))` is zero (spec §3.3).
function _slot_piece_condition(eq, ctx, slots)
    ex = unwrap(eq.lhs - eq.rhs)
    B = PDEBase.assert_no_traces(PDEBase.substitute_traces(ex, _trace_rules!(Dict{Any, Any}(), ex, ctx, slots, eq)))
    B = substitute(Symbolics.expand_derivatives(Num(B)), _leaf_rules(ctx, slots))
    allowed = Set(unwrap.(vcat(slots.U, slots.G, slots.X, [slots.T], slots.N, slots.P)))
    leftover = filter(s -> !(s in allowed), Symbolics.get_variables(B))
    (isempty(leftover) && !_has_op(B, Differential)) ||
        throw(ArgumentError("unresolved symbols in boundary condition $eq: $(Num(B)); register the derivative of any coefficient function"))
    return B
end

function _with_boundary_conditions(an, piecebcs, ctx)
    s = an.slots
    essential, natural = StrongForm.EssentialBC{Any}[], StrongForm.NaturalBC{Any}[]
    for (Γ, eq) in piecebcs
        B = _slot_piece_condition(eq, ctx, s)
        if _depends(B, vcat(s.G, s.N))
            e, Φ = StrongForm.route_natural(B, an.flux, s.N, s; label = eq)
            push!(natural, StrongForm.NaturalBC{Any}(Γ, e, Φ))
        else
            f, g = StrongForm.essential_condition(B, s; label = eq)
            push!(essential, StrongForm.EssentialBC{Any}(Γ, f, g))
        end
    end
    return StrongForm.StrongFormAnalysis{Any}(
        s, an.nfields, an.dim, an.strong, an.mass, an.flux, an.remainder, an.conormal,
        essential, natural, StrongForm.PeriodicBC{Any}[]
    )
end

# Reorder the equations so that equation `f` is the one tested with field `f` (spec §5.4).
function _paired(an, eqs, ctx, disc)
    an.nfields == length(eqs) ||
        throw(ArgumentError("the system has $(length(eqs)) equations for $(an.nfields) dependent variables"))
    explicit = Dict{Int, Int}()
    for (eq, dv) in disc.pairing
        e = findfirst(x -> isequal(x, eq), eqs)
        e === nothing && throw(ArgumentError("pairing key $eq is not an equation of the system"))
        f = findfirst(op -> isequal(op, _field_op(dv)), ctx.ops)
        f === nothing && throw(ArgumentError("pairing value $dv is not a dependent variable of the system"))
        explicit[e] = f
    end
    order = invperm(StrongForm.pair_equations(an.mass, an.slots; pairing = explicit))
    return StrongForm.StrongFormAnalysis{Any}(
        an.slots, an.nfields, an.dim, an.strong[order], an.mass[order], an.flux[order],
        an.remainder[order], an.conormal[order], an.essential, an.natural, an.periodic
    )
end

function _check_linear(wf)
    for term in vcat(wf.volume, wf.facet)
        term.kind === :nonlinear &&
            throw(ArgumentError("nonlinear terms land in M3: equation $(term.eq) has the nonlinear term $(term.integrand)"))
        term.kind in (:mass, :bilinear) && term.timedep && throw(
            ArgumentError("time-dependent coefficients in the mass or stiffness operator land in M3: equation $(term.eq) has $(term.integrand)")
        )
    end
    return nothing
end

function _parameter_values(pdesys, ctx)
    given = Dict{Any, Any}(unwrap(p.first) => p.second for p in pdesys.ps if p isa Pair)
    return map(ctx.params) do p
        haskey(given, p) && given[p] isa Number ||
            throw(ArgumentError("parameter $p needs a numeric value in the PDESystem, as $p => value"))
        Float64(given[p])
    end
end

function _space(disc, ctx, analysis, dirichlet, pvals, t0)
    grid = disc.domain.mesh
    shape = Ferrite.getrefshape(Ferrite.getcelltype(grid))
    names = Symbol[nameof(op) for op in ctx.ops]
    ips = [get(disc.spaces, op, Ferrite.Lagrange{shape, 1}()) for op in ctx.ops]
    allequal(ips) || throw(ArgumentError("fields with different interpolations land in M4; got $(join(ips, ", "))"))
    Ferrite.getrefshape(first(ips)) == shape ||
        throw(ArgumentError("the interpolation $(first(ips)) is not on the reference shape $shape of the mesh cells"))
    bcs = [
        (names[bc.field], String(bc.piece.name), (x, t) -> g(x, t, pvals))
            for (bc, (_, _, g)) in zip(analysis.essential, dirichlet)
    ]
    return FEMSpace(grid, names .=> ips; dirichlet = bcs, quadrature = disc.quadrature, t0), names, ips
end

# Nodal coordinates of every dof; Lagrange interpolation of `x_d` gives them exactly.
function _dof_coordinates(space)
    dim = _meshdim(space.grid)
    comps = map(1:dim) do d
        a = zeros(Ferrite.ndofs(space.dh))
        for name in space.fields
            Ferrite.apply_analytical!(a, space.dh, name, x -> x[d])
        end
        a
    end
    return [Ferrite.Vec{dim}(ntuple(d -> comps[d][i], dim)) for i in 1:Ferrite.ndofs(space.dh)]
end

# The Dirichlet values at `t` are evaluated from the symbolic data rather than through the
# ConstraintHandler so that `dirichlet_rates` can differentiate them; like Ferrite, the condition
# added last wins on a dof shared by two pieces.
function _dirichlet_values(space, dirichlet, coords, pvals, t0)
    owner = zeros(Int, Ferrite.ndofs(space.dh))
    for (i, (Γ, f, _)) in enumerate(dirichlet)
        ch = Ferrite.ConstraintHandler(space.dh)
        Ferrite.add!(ch, Ferrite.Dirichlet(space.fields[f], Ferrite.getfacetset(space.grid, String(Γ.name)), (x, t) -> 0.0))
        Ferrite.close!(ch)
        owner[ch.prescribed_dofs] .= i
    end
    gs = [g for (_, _, g) in dirichlet]
    dofs = prescribed_dofs(space)
    values!(out, t) = (foreach(((k, d),) -> out[k] = gs[owner[d]](coords[d], t, pvals), enumerate(dofs)); out)
    return values!, values!(zeros(length(dofs)), t0)
end

# Source-only assembly through m1's cell loop: the time-dependent load is re-evaluated at every
# residual call, where assembling zero M and K blocks would dominate.
struct SourceOp
    F::Vector{Float64}
    Fe::Vector{Float64}
end

_start_cell!(op::SourceOp, dofs) = (fill!(op.Fe, 0); nothing)
_finish_cell!(op::SourceOp, dofs) = (Ferrite.assemble!(op.F, dofs, op.Fe); nothing)

function _kernel!(op::SourceOp, k::KernelSpec, vals, xs, ranges::NTuple{NF}, p, t) where {NF}
    rows = ranges[k.eq]
    U0 = ntuple(_ -> 0.0, Val(NF))
    for q in 1:Ferrite.getnquadpoints(vals)
        dΩ = Ferrite.getdetJdV(vals, q)
        X = Ferrite.spatial_coordinate(vals, q, xs)
        G0 = _unit_gradient(Val(NF), 0, zero(X))
        N = _normal(vals, q, X)
        for i in 1:Ferrite.getnbasefunctions(vals)
            V, W = Ferrite.shape_value(vals, q, i), Ferrite.shape_gradient(vals, q, i)
            op.Fe[rows[i]] -= k.f(U0, G0, U0, V, W, X, t, N, p) * dΩ
        end
    end
    return nothing
end

"Write `F = −∫source` of the `:source` kernels at time `t` into `F`."
function assemble_sources!(F, ctx::AssemblyContext{NF}, kernels, p, t) where {NF}
    fill!(F, 0)
    op = SourceOp(F, zeros(NF * Ferrite.getnbasefunctions(ctx.cv)))
    _assemble!(op, ctx, _split(ctx, kernels, (:source,))..., p, t)
    return F
end

# `reinit!` mutates CellValues and FacetValues, so each load evaluation gets its own copies and
# concurrent solves of one problem's remakes share no mutable state.
function _fresh_values(ctx::AssemblyContext{NF, DH, CV, FV}) where {NF, DH, CV, FV}
    return AssemblyContext{NF, DH, CV, FV}(ctx.dh, ctx.fields, copy(ctx.cv), copy(ctx.fv), ctx.facetsets)
end

function _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    kernels = emit_kernels(wf)
    timedep = [term.kind === :source && term.timedep for term in vcat(wf.volume, wf.facet)]
    static = Tuple(k for (k, td) in zip(kernels, timedep) if !td)
    dynamic = Tuple(k for (k, td) in zip(kernels, timedep) if td)
    pieces = Dict(bc.piece => space.pieces[bc.piece.name] for bc in wf.analysis.natural)
    ctx = AssemblyContext(space.dh, space.fields, first(space.cellvalues), first(space.facetvalues), pieces)
    n = Ferrite.ndofs(space.dh)
    M, K, F = copy(space.pattern), copy(space.pattern), zeros(n)
    assemble_linear!(M, K, F, ctx, static, pvals; t = t0)
    load = isempty(dynamic) ? F : (out, t) -> (assemble_sources!(out, _fresh_values(ctx), dynamic, pvals, t); out .+= F)
    coords = _dof_coordinates(space)
    values!, values0 = _dirichlet_values(space, dirichlet, coords, pvals, t0)
    ops = FEMOperators(
        M, K, load, [name => field_range(space, name) for name in names];
        prescribed = prescribed_dofs(space), dirichlet_values = timedep_dirichlet ? values! : values0
    )
    return ops, coords
end

function _initial_values(ics, ctx, space, coords, an, names, pvals)
    u0 = zeros(Ferrite.ndofs(space.dh))
    given = falses(length(names))
    params = Dict{Any, Any}(p => v for (p, v) in zip(ctx.params, pvals))
    for ic in ics
        ex = unwrap(ic.lhs - ic.rhs)
        calls = Any[]
        _walk(e -> _field_index(ctx, e) === nothing || push!(calls, e), ex)
        length(unique(calls)) == 1 ||
            throw(ArgumentError("initial condition $ic must give the initial value of one dependent variable"))
        call = first(calls)
        f = _field_index(ctx, call)
        z = Symbolics.variable(gensym(:ic))
        r = substitute(Num(ex), Dict(call => z))
        a = Symbolics.derivative(r, z)
        (_depends(a, [z]) || isequal(a, 0)) && throw(ArgumentError("initial condition $ic must be linear in $(Num(call))"))
        value = substitute(-substitute(r, Dict(z => 0)) / a, params)
        fun = Symbolics.build_function(value, ctx.sivs...; expression = Val(false))
        for d in field_range(space, names[f])
            u0[d] = fun(coords[d]...)
            isfinite(u0[d]) ||
                throw(ArgumentError("initial condition $ic is not finite at the node coordinate $(Tuple(coords[d])): $(u0[d])"))
        end
        given[f] = true
    end
    for (f, op) in enumerate(ctx.ops)
        given[f] || !any(m -> _depends(m, [an.slots.Ud[f]]), an.mass) ||
            throw(ArgumentError("$op has a time derivative but no initial condition"))
    end
    return u0
end

function _lower(pdesys, disc::FEMDiscretization)
    disc.time === nothing && throw(ArgumentError("steady problems land in M4; give FEMDiscretization a time variable"))
    _check_domains(pdesys, disc)
    sys, replaced = PDEBase.make_pdesys_compatible(pdesys)
    v = PDEBase.VariableMap(sys, disc; replaced_vars = replaced)
    PDEBase.interface_errors(sys, v, disc)
    ics, piecebcs = _split_conditions(sys.bcs, v, disc)
    ctx = StrongForm.SlotContext(sys, v)
    an = StrongForm.analyse(sys, v, Dict(); ibp = disc.ibp)
    an = _with_boundary_conditions(_paired(an, pdesys.eqs, ctx, disc), piecebcs, ctx)
    isempty(an.periodic) || throw(ArgumentError(_PERIODIC_M4))
    covered = Set(first.(piecebcs))
    if any(c -> !all(x -> isequal(x, 0), c), an.conormal)
        for Γ in PDEBase.boundarypieces(disc.domain)
            Γ in covered || @warn "boundary piece $Γ has no condition, so it gets the natural zero-flux condition"
        end
    end
    wf = weak_form(an)
    _check_linear(wf)
    pvals = _parameter_values(sys, ctx)
    dirichlet = emit_dirichlet(an)
    t0, t1 = v.intervals[v.time]
    space, names, ips = _space(disc, ctx, an, dirichlet, pvals, t0)
    timedep_dirichlet = any(bc -> _depends(bc.g, [an.slots.T]), an.essential)
    ops, coords = _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    u0 = _initial_values(ics, ctx, space, coords, an, names, pvals)
    dirichlet_values!(view(u0, ops.prescribed), ops.load, t0)
    iv = disc.time
    unknowns = [only(@variables $name(iv)[1:length(field_range(space, name))]) for name in names]
    fields = [FieldInfo(dv, name, u, ip) for (dv, name, u, ip) in zip(StrongForm.declared_fields(sys, v), names, unknowns, ips)]
    pieces = Dict{Any, String}(Γ => String(Γ.name) for Γ in PDEBase.boundarypieces(disc.domain))
    md = FEMMetadata(space, iv, v.x̄, fields, pieces, pdesys; u0)
    return (; ops, u0, tspan = (t0, t1), md, iv, unknowns, name = nameof(pdesys))
end

"""
    DAEProblem(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Discretize `pdesys` with `disc` and return the `DAEProblem` `M du + K u = F(t)` of the
Galerkin operators, with Dirichlet rows `u_d = g_d(t)` as algebraic equations, a sparse
analytic Jacobian and the `FEMMetadata` as its `problem_type`, so `solve` returns a
`PDETimeSeriesSolution`. Nothing calls `mtkcompile`; anything this milestone cannot
discretize raises. Keywords (after `disc.kwargs`) are passed to the `DAEProblem`.

The Dirichlet rows are linear with identity Jacobian rows, so a Newton step with a current
Jacobian satisfies them exactly. OrdinaryDiffEq's BDF methods reuse the Jacobian across
step-size changes and then rescale the Newton step, which leaves these rows off by about the
solver tolerance; `DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0))` refreshes it on every step
size change and holds `u(t, Γ) ~ g(t)` to round-off.
"""
function SciMLBase.DAEProblem(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    parts = _lower(pdesys, disc)
    return numeric_dae_problem(
        parts.ops, parts.u0, parts.tspan, parts.md;
        iv = parts.iv, unknowns = parts.unknowns, name = parts.name, disc.kwargs..., kwargs...
    )
end

SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...) =
    DAEProblem(pdesys, disc; kwargs...)

# `(sys, tspan)`, the shape PDEBase and MethodOfLines return.
function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    prob = DAEProblem(pdesys, disc)
    return prob.f.sys, prob.tspan
end
