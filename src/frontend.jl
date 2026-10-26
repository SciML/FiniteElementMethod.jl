# The PDESystem front end (spec §4.6, §5, §7): PDEBase classifies the conditions and analyses the
# strong form, this file slots the trace conditions on mesh pieces, projects with Galerkin, assembles
# with m1's driver and lowers to the numeric DAE path.
#
# PDEBase's generic `symbolic_discretize` cannot drive a mesh discretization yet: it calls
# `parse_bcs(bcs, v, orders)`, which has no mesh method and fails on traces. This driver calls the
# same steps itself, in the same order, until the `AbstractMeshDiscretization` `parse_bcs` lands.

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

# Initial conditions, and `piece => equation` for every boundary condition.
function _split_conditions(bcs, v)
    ics, piecebcs = Equation[], Pair{Any, Equation}[]
    for eq in bcs
        length(_trace_pieces(eq.lhs - eq.rhs)) > 1 && throw(ArgumentError("$(_PERIODIC_M4): $eq"))
        c = PDEBase.classify_condition(v, eq)
        c.kind === :periodic && throw(ArgumentError("$(_PERIODIC_M4): $eq"))
        if c.kind === :initial
            push!(ics, eq)
            continue
        end
        loc = only(c.locations)
        loc isa PDEBase.Piece ||
            throw(ArgumentError("condition $eq is on $loc; on a mesh, write it as a trace on a boundary piece"))
        push!(piecebcs, loc.piece => eq)
    end
    return ics, piecebcs
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

function _space(disc, ctx, analysis, dirichlet, pvals)
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
    return FEMSpace(grid, names .=> ips; dirichlet = bcs, quadrature = disc.quadrature), names, ips
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
    load = isempty(dynamic) ? F : function (out, t)
            assemble_linear!(copy(space.pattern), copy(space.pattern), out, ctx, dynamic, pvals; t)
            out .+= F
            return out
    end
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
    ics, piecebcs = _split_conditions(sys.bcs, v)
    ctx = StrongForm.SlotContext(v)
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
    space, names, ips = _space(disc, ctx, an, dirichlet, pvals)
    t0, t1 = v.intervals[v.time]
    timedep_dirichlet = any(bc -> _depends(bc.g, [an.slots.T]), an.essential)
    ops, coords = _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    u0 = _initial_values(ics, ctx, space, coords, an, names, pvals)
    dirichlet_values!(view(u0, ops.prescribed), ops.load, t0)
    iv = disc.time
    unknowns = [only(@variables $name(iv)[1:length(field_range(space, name))]) for name in names]
    fields = [FieldInfo(dv, name, u, ip) for (dv, name, u, ip) in zip(PDEBase.depvars(v), names, unknowns, ips)]
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

function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    return DAEProblem(pdesys, disc).f.sys
end
