# The PDESystem front end (spec §4.6, §5, §7): PDEBase classifies the conditions and analyses the
# strong form with its boundary conditions; this file projects with Galerkin, assembles with m1's
# driver and lowers to the numeric DAE path.
#
# This driver calls PDEBase's steps (make_pdesys_compatible, VariableMap, interface_errors,
# parse_bcs, split_ics_bcs, StrongForm.analyse) itself rather than through the generic
# `symbolic_discretize`, whose remaining hooks build an MTK system this numeric path does not use.
# Tracking: move to PDEBase's `symbolic_discretize` with hook methods once the numeric path builds
# its system through `generate_system` (PDEBase M0 release as the compat floor).

const _PERIODIC_UNSUPPORTED = "periodic boundary conditions are not supported"

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
    if t === nothing
        j = findfirst(d -> d.domain isa DomainSets.AbstractInterval, pairings)
        j === nothing || throw(
            ArgumentError(
                "the PDESystem pairs $(pairings[j].variables) with an interval, but the FEMDiscretization has no time " *
                    "variable; for a time-dependent problem write FEMDiscretization(Ω, $(pairings[j].variables))"
            )
        )
        return nothing
    end
    j = findfirst(d -> isequal(unwrap(d.variables), unwrap(t)), pairings)
    (j === nothing || !(pairings[j].domain isa DomainSets.AbstractInterval)) &&
        throw(ArgumentError("time $t must be paired with an interval, as in $t ∈ Interval(0.0, 1.0)"))
    return nothing
end

function PDEBase.interface_errors(pdesys::ModelingToolkit.PDESystem, v::PDEBase.VariableMap, disc::FEMDiscretization)
    _check_domains(pdesys, disc)
    dvs = PDEBase.depvars(v)
    ops = [_field_op(u) for u in dvs]
    for (kw, dict) in (("spaces", disc.spaces), ("regions", disc.regions))
        unknown = [k for k in keys(dict) if !any(op -> isequal(op, k), ops)]
        isempty(unknown) || throw(
            ArgumentError(
                "`$kw` has keys that are not dependent variables of the system: " *
                    "$(join(sort(string.(unknown)), ", ")); the dependent variables are $(join(sort(string.(dvs)), ", "))"
            )
        )
    end
    t = v.time
    for u in dvs
        args = SymbolicUtils.arguments(unwrap(u))
        t !== nothing && any(a -> isequal(a, t), args[2:(end - 1)]) &&
            throw(ArgumentError("time must be the first or the last argument of $u, as in u(t, x, y) or u(x, y, t)"))
    end
    grid = disc.domain.mesh
    if !isconcretetype(Ferrite.getcelltype(grid))
        unset = [u for u in PDEBase.depvars(v) if !haskey(disc.spaces, _field_op(u))]
        isempty(unset) || throw(
            ArgumentError(
                "the mesh has mixed cell types, so every dependent variable needs a space in " *
                    "`spaces`; $(join(unset, ", ")) has none"
            )
        )
        throw(ArgumentError("meshes with mixed cell types are not supported; use a mesh with a single cell type"))
    end
    check_facetsets(grid, [Γ.name for Γ in _condition_pieces(pdesys.bcs)])
    return nothing
end

function _boundary_map(bcs, v, disc)
    for eq in bcs
        length(_trace_pieces(eq.lhs - eq.rhs)) > 1 && throw(ArgumentError("$(_PERIODIC_UNSUPPORTED): $eq"))
    end
    orders = Dict(x => PDEBase.d_orders(x, bcs) for x in PDEBase.all_ivs(v))
    return PDEBase.parse_bcs(bcs, v, orders, disc)
end

_field_index(ctx, ex) = SymbolicUtils.iscall(ex) ?
    findfirst(op -> isequal(op, SymbolicUtils.operation(ex)), ctx.ops) : nothing

_depends(ex, syms) = any(s -> any(isequal(unwrap(s)), Symbolics.get_variables(ex)), syms)

# Reorder the equations so that equation `f` is the one tested with field `f` (spec §5.4); a
# natural condition routed to equation `e` moves with it.
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
    # A steady system has no time derivatives to pair by, so the unnamed equations take the
    # unnamed fields in declared order.
    if disc.time === nothing
        free = [f for f in eachindex(ctx.ops) if !(f in values(explicit))]
        for e in eachindex(eqs)
            haskey(explicit, e) || (explicit[e] = popfirst!(free))
        end
    end
    fields = StrongForm.pair_equations(an.mass, an.slots; pairing = explicit)
    order = invperm(fields)
    natural = [typeof(bc)(bc.piece, fields[bc.eq], bc.Φ) for bc in an.natural]
    return typeof(an)(
        an.slots, an.nfields, an.dim, an.strong[order], an.mass[order], an.flux[order],
        an.remainder[order], an.conormal[order], an.essential, natural, an.periodic
    )
end

# A SUPG whose τ and b name the coordinates and time of the PDESystem, rewritten onto the coordinate and time slots the
# kernels evaluate; a `b` read from the equation is in slots already.
function _slotted(space::SUPG, ctx, slots)
    ivs = ctx.t === nothing ? ctx.sivs : [ctx.sivs; ctx.t]
    to_slots = Dict(unwrap(iv) => unwrap(s) for (iv, s) in zip(ivs, ctx.t === nothing ? slots.X : [slots.X; slots.T]))
    slot(ex) = substitute(ex, to_slots)
    τ = space.τ isa Function ? (hK, b) -> slot(space.τ(hK, b)) : slot(space.τ)
    return SUPG(τ, space.b === nothing ? nothing : map(slot, space.b))
end

# τ and b are compiled into the kernels as functions of the coordinates, time and hK only, so they cannot name a
# parameter, whose value the kernels would not see.
function _check_supg(test_space, an, ctx)
    test = TestSlots(an.dim)
    for e in eachindex(an.flux)
        weight = test_weight(test_space, an, e, test)
        weight === nothing && continue
        τ, bW = weight
        any(ex -> _depends(unwrap(Num(ex)), [ctx.params; an.slots.P]), (τ, bW)) && throw(
            ArgumentError(
                "SUPG: equation $e has τ = $τ and b·∇v = $bW, which reference parameters of the PDESystem; " *
                    "τ and b must be numbers or expressions in the coordinates, time and hK, so write the parameter's value"
            )
        )
    end
    return nothing
end

function _check_terms(wf)
    for term in vcat(wf.volume, wf.facet)
        term.kind in (:mass, :bilinear) && term.timedep && throw(
            ArgumentError("time-dependent coefficients in the mass or stiffness operator are not supported: equation $(term.eq) has $(term.integrand)")
        )
    end
    return nothing
end

function _parameter_values(pdesys, ctx)
    ps = pdesys.ps isa SciMLBase.NullParameters ? () : pdesys.ps
    given = Dict{Any, Any}(unwrap(p.first) => p.second for p in ps if p isa Pair)
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
    for ip in ips
        Ferrite.getrefshape(ip) == shape ||
            throw(ArgumentError("the interpolation $ip is not on the reference shape $shape of the mesh cells"))
    end
    bcs = [
        (names[bc.field], String(bc.piece.name), (x, t) -> g(x, t, pvals))
            for (bc, (_, _, g)) in zip(analysis.essential, dirichlet)
    ]
    regions = Dict(names[f] => String(disc.regions[op].name) for (f, op) in enumerate(ctx.ops) if haskey(disc.regions, op))
    _check_region_pieces(grid, regions, names, analysis)
    space = FEMSpace(
        grid, names .=> ips; dirichlet = bcs, quadrature = disc.quadrature, t0, regions,
        hessians = disc.test_space isa SUPG
    )
    return space, names, ips
end

# A condition on a field restricted to a region must lie on the region: Ferrite would otherwise drop the facets
# outside it without a word.
function _check_region_pieces(grid, regions, names, analysis)
    conditions = [[(bc.field, bc.piece) for bc in analysis.essential]; [(bc.eq, bc.piece) for bc in analysis.natural]]
    for (f, Γ) in conditions
        haskey(regions, names[f]) || continue
        cells = Ferrite.getcellset(grid, regions[names[f]])
        all(fi -> fi[1] in cells, Ferrite.getfacetset(grid, String(Γ.name))) || throw(
            ArgumentError(
                "the condition on $(names[f]) on $Γ has facets outside the region :$(regions[names[f]]) of $(names[f]); " *
                    "a field on a region takes conditions only on pieces whose facets belong to the region's cells"
            )
        )
    end
    return nothing
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
# residual call, where assembling zero M and K blocks would dominate. A `LinearOp` without M and K
# assemblers only scatters `Fe`.
_finish_cell!(op::LinearOp{Nothing, Nothing}, dofs) = (Ferrite.assemble!(op.F, dofs, op.Fe); nothing)

"Write `F = −∫source` of the `:source` kernels at time `t` into `F`."
function assemble_sources!(F, ctx::AssemblyContext, kernels, p, t)
    fill!(F, 0)
    ne = _max_ndofs(ctx.dh)
    op = LinearOp(nothing, nothing, F, zeros(ne, ne), zeros(ne, ne), zeros(eltype(F), ne))
    _assemble!(op, ctx, _split(ctx, kernels, (:source,))..., p, t)
    return F
end

function _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    kernels = emit_kernels(wf)
    # `emit_kernels` sums the terms sharing an equation, kind and piece, so a kernel is dynamic when any of its terms is.
    terms = [[(t, nothing) for t in wf.volume]; [(t, t.piece) for t in wf.facet]]
    dynamic_keys = Set((t.eq, t.kind, piece) for (t, piece) in terms if t.kind === :source && t.timedep)
    timedep = [(eq, kind, piece) in dynamic_keys for (eq, kind, piece, _) in fused_terms(wf)]
    static = Tuple(k for (k, td) in zip(kernels, timedep) if !td)
    dynamic = Tuple(k for (k, td) in zip(kernels, timedep) if td)
    pieces = Dict(bc.piece => space.pieces[bc.piece.name] for bc in wf.analysis.natural)
    ctx = AssemblyContext(space.dh, space.fields, space.cellvalues, space.facetvalues, pieces)
    n = Ferrite.ndofs(space.dh)
    M, K, F = copy(space.pattern), copy(space.pattern), zeros(n)
    assemble_linear!(M, K, F, ctx, static, pvals; t = t0)
    load = isempty(dynamic) ? F : (out, t) -> (assemble_sources!(out, _fresh_values(ctx), dynamic, pvals, t); out .+= F)
    nonlinear = any(k -> k.kind === :nonlinear, kernels) ?
        FEMNonlinear(NonlinearOperator(_fresh_values(ctx), kernels; ch = space.ch), pvals) : nothing
    coords = _dof_coordinates(space)
    values!, values0 = _dirichlet_values(space, dirichlet, coords, pvals, t0)
    ops = FEMOperators(
        M, K, load, [name => field_range(space, name) for name in names];
        prescribed = prescribed_dofs(space), dirichlet_values = timedep_dirichlet ? values! : values0, nonlinear
    )
    raw = (; M, K, load, nonlinear = nonlinear !== nothing, prescribed = prescribed_dofs(space), dirichlet_values = values0, timedep_dirichlet)
    return ops, coords, raw
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

# Boundary facets that carry no condition for a field whose equation has a flux get the natural zero-flux condition
# silently, which hides a forgotten piece. A piece is covered for a field by a Dirichlet condition on that field, a
# Neumann or Robin condition in its equation, or an entry in `disc.natural`.
function _check_boundary_conditions(disc::FEMDiscretization, an, ctx)
    disc.boundary_check === :off && return nothing
    grid = disc.domain.mesh
    boundary = _boundary_facets(grid)
    facetsets = Dict(name => Set(Ferrite.getfacetset(grid, String(name))) for name in _facetset_names(grid))
    lines = String[]
    for (f, op) in enumerate(ctx.ops)
        all(c -> isequal(c, 0), an.conormal[f]) && continue
        covered = Set{Ferrite.FacetIndex}()
        for name in [disc.natural; [bc.piece.name for bc in an.essential if bc.field == f]; [bc.piece.name for bc in an.natural if bc.eq == f]]
            union!(covered, facetsets[name])
        end
        cells = haskey(disc.regions, op) ? Set(Ferrite.getcellset(grid, String(disc.regions[op].name))) : nothing
        open = Set(fi for fi in boundary if fi ∉ covered && (cells === nothing || fi[1] in cells))
        isempty(open) && continue
        pieces = [(name, length(intersect(open, set))) for (name, set) in sort!(collect(facetsets); by = first)]
        listed = ["Γ_$name ($n facets)" for (name, n) in pieces if n > 0]
        unnamed = length(setdiff(open, union(Set{Ferrite.FacetIndex}(), values(facetsets)...)))
        unnamed > 0 && push!(listed, "$unnamed facets in no piece")
        push!(lines, "  $(nameof(op)): $(length(open)) of $(length(boundary)) boundary facets have no condition: $(join(listed, ", "))")
    end
    isempty(lines) && return nothing
    msg = "boundary facets with no condition get the natural zero-flux condition:\n" * join(lines, "\n") *
        "\nDeclare them with `FEMDiscretization(…; natural = [pieces])` or a zero Neumann condition, or set `boundary_check = :off`"
    disc.boundary_check === :error && throw(ArgumentError(msg))
    @warn msg
    return nothing
end

function _lower(pdesys, disc::FEMDiscretization)
    _check_domains(pdesys, disc)
    sys, replaced = PDEBase.make_pdesys_compatible(pdesys)
    v = PDEBase.VariableMap(sys, disc; replaced_vars = replaced)
    PDEBase.interface_errors(sys, v, disc)
    boundarymap = _boundary_map(sys.bcs, v, disc)
    ics, _ = PDEBase.split_ics_bcs(boundarymap, v, disc)
    ctx = StrongForm.SlotContext(sys, v)
    an = _paired(StrongForm.analyse(sys, v, boundarymap; ibp = disc.ibp), pdesys.eqs, ctx, disc)
    isempty(an.periodic) || throw(ArgumentError(_PERIODIC_UNSUPPORTED))
    test_space = disc.test_space isa SUPG ? _slotted(disc.test_space, ctx, an.slots) : disc.test_space
    test_space isa SUPG && _check_supg(test_space, an, ctx)
    wf = weak_form(an; test_space)
    steady = disc.time === nothing
    _check_terms(wf)
    disc.test_space isa SUPG && any(term -> term.kind === :nonlinear, wf.volume) && throw(
        ArgumentError("SUPG stabilises linear terms only, but the system has a nonlinear term; use test_space = Galerkin()")
    )
    pvals = _parameter_values(sys, ctx)
    dirichlet = emit_dirichlet(an)
    t0, t1 = steady ? (0.0, 0.0) : v.intervals[v.time]
    space, names, ips = _space(disc, ctx, an, dirichlet, pvals, t0)
    _check_boundary_conditions(disc, an, ctx)
    timedep = findfirst(bc -> _depends(bc.g, [an.slots.T]), an.essential)
    disc.dirichlet === :eliminate && timedep !== nothing && throw(
        ArgumentError(
            "dirichlet = :eliminate needs Dirichlet data that are constant in time, but the condition on boundary piece " *
                "$(an.essential[timedep].piece.name) depends on time; use the default dirichlet = :algebraic"
        )
    )
    ops, coords, raw = _operators(space, wf, names, dirichlet, pvals, t0, timedep !== nothing)
    u0 = _initial_values(Equation[b.eq for b in ics], ctx, space, coords, an, names, pvals)
    dirichlet_values!(view(u0, ops.prescribed), ops.load, t0)
    elimination = nothing
    unknown_u0 = u0
    if disc.dirichlet === :eliminate
        ops, elimination = eliminate_dirichlet(ops)
        unknown_u0 = u0[elimination.free]
    end
    iv = disc.time
    unknowns = map(ops.fields) do (name, r)
        steady ? only(@variables $name[1:length(r)]) : only(@variables $name(iv)[1:length(r)])
    end
    unknown_of = Dict(zip(first.(ops.fields), unknowns))
    fields = [
        FieldInfo(dv, name, get(unknown_of, name, nothing), ip)
            for (dv, name, ip) in zip(StrongForm.declared_fields(sys, v), names, ips)
    ]
    pieces = Dict{Any, String}(Γ => String(Γ.name) for Γ in PDEBase.boundarypieces(disc.domain))
    md = FEMMetadata(space, iv, v.x̄, fields, pieces, pdesys; u0, elimination)
    parameters = [q => v for (q, v) in zip(ctx.params, pvals)]
    selfnonlinear = Set(names[f] for f in 1:an.nfields if _nonlinear_in_own_field(wf, an, f))
    return (; ops, u0 = unknown_u0, tspan = (t0, t1), md, iv, unknowns, name = nameof(pdesys), parameters, selfnonlinear, raw, full_u0 = u0)
end

# The parameters of a steady problem as a symbolic container, so `remake(prob; p = [k => 3.0])` resolves `k` to its
# slot in `p` and reaches the same check as `remake(prob; p = [3.0])`, instead of leaving `p` unchanged.
struct SteadyParameters
    syms::Vector{Any}
end

SymbolicIndexingInterface.parameter_symbols(sys::SteadyParameters) = sys.syms
SymbolicIndexingInterface.is_parameter(sys::SteadyParameters, sym) = parameter_index(sys, sym) !== nothing
SymbolicIndexingInterface.parameter_index(sys::SteadyParameters, sym) = findfirst(q -> isequal(q, sym), sys.syms)
SymbolicIndexingInterface.variable_symbols(::SteadyParameters) = []
SymbolicIndexingInterface.is_variable(::SteadyParameters, sym) = false
SymbolicIndexingInterface.variable_index(::SteadyParameters, sym) = nothing
SymbolicIndexingInterface.is_independent_variable(::SteadyParameters, sym) = false
SymbolicIndexingInterface.independent_variable_symbols(::SteadyParameters) = []
SymbolicIndexingInterface.is_time_dependent(::SteadyParameters) = false
SymbolicIndexingInterface.is_observed(::SteadyParameters, sym) = false
SymbolicIndexingInterface.constant_structure(::SteadyParameters) = true

# `K u + N(u, t0) - F` with the Dirichlet rows `u_d - g_d`: `K` holds `e_d`, `F` holds `g_d` and `N` is zero there.
function _steady_problem(parts; kwargs...)
    (; ops) = parts
    t0 = first(parts.tspan)
    F = load!(zeros(length(parts.u0)), ops.load, t0)
    N = ops.nonlinear
    syms = [q for (q, _) in parts.parameters]
    assembled = [v for (_, v) in parts.parameters]
    check_parameters(p) = isempty(syms) || check_assembled_values(syms, assembled, p)
    function f!(res, u, p)
        check_parameters(p)
        mul!(res, ops.K, u)
        res .-= F
        N === nothing || add_nonlinear!(res, N, u, t0)
        return nothing
    end
    function jac!(J, u, p)
        check_parameters(p)
        check_pattern(J, ops.K, "the Jacobian")
        copyto!(nonzeros(J), nonzeros(ops.K))
        N === nothing || add_nonlinear_jacobian!(J, N, u, t0)
        return nothing
    end
    f = SciMLBase.NonlinearFunction{true}(f!; jac = jac!, jac_prototype = copy(ops.K), sys = isempty(syms) ? nothing : SteadyParameters(syms))
    return SciMLBase.NonlinearProblem{true}(f, parts.u0, isempty(syms) ? SciMLBase.NullParameters() : copy(assembled), parts.md; kwargs...)
end

"""
    discretize(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Discretize `pdesys` with `disc` and return the `DAEProblem` `M du + K u + N(u, t) = F(t)` of
the Galerkin operators, with Dirichlet rows `u_d = g_d(t)` as algebraic equations (or, with
`dirichlet = :eliminate`, restricted to the free dofs), a sparse
analytic Jacobian (the nonlinear part `N` from the terms whose coefficients depend on the
state, with its symbolic tangent) and the `FEMMetadata` as its `problem_type`, so `solve` returns a
`PDETimeSeriesSolution` (`SciMLBase.wrap_sol` of the solver's solution). Nothing calls
`mtkcompile`; anything it cannot discretize raises. A `disc` without a time variable lowers instead to the steady
`NonlinearProblem` documented under `NonlinearProblem(pdesys, disc)`. Keywords are passed to
the `DAEProblem` after `disc.kwargs`, so a keyword given here wins; the keywords the discretization sets (`iv`, `unknowns`, `du0`, `coordinate`,
`name`, `parameters`) are rejected, and so are `tspan` (the `PDESystem`'s time domain), `u0` (its
initial conditions; use `remake(prob; u0)`), `p` (its parameter defaults) and `differential_vars`
(computed from the weak form). The problem carries an `abstol` vector when `disc.algebraic_abstol`
applies (see [`FEMDiscretization`](@ref)). Any other SciML problem or solver keyword (`saveat`,
`tstops`, `callback`, ...) is forwarded. Further positional arguments, a first argument that is not
a `PDESystem`, and an equation, domain or boundary condition the discretization does not support
raise an `ArgumentError` before the operators are assembled.
[`symbolic_discretize`](@ref SciMLBase.symbolic_discretize(::ModelingToolkit.PDESystem, ::FEMDiscretization))
returns the symbolic system of the same discretization.

The Dirichlet rows are linear with identity Jacobian rows, so a Newton step with a current
Jacobian satisfies them exactly. OrdinaryDiffEq's BDF methods reuse the Jacobian across
step-size changes and then rescale the Newton step, which leaves these rows off by about the
solver tolerance; `DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0))` refreshes it on every step
size change and holds `u(t, Γ) ~ g(t)` to round-off.
"""
function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    _check_reserved(kwargs)
    parts = _lower(pdesys, disc)
    disc.time === nothing && return _steady_problem(parts; disc.kwargs..., kwargs...)
    kwargs = merge(disc.kwargs, values(kwargs))
    return numeric_dae_problem(
        parts.ops, parts.u0, parts.tspan, parts.md;
        iv = parts.iv, unknowns = parts.unknowns, name = parts.name, parameters = parts.parameters, kwargs...,
        _abstol(disc.algebraic_abstol, parts.ops, parts.selfnonlinear, get(kwargs, :abstol, nothing))...
    )
end

# An algebraic field whose own unknowns are absent from its equation (the pressure in the continuity equation) is
# fixed only through the other fields, which makes the DAE index 2. The nonlinear Jacobian shares K's pattern and
# says nothing about which blocks are zero, so a field in `selfnonlinear`, whose equation has a nonlinear term in
# the field itself (0 = v² − u), counts as present.
function _index2(ops::FEMOperators, selfnonlinear)
    free = trues(size(ops.K, 1))
    free[ops.prescribed] .= false
    return any(ops.fields) do (name, r)
        rows = filter(i -> free[i], r)
        !isempty(rows) && !(name in selfnonlinear) && !any(ops.differential_vars[rows]) && iszero(ops.K[rows, r])
    end
end

# Whether a nonlinear term of equation `f` (paired with field `f`) reads field `f`'s value, gradient or Hessian.
function _nonlinear_in_own_field(wf, an, f)
    d = an.dim
    own = [an.slots.U[f]; an.slots.G[((f - 1) * d + 1):(f * d)]; an.slots.H[((f - 1) * d * d + 1):(f * d * d)]]
    return any(term -> term.kind === :nonlinear && term.eq == f && _depends(unwrap(term.integrand), own), [wf.volume; wf.facet])
end

# The `abstol` keyword of the `DAEProblem` (see `FEMDiscretization`'s `algebraic_abstol`); empty when none is set.
function _abstol(algebraic, ops, selfnonlinear, abstol)
    algebraic === :auto && (algebraic = _index2(ops, selfnonlinear) ? 1.0e8 : nothing)
    (algebraic === nothing || !(abstol isa Union{Nothing, Real})) && return (;)
    differential = something(abstol, 1.0e-6)
    return (; abstol = [d ? differential : algebraic for d in ops.differential_vars])
end

function SciMLBase.DAEProblem(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    Base.depwarn("`DAEProblem(pdesys, disc)` is deprecated, use `discretize(pdesys, disc)` instead", :DAEProblem)
    return _daeproblem(pdesys, disc; kwargs...)
end

_daeproblem(pdesys, disc; kwargs...) =
    disc.time === nothing ?
    throw(
        ArgumentError(
            "the FEMDiscretization has no time variable, so it lowers to a NonlinearProblem, not a DAEProblem; " *
            "call discretize(pdesys, disc)"
        )
    ) : SciMLBase.discretize(pdesys, disc; kwargs...)

"""
    NonlinearProblem(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Deprecated: use `discretize(pdesys, disc; kwargs...)`, which returns the same problem.

Discretize the steady `pdesys` with a `disc` that has no time variable and return the
`NonlinearProblem` `K u + N(u) - F = 0` of the Galerkin operators, with Dirichlet rows
`u_d - g_d` as algebraic equations, the sparse analytic Jacobian `K + ∂N/∂u` and the
`FEMMetadata` as its `problem_type`, so `solve` returns a `PDENoTimeSolution`. `N` holds the
terms whose coefficients depend on the state, as on the time-dependent path; without them
Newton solves the problem in one step.

The initial guess `u0` is zero, with the Dirichlet data `g_d` wherever the problem's unknown
vector holds Dirichlet dofs; a steady `PDESystem` has no initial conditions. To start Newton
elsewhere, `remake(prob; u0)` with a vector laid out like `prob.u0`, the problem's unknown
vector (with the default `dirichlet = :algebraic` it holds every dof, and field `name`
occupies `FiniteElementMethod.field_range(prob.problem_type.space, name)`).
Keywords are passed to the `NonlinearProblem` after `disc.kwargs`; the keywords
`iv`, `unknowns`, `du0`, `coordinate`, `name`, `parameters`, `tspan`, `u0`, `p` and
`differential_vars` are rejected, as for the `DAEProblem`.
"""
function SciMLBase.NonlinearProblem(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    Base.depwarn("`NonlinearProblem(pdesys, disc)` is deprecated, use `discretize(pdesys, disc)` instead", :NonlinearProblem)
    return _nonlinearproblem(pdesys, disc; kwargs...)
end

function _nonlinearproblem(pdesys, disc; kwargs...)
    disc.time === nothing || throw(
        ArgumentError(
            "the FEMDiscretization has the time variable $(disc.time), so it lowers to a DAEProblem; " *
                "for a steady problem write FEMDiscretization(Ω) without a time variable"
        )
    )
    return SciMLBase.discretize(pdesys, disc; kwargs...)
end

SciMLBase.discretize(::ModelingToolkit.PDESystem, ::FEMDiscretization, arg, args...; kwargs...) =
    throw(ArgumentError("discretize(pdesys, disc; kwargs...) takes no further positional arguments; got $(length(args) + 1)"))

SciMLBase.discretize(x, ::FEMDiscretization, args...; kwargs...) =
    throw(ArgumentError("FEMDiscretization discretizes a ModelingToolkit.PDESystem; got $(typeof(x))"))

"""
    symbolic_discretize(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Return `(sys, tspan)`, the shape PDEBase and MethodOfLines return, for the discretization that
`discretize(pdesys, disc; kwargs...)` builds: it runs the same code path, so it assembles the
Galerkin operators, and returns the `DAEProblem`'s `prob.f.sys` and `prob.tspan`. `sys` is the
completed `ModelingToolkit.System` of [`FiniteElementMethod.fem_system`](@ref), whose unknowns are
the arrays of field values, with the operators as parameters and the `FEMMetadata` (the
problem's `problem_type`) attached for rebuilding the solution; `tspan` is the time interval of
`pdesys`. Keywords are passed to that problem construction as for `discretize`,
so the reserved keywords raise the same `ArgumentError`. Further positional arguments raise an
`ArgumentError`.

A steady `disc` (no time variable) raises an `ArgumentError`: its `NonlinearProblem` is built
from a numeric `NonlinearFunction` with no `System`, so there is no symbolic system describing
it; call `discretize(pdesys, disc)` for the `NonlinearProblem`.
"""
function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization, args...; kwargs...)
    isempty(args) || throw(
        ArgumentError(
            "symbolic_discretize(pdesys, disc; kwargs...) takes no further positional arguments; got $(length(args))"
        )
    )
    disc.time === nothing && throw(
        ArgumentError(
            "a steady FEMDiscretization builds no symbolic System, so it has no symbolic_discretize; " *
                "call discretize(pdesys, disc) for its NonlinearProblem"
        )
    )
    prob = SciMLBase.discretize(pdesys, disc; kwargs...)
    return prob.f.sys, prob.tspan
end

"""
    operators(pdesys::PDESystem, disc::FEMDiscretization) -> NamedTuple

The assembled Galerkin operators of `discretize(pdesys, disc)` as plain data, for a caller that
builds its own system around them (such as a component with boundary ports). Returns

  - `M`, `K`: the mass and stiffness matrices with one sparsity pattern, before any Dirichlet row
    is replaced, so a caller can impose its own conditions or read reactions from them;
  - `F`: the load vector, the volume sources and natural boundary data, without Dirichlet values;
  - `fields`: `name => dof range` per field, in declared order;
  - `differential_vars`: rows of `M` with a nonzero entry;
  - `prescribed`, `dirichlet_values`: the dofs of the `PDESystem`'s Dirichlet conditions and
    their values (the rows `M` and `K` would have replaced);
  - `u0`: the initial dof vector, with the Dirichlet values on the prescribed dofs. It always has
    every dof, also under `dirichlet = :eliminate`, where the `DAEProblem`'s own `u0` holds the
    free dofs only;
  - `metadata`: the [`FEMMetadata`](@ref), for [`facet_dofs`](@ref),
    [`dof_coordinates`](@ref) and [`cell_dofs`](@ref).

It is the linear, time-independent case only, and anything else raises an `ArgumentError`: a
nonlinear term ("operators returns linear operators only, but the PDESystem has nonlinear
terms"), a time-dependent source ("operators needs time-independent sources, but a source
depends on time") and time-dependent Dirichlet data ("operators needs Dirichlet data that are
constant in time"). Parameters are substituted by their values, as for the `DAEProblem`.
"""
function operators(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    parts = _lower(pdesys, disc)
    (; M, K, load, nonlinear, prescribed, dirichlet_values, timedep_dirichlet) = parts.raw
    nonlinear && throw(ArgumentError("operators returns linear operators only, but the PDESystem has nonlinear terms"))
    load isa AbstractVector || throw(ArgumentError("operators needs time-independent sources, but a source depends on time"))
    timedep_dirichlet && throw(ArgumentError("operators needs Dirichlet data that are constant in time"))
    differential_vars = falses(size(M, 1))
    for (k, i) in enumerate(rowvals(M))
        iszero(nonzeros(M)[k]) || (differential_vars[i] = true)
    end
    md = parts.md
    fields = [f.name => field_range(md.space, f.name) for f in md.fields]
    return (;
        M = copy(M), K = copy(K), F = copy(load), fields, differential_vars, prescribed = copy(prescribed),
        dirichlet_values = copy(dirichlet_values), u0 = parts.full_u0, metadata = md,
    )
end
