# The PDESystem front end (spec §4.6, §5, §7): PDEBase classifies the conditions and analyses the
# strong form with its boundary conditions; this file projects with Galerkin, assembles with m1's
# driver and lowers to the numeric DAE path.
#
# This driver calls PDEBase's steps (make_pdesys_compatible, VariableMap, interface_errors,
# parse_bcs, split_ics_bcs, StrongForm.analyse) itself rather than through the generic
# `symbolic_discretize`, whose remaining hooks build an MTK system this numeric path does not use.
# Tracking: move to PDEBase's `symbolic_discretize` with hook methods once the numeric path builds
# its system through `generate_system` (PDEBase M0 release as the compat floor).

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
        t !== nothing && any(a -> isequal(a, t), args) && !isequal(first(args), t) &&
            throw(ArgumentError("time must be the first argument of $u; time-last ordering lands later"))
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
        throw(ArgumentError("meshes with mixed cell types land in M4"))
    end
    check_facetsets(grid, [Γ.name for Γ in _condition_pieces(pdesys.bcs)])
    return nothing
end

function _boundary_map(bcs, v, disc)
    for eq in bcs
        length(_trace_pieces(eq.lhs - eq.rhs)) > 1 && throw(ArgumentError("$(_PERIODIC_M4): $eq"))
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
    fields = StrongForm.pair_equations(an.mass, an.slots; pairing = explicit)
    order = invperm(fields)
    natural = [typeof(bc)(bc.piece, fields[bc.eq], bc.Φ) for bc in an.natural]
    return typeof(an)(
        an.slots, an.nfields, an.dim, an.strong[order], an.mass[order], an.flux[order],
        an.remainder[order], an.conormal[order], an.essential, natural, an.periodic
    )
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
    op = LinearOp(nothing, nothing, F, zeros(ne, ne), zeros(ne, ne), zeros(ne))
    _assemble!(op, ctx, _split(ctx, kernels, (:source,))..., p, t)
    return F
end

# `reinit!` mutates CellValues and FacetValues, so each load evaluation gets its own copies and
# concurrent solves of one problem's remakes share no mutable state.
function _fresh_values(ctx::AssemblyContext{NF, DH, CV, FV}) where {NF, DH, CV, FV}
    return AssemblyContext{NF, DH, CV, FV}(
        ctx.dh, ctx.fields, map(copy, ctx.cellvalues), map(copy, ctx.facetvalues), ctx.facetsets, ctx.facetcells
    )
end

function _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    kernels = emit_kernels(wf)
    timedep = [term.kind === :source && term.timedep for term in vcat(wf.volume, wf.facet)]
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
    boundarymap = _boundary_map(sys.bcs, v, disc)
    ics, bcmap = PDEBase.split_ics_bcs(boundarymap, v, disc)
    ctx = StrongForm.SlotContext(sys, v)
    an = _paired(StrongForm.analyse(sys, v, boundarymap; ibp = disc.ibp), pdesys.eqs, ctx, disc)
    isempty(an.periodic) || throw(ArgumentError(_PERIODIC_M4))
    covered = Set(Γ for bykey in values(bcmap) for (Γ, bs) in bykey if !isempty(bs))
    if any(c -> !all(x -> isequal(x, 0), c), an.conormal)
        for Γ in PDEBase.boundarypieces(disc.domain)
            Γ in covered || @warn "boundary piece $Γ has no condition, so it gets the natural zero-flux condition"
        end
    end
    disc.test_space isa SUPG && _check_supg(disc.test_space, an, ctx)
    wf = weak_form(an; test_space = disc.test_space)
    _check_terms(wf)
    pvals = _parameter_values(sys, ctx)
    dirichlet = emit_dirichlet(an)
    t0, t1 = v.intervals[v.time]
    space, names, ips = _space(disc, ctx, an, dirichlet, pvals, t0)
    timedep_dirichlet = any(bc -> _depends(bc.g, [an.slots.T]), an.essential)
    ops, coords = _operators(space, wf, names, dirichlet, pvals, t0, timedep_dirichlet)
    u0 = _initial_values(Equation[b.eq for b in ics], ctx, space, coords, an, names, pvals)
    dirichlet_values!(view(u0, ops.prescribed), ops.load, t0)
    iv = disc.time
    unknowns = [only(@variables $name(iv)[1:length(field_range(space, name))]) for name in names]
    fields = [FieldInfo(dv, name, u, ip) for (dv, name, u, ip) in zip(StrongForm.declared_fields(sys, v), names, unknowns, ips)]
    pieces = Dict{Any, String}(Γ => String(Γ.name) for Γ in PDEBase.boundarypieces(disc.domain))
    md = FEMMetadata(space, iv, v.x̄, fields, pieces, pdesys; u0)
    parameters = [q => v for (q, v) in zip(ctx.params, pvals)]
    return (; ops, u0, tspan = (t0, t1), md, iv, unknowns, name = nameof(pdesys), parameters)
end

"""
    DAEProblem(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Discretize `pdesys` with `disc` and return the `DAEProblem` `M du + K u + N(u, t) = F(t)` of
the Galerkin operators, with Dirichlet rows `u_d = g_d(t)` as algebraic equations, a sparse
analytic Jacobian (the nonlinear part `N` from the terms whose coefficients depend on the
state, with its symbolic tangent) and the `FEMMetadata` as its `problem_type`, so `solve` returns a
`PDETimeSeriesSolution`. Nothing calls `mtkcompile`; anything this milestone cannot
discretize raises. Keywords are passed to the `DAEProblem` after `disc.kwargs`, so a keyword
given here wins; the keywords the discretization sets (`iv`, `unknowns`, `du0`, `coordinate`,
`name`, `parameters`) are rejected.

The Dirichlet rows are linear with identity Jacobian rows, so a Newton step with a current
Jacobian satisfies them exactly. OrdinaryDiffEq's BDF methods reuse the Jacobian across
step-size changes and then rescale the Newton step, which leaves these rows off by about the
solver tolerance; `DFBDF(nlsolve = NLNewton(new_W_dt_cutoff = 0))` refreshes it on every step
size change and holds `u(t, Γ) ~ g(t)` to round-off.
"""
function SciMLBase.DAEProblem(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    _check_reserved(kwargs)
    parts = _lower(pdesys, disc)
    return numeric_dae_problem(
        parts.ops, parts.u0, parts.tspan, parts.md;
        iv = parts.iv, unknowns = parts.unknowns, name = parts.name, parameters = parts.parameters,
        disc.kwargs..., kwargs...
    )
end

SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...) =
    DAEProblem(pdesys, disc; kwargs...)

# `(sys, tspan)`, the shape PDEBase and MethodOfLines return.
function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    prob = DAEProblem(pdesys, disc)
    return prob.f.sys, prob.tspan
end
