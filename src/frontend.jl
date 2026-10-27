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
    unknown = [k for k in keys(disc.spaces) if !any(op -> isequal(op, k), ops)]
    isempty(unknown) || throw(
        ArgumentError(
            "`spaces` has keys that are not dependent variables of the system: " *
                "$(join(sort(string.(unknown)), ", ")); the dependent variables are $(join(sort(string.(dvs)), ", "))"
        )
    )
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
    wf = weak_form(an)
    steady = disc.time === nothing
    _check_terms(wf)
    pvals = _parameter_values(sys, ctx)
    dirichlet = emit_dirichlet(an)
    t0, t1 = steady ? (0.0, 0.0) : v.intervals[v.time]
    space, names, ips = _space(disc, ctx, an, dirichlet, pvals, t0)
    timedep = findfirst(bc -> _depends(bc.g, [an.slots.T]), an.essential)
    disc.dirichlet === :eliminate && timedep !== nothing && throw(
        ArgumentError(
            "dirichlet = :eliminate needs Dirichlet data that are constant in time, but the condition on boundary piece " *
                "$(an.essential[timedep].piece.name) depends on time; use the default dirichlet = :algebraic"
        )
    )
    ops, coords = _operators(space, wf, names, dirichlet, pvals, t0, timedep !== nothing)
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
    return (; ops, u0 = unknown_u0, tspan = (t0, t1), md, iv, unknowns, name = nameof(pdesys), parameters)
end

# `K u + N(u, t0) - F` with the Dirichlet rows `u_d - g_d`: `K` holds `e_d`, `F` holds `g_d` and `N` is zero there.
function _steady_problem(parts; kwargs...)
    (; ops) = parts
    t0 = first(parts.tspan)
    F = load!(zeros(length(parts.u0)), ops.load, t0)
    N = ops.nonlinear
    function f!(res, u, p)
        mul!(res, ops.K, u)
        res .-= F
        N === nothing || add_nonlinear!(res, N, u, t0)
        return nothing
    end
    function jac!(J, u, p)
        check_pattern(J, ops.K, "the Jacobian")
        copyto!(nonzeros(J), nonzeros(ops.K))
        N === nothing || add_nonlinear_jacobian!(J, N, u, t0)
        return nothing
    end
    f = SciMLBase.NonlinearFunction{true}(f!; jac = jac!, jac_prototype = copy(ops.K))
    return SciMLBase.NonlinearProblem{true}(f, parts.u0, SciMLBase.NullParameters(), parts.md; kwargs...)
end

"""
    DAEProblem(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

Discretize `pdesys` with `disc` and return the `DAEProblem` `M du + K u + N(u, t) = F(t)` of
the Galerkin operators, with Dirichlet rows `u_d = g_d(t)` as algebraic equations (or, with
`dirichlet = :eliminate`, restricted to the free dofs), a sparse
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
    disc.time === nothing && throw(
        ArgumentError(
            "the FEMDiscretization has no time variable, so it lowers to a NonlinearProblem, not a DAEProblem; " *
                "call discretize(pdesys, disc) or NonlinearProblem(pdesys, disc)"
        )
    )
    parts = _lower(pdesys, disc)
    return numeric_dae_problem(
        parts.ops, parts.u0, parts.tspan, parts.md;
        iv = parts.iv, unknowns = parts.unknowns, name = parts.name, parameters = parts.parameters,
        disc.kwargs..., kwargs...
    )
end

"""
    NonlinearProblem(pdesys::PDESystem, disc::FEMDiscretization; kwargs...)

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
`iv`, `unknowns`, `du0`, `coordinate`, `name` and `parameters` are rejected, as for the `DAEProblem`.
"""
function SciMLBase.NonlinearProblem(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...)
    _check_reserved(kwargs)
    disc.time === nothing || throw(
        ArgumentError(
            "the FEMDiscretization has the time variable $(disc.time), so it lowers to a DAEProblem; " *
                "for a steady problem write FEMDiscretization(Ω) without a time variable"
        )
    )
    return _steady_problem(_lower(pdesys, disc); disc.kwargs..., kwargs...)
end

SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization; kwargs...) =
    disc.time === nothing ? NonlinearProblem(pdesys, disc; kwargs...) : DAEProblem(pdesys, disc; kwargs...)

# `(sys, tspan)`, the shape PDEBase and MethodOfLines return.
function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    disc.time === nothing && throw(
        ArgumentError(
            "a steady FEMDiscretization builds no symbolic System, so it has no symbolic_discretize; " *
                "call discretize(pdesys, disc) for its NonlinearProblem"
        )
    )
    prob = DAEProblem(pdesys, disc)
    return prob.f.sys, prob.tspan
end
