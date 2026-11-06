# PDESystem front end of the Maxwell model: `discretize` with `FEMDiscretization(…; omega)` reads the
# time-domain curl-curl equation off PDEBase's strong-form analysis, samples its coefficients per cell and
# hands them to `MaxwellModel` and `driven_problem`. It bypasses the scalar weak form, assembly and DAE path.

"""
    curl(F, (x, y, z))

The curl `∇ × F` of the 3-vector of symbolic expressions `F` in the coordinates `(x, y, z)`, as a vector of
`Differential` expressions, for writing the Maxwell operator in a `PDESystem`:
`curl(curl(E(t, x, y, z), (x, y, z)) / μ, (x, y, z))`. This is a method of Ferrite's (Tensors') `curl`, which
`FiniteElementMethod` re-exports, so that loading both packages leaves one `curl`.
"""
function Ferrite.curl(F::AbstractVector, x::NTuple{3, Any})
    length(F) == 3 || throw(ArgumentError("curl needs a 3-vector, got $(length(F)) components"))
    D = Differential.(collect(x))
    return [D[2](F[3]) - D[3](F[2]), D[3](F[1]) - D[1](F[3]), D[1](F[2]) - D[2](F[1])]
end

"""
    facet_normal(Γ::BoundaryPiece)

The symbolic outward unit normal of the boundary piece `Γ`, `PDEBase.normal(Γ)`, for boundary conditions
such as the perfect conductor `cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0`. It needs a PDEBase
with facet normals (SciML/PDEBase.jl#164); the name avoids DomainSets' own exported `normal`.
"""
function facet_normal(Γ::PDEBase.BoundaryPiece)
    hasmethod(PDEBase.normal, Tuple{typeof(Γ)}) || throw(
        ArgumentError("facet_normal needs a PDEBase with PDEBase.normal(Γ) for boundary pieces (SciML/PDEBase.jl#164)")
    )
    return PDEBase.normal(Γ)
end

# PDEBase's facet normal operator, absent before SciML/PDEBase.jl#164.
const _FacetNormal = isdefined(PDEBase, :FacetNormal) ? getfield(PDEBase, :FacetNormal) : Nothing

# A fixed, well-spread sequence in (0, 1) for the sample points of the coefficient checks.
_spread(k) = mod(0.5 + k * 0.6180339887498949, 1.0)

const _MAXWELL_FORM = "write the operator as curl(ν*curl(E)) with curl(F, (x, y, z)), as in curl(curl(E, X) / μ, X)"

# Sample points of the coefficient checks: cell centroids spread through the mesh, with generic values of the
# state, gradient, time and frequency slots.
function _sample_values(grid, an, pvals, t0, scale; n = 4)
    (; slots) = an
    cells = unique(round.(Int, range(1, Ferrite.getncells(grid); length = n)))
    k = 0
    next() = (k += 1; _spread(k))
    return map(cells) do cell
        x = sum(Ferrite.getcoordinates(grid, cell)) / 4
        vals = Dict{Any, Any}(unwrap(slots.X[d]) => x[d] for d in 1:3)
        vals[unwrap(slots.T)] = t0 + next()
        for (p, value) in zip(slots.P, pvals)
            vals[unwrap(p)] = value
        end
        for s in [slots.U; slots.G]
            vals[unwrap(s)] = 2 * next() - 1
        end
        for s in scale
            vals[unwrap(s)] = 1 + next()
        end
        vals
    end
end

function _number(ex, vals)
    value = SymbolicUtils.unwrap_const(unwrap(substitute(ex, vals; fold = Val(true))))
    value isa Number || throw(ArgumentError("a coefficient of the Maxwell equation does not evaluate to a number: $ex"))
    value isa Real || throw(ArgumentError("coefficients and sources of the Maxwell equation must be real, got $value in $ex"))
    return Float64(value)
end

# `Dt(E_c)` becomes `s₁ E_c` and `Dt(Dt(E_c))` `s₂ E_c` (the phasor of exp(+iωt) has s₁ = iω, s₂ = -ω²).
_phasor(ex, t, isfield, s) = substitute(ex, _phasor_rules!(Dict{Any, Any}(), unwrap(ex), t, isfield, s))

function _phasor_rules!(rules, ex, t, isfield, s)
    SymbolicUtils.iscall(ex) || return rules
    op = SymbolicUtils.operation(ex)
    if op isa Differential && isequal(op.x, t)
        order, inner = 0, ex
        while SymbolicUtils.iscall(inner) && SymbolicUtils.operation(inner) isa Differential &&
                isequal(SymbolicUtils.operation(inner).x, t)
            order += SymbolicUtils.operation(inner).order
            inner = unwrap(only(SymbolicUtils.arguments(inner)))
        end
        isfield(inner) && order <= 2 || throw(
            ArgumentError("the time derivative $(Num(ex)) must be a first or second time derivative of a component of the unknown")
        )
        rules[ex] = unwrap(s[order] * inner)
        return rules
    end
    foreach(a -> _phasor_rules!(rules, unwrap(a), t, isfield, s), SymbolicUtils.arguments(ex))
    return rules
end

# The unknown with the Nédélec space and the component of each flattened field.
function _maxwell_components(fields, replaced, disc)
    nedelec = [k for (k, ip) in disc.spaces if ip isa Ferrite.Nedelec]
    length(nedelec) == 1 && length(disc.spaces) == 1 || throw(
        ArgumentError("omega needs one vector unknown with a Nédélec space and no other spaces, as spaces = Dict(E => Nedelec{RefTetrahedron, 1}())")
    )
    E = only(nedelec)
    return map(fields) do u
        r = get(replaced, unwrap(u), nothing)
        indexed = r === nothing ? nothing : unwrap(r)
        ok = indexed !== nothing && SymbolicUtils.iscall(indexed) && SymbolicUtils.operation(indexed) === getindex &&
            isequal(_field_op(SymbolicUtils.arguments(indexed)[1]), E)
        ok || throw(
            ArgumentError(
                "omega discretizes one 3-component vector unknown with a Nédélec space, as @variables E(..)[1:3]; " *
                    "$(Num(u)) is not a component of it"
            )
        )
        Int(SymbolicUtils.unwrap_const(SymbolicUtils.arguments(indexed)[2]))
    end
end

# Field `c` of the analysis is component `c` of E. Equation `e` in slot form tests component `comps[e]`; returns ν, the reaction λ and the source in slots,
# after checking that the second-order part is curl(ν curl E) and the rest is λ E - f.
function _recognise_volume(an, samples, s)
    (; slots) = an
    Gs(c, b) = slots.G[(c - 1) * 3 + b]
    derivs = [[Symbolics.derivative(an.flux[e][a], Gs(c, b)) for a in 1:3, c in 1:3, b in 1:3] for e in 1:3]
    state = [slots.U; slots.G]
    nonlinear(e) = ArgumentError("the Maxwell front end discretizes linear equations only, but equation $e is nonlinear in E")
    comps = zeros(Int, 3)
    nus, lambdas, sources = Num[], Num[], Num[]
    for e in 1:3
        any(ex -> _depends(unwrap(ex), state), derivs[e]) && throw(nonlinear(e))
        remainder = an.remainder[e]
        dU = [Symbolics.derivative(remainder, slots.U[c]) for c in 1:3]
        dG = [Symbolics.derivative(remainder, g) for g in slots.G]
        any(ex -> _depends(unwrap(ex), state), [dU; dG]) && throw(nonlinear(e))
        flux0 = [substitute(an.flux[e][a], Dict(unwrap(g) => 0 for g in slots.G)) for a in 1:3]
        trace = [sum(derivs[e][a, c, a] for a in 1:3) for c in 1:3]
        for vals in samples
            B = [_number(ex, vals) for ex in derivs[e]]
            tr = [_number(ex, vals) for ex in trace]
            λU = [_number(ex, vals) for ex in dU]
            scale = max(maximum(abs, B), maximum(abs, λU), 1.0e-300)
            tol = 1.0e-10 * scale
            if comps[e] == 0
                comps[e] = maximum(abs, tr) > tol ? argmax(abs.(tr)) : argmax(abs.(λU))
                maximum(abs, tr) > tol || maximum(abs, λU) > tol ||
                    throw(ArgumentError("equation $e has neither a curl-curl nor a reaction term in E"))
            end
            ce = comps[e]
            other = mod1(ce + 1, 3)
            α, β, γ = B[other, ce, other], B[other, other, ce], B[ce, other, other]
            expect = [α * (c == ce) * (a == b) + β * (b == ce) * (a == c) + γ * (a == ce) * (c == b) for a in 1:3, c in 1:3, b in 1:3]
            abs(γ) > tol && throw(
                ArgumentError("equation $e has a grad div E term, which has no meaning on Nédélec elements; $_MAXWELL_FORM")
            )
            maximum(abs, B - expect) > tol && throw(
                ArgumentError("the second-order part of equation $e is anisotropic or couples the components of E differently from curl(ν*curl(E)); $_MAXWELL_FORM")
            )
            abs(β) <= tol && abs(α) > tol && throw(
                ArgumentError("equation $e has the vector Laplacian of E, which is component-wise H¹ and not H(curl)-conforming; $_MAXWELL_FORM")
            )
            abs(α + β) > tol && throw(
                ArgumentError("the second-order part of equation $e adds a component-wise Laplacian of E to curl(ν*curl(E)), which is not H(curl)-conforming; $_MAXWELL_FORM")
            )
            any(ex -> abs(_number(ex, vals)) > tol, flux0) && throw(
                ArgumentError("equation $e has a first-order term inside a derivative; $_MAXWELL_FORM")
            )
            any(ex -> abs(_number(ex, vals)) > tol, dG) && throw(
                ArgumentError("equation $e has a first-order term in the derivatives of E, which curl(ν*curl(E)) does not have; $_MAXWELL_FORM")
            )
            maximum(abs, λU[[c for c in 1:3 if c != ce]]; init = 0.0) > tol && throw(
                ArgumentError("the zeroth-order term of equation $e couples the components of E; the coefficients of E and its time derivatives must be scalars")
            )
        end
        ce = comps[e]
        push!(nus, derivs[e][mod1(ce + 1, 3), mod1(ce + 1, 3), ce])
        push!(lambdas, dU[ce])
        push!(sources, -substitute(remainder, Dict(unwrap(g) => 0 for g in state)))
    end
    sort(comps) == 1:3 || throw(ArgumentError("the equations test the components $comps of E; write one equation per component"))
    for vals in samples, (name, exs) in (("ν", nus), ("the coefficient of E", lambdas))
        v = [_number(ex, vals) for ex in exs]
        maximum(abs, v .- v[1]) <= 1.0e-10 * maximum(abs, v) || throw(
            ArgumentError("$name differs between the component equations, so the medium is anisotropic; the coefficients must be scalars")
        )
    end
    order = invperm(comps)
    λ = lambdas[1]
    zero_s = Dict(unwrap(x) => 0 for x in s)
    a = (
        a0 = substitute(λ, zero_s),
        a1 = Symbolics.derivative(λ, s[1]), a2 = Symbolics.derivative(λ, s[2]),
    )
    for vals in samples
        v = Dict(vals)
        lin = _number(a.a0, v) + v[unwrap(s[1])] * _number(a.a1, v) + v[unwrap(s[2])] * _number(a.a2, v)
        abs(_number(λ, v) - lin) <= 1.0e-10 * max(abs(lin), 1.0) ||
            throw(ArgumentError("the coefficient of E is nonlinear in its time derivatives"))
    end
    nu = nus[1]
    sources = sources[order]
    for (what, ex) in ["ν" => nu; ["a coefficient" => c for c in a]; ["a source" => f for f in sources]]
        _depends(unwrap(ex), [slots.T]) && throw(
            ArgumentError("$what of the Maxwell equation depends on time; with omega the coefficients and sources are time-independent, and a source is the phasor amplitude")
        )
    end
    return (; nu, a..., sources)
end

# n × E = 0 conditions: each residual is linear in the traces and the normal, tangential (C n = 0) and of rank 2
# together on its piece.
function _recognise_pec(bcs, fields, pvals, ctx, domain)
    U = [Symbolics.variable(:U, c) for c in 1:3]
    N = [Symbolics.variable(:N, d) for d in 1:3]
    byp = Dict{Symbol, Vector{Num}}()
    pform = "write the perfect conductor as cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0"
    for eq in bcs
        res = eq.lhs - eq.rhs
        pieces = _trace_pieces(res)
        _walk(res) do ex
            SymbolicUtils.iscall(ex) && SymbolicUtils.operation(ex) isa _FacetNormal && push!(pieces, SymbolicUtils.operation(ex).piece)
        end
        unique!(pieces)
        length(pieces) == 1 || throw(
            ArgumentError("with omega every condition must be a boundary condition on one piece, and initial conditions are not used; got $eq")
        )
        Γ = only(pieces)
        Γ.parent === domain || throw(ArgumentError("the boundary piece $(Γ.name) of $eq is not a piece of the FEMDiscretization's MeshDomain"))
        rules = Dict{Any, Any}()
        _walk(res) do ex
            SymbolicUtils.iscall(ex) || return
            op = SymbolicUtils.operation(ex)
            if op isa Union{Differential, PDEBase.NormalDerivative}
                throw(ArgumentError("$eq has a derivative; with omega only the perfect conductor n × E = 0 is supported: $pform"))
            elseif op isa _FacetNormal
                rules[ex] = unwrap(N[Int(SymbolicUtils.unwrap_const(only(SymbolicUtils.arguments(ex))))])
            else
                f = findfirst(isequal(op), ctx.ops)
                f === nothing || (rules[ex] = unwrap(U[fields[f]]))
            end
        end
        for (p, value) in zip(ctx.params, pvals)
            rules[p] = value
        end
        B = substitute(res, rules)
        extra = [x for x in Symbolics.get_variables(B) if !any(isequal(x), unwrap.([U; N]))]
        isempty(extra) || throw(ArgumentError("$eq depends on $(join(Num.(extra), ", ")); $pform"))
        push!(get!(byp, Γ.name, Num[]), B)
    end
    for (name, Bs) in byp
        C = [Symbolics.derivative(B, u) for B in Bs, u in U]
        any(ex -> _depends(unwrap(ex), U), C) && throw(ArgumentError("the conditions on $name are nonlinear in E; $pform"))
        for k in 1:3
            n = [2 * _spread(3k + d) - 1 for d in 1:3]
            n ./= norm(n)
            vals = Dict{Any, Any}(unwrap(N[d]) => n[d] for d in 1:3)
            merge!(vals, Dict(unwrap(u) => 0.0 for u in U))
            any(B -> abs(_number(B, vals)) > 1.0e-12, Bs) &&
                throw(ArgumentError("the conditions on $name prescribe a nonzero tangential field, which is not supported; $pform"))
            Cn = [_number(ex, vals) for ex in C]
            maximum(abs, Cn * n) > 1.0e-10 * maximum(abs, Cn) && throw(
                ArgumentError("the conditions on $name are not tangential for every normal direction, as n × E = 0 is; $pform")
            )
            rank(Cn) == 2 || throw(
                ArgumentError("the conditions on $name constrain $(rank(Cn)) of the two tangential components of E; $pform")
            )
        end
    end
    pec = sort!(String.(collect(keys(byp))))
    check_facetsets(domain.mesh, Symbol.(pec); disjoint = false)
    return pec
end

"""
    maxwell_model(pdesys::PDESystem, disc::FEMDiscretization) -> (; model, omega, load)

The [`MaxwellModel`](@ref), angular frequency and load vector that `discretize(pdesys, disc)` builds for a
`disc` with `omega`, so that the solution `u` of its `LinearProblem` can be wrapped as
`MaxwellSolution(model, omega, u)` and evaluated with [`field_values`](@ref).

The `PDESystem` is the time-domain equation for a vector unknown `E` with `spaces = Dict(E => Nedelec{RefTetrahedron, 1}())`,

    curl(ν curl E) + a₂ ∂ₜ²E + a₁ ∂ₜE + a₀ E = f,

one equation per component, written with [`curl`](@ref). ν, a₀, a₁ and a₂ are real scalar functions of
the coordinates and parameters, and the source `f(x)` is read as the phasor amplitude of `exp(+iωt)`. Each
cell gets `MaxwellMaterial(eps_r = a₂ - a₀/ω², mu_r = 1/ν, sigma = a₁)` sampled at its centroid, with
`eps0 = mu0 = length_scale = 1`, so the system is `Kc - ω² Mε + iω Mσ` of the model and the load is
`∫ f ⋅ Nᵢ`. Boundary conditions are the perfect conductor `cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0`;
pieces without a condition get the natural `n × (ν curl E) = 0`. A complex permittivity has no real
time-domain form here; build a `MaxwellModel` directly for it.
"""
function maxwell_model(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization)
    disc.omega === nothing &&
        throw(ArgumentError("maxwell_model needs a FEMDiscretization with omega, as FEMDiscretization(Ω, t; omega = 2π, spaces)"))
    _check_domains(pdesys, disc)
    grid = disc.domain.mesh
    celltype = Ferrite.getcelltype(grid)
    celltype === Ferrite.Tetrahedron ||
        throw(ArgumentError("omega needs a mesh of Tetrahedron cells, got $(celltype isa DataType ? nameof(celltype) : celltype)"))
    any(dv -> dv isa Complex || eltype(SymbolicUtils.symtype(unwrap(dv))) <: Complex, pdesys.dvs) &&
        throw(ArgumentError("the unknown must be real: with omega the PDESystem is the real time-domain equation and the phasor is complex"))
    sys, replaced = PDEBase.make_pdesys_compatible(pdesys)
    v = PDEBase.VariableMap(sys, disc; replaced_vars = replaced)
    fields = StrongForm.declared_fields(sys, v)
    comps = _maxwell_components(fields, replaced, disc)
    length(fields) == 3 && comps == 1:3 ||
        throw(ArgumentError("omega needs all three components of the vector unknown and no other unknowns; got $(join(Num.(fields), ", "))"))
    s = collect(@parameters maxwell_dt maxwell_dt2)
    ops = [SymbolicUtils.operation(unwrap(u)) for u in fields]
    isfield(ex) = SymbolicUtils.iscall(ex) && any(isequal(SymbolicUtils.operation(ex)), ops)
    eqs = [Num(_phasor(eq.lhs, v.time, isfield, s)) ~ Num(_phasor(eq.rhs, v.time, isfield, s)) for eq in sys.eqs]
    length(eqs) == 3 || throw(ArgumentError("the system has $(length(eqs)) equations for the 3 components of the vector unknown"))
    ps = sys.ps isa SciMLBase.NullParameters ? [] : collect(sys.ps)
    phasor_sys = ModelingToolkit.PDESystem(
        eqs, Equation[], sys.domain, sys.ivs, sys.dvs, [ps; [p => 0.0 for p in s]]; name = nameof(pdesys)
    )
    pv = PDEBase.VariableMap(phasor_sys, disc; replaced_vars = replaced)
    an = StrongForm.analyse(phasor_sys, pv, Dict())
    ctx = StrongForm.SlotContext(phasor_sys, pv)
    pvals = _parameter_values(phasor_sys, ctx)
    js = [findfirst(isequal(unwrap(p)), ctx.params) for p in s]
    slot_s = an.slots.P[js]
    keep = setdiff(eachindex(ctx.params), js)
    t0 = first(pv.intervals[pv.time])
    samples = _sample_values(grid, an, pvals, t0, slot_s)
    coeffs = _recognise_volume(an, samples, slot_s)
    pec = _recognise_pec(sys.bcs, comps, pvals[keep], (; ops, params = ctx.params[keep]), disc.domain)

    # Compile the coefficients and sources as functions of the mesh coordinate.
    fixed = Dict{Any, Any}(unwrap(an.slots.P[j]) => pvals[j] for j in keep)
    X = an.slots.X
    compile(exs) = Symbolics.build_function([substitute(ex, fixed) for ex in exs], X...; expression = Val(false))[1]
    coefficient_at = compile([coeffs.nu, coeffs.a0, coeffs.a1, coeffs.a2])
    source_at = compile(coeffs.sources)
    ω = disc.omega
    materials = map(1:Ferrite.getncells(grid)) do cell
        x = sum(Ferrite.getcoordinates(grid, cell)) / length(Ferrite.getcoordinates(grid, cell))
        nu, a0, a1, a2 = _real_values(coefficient_at(x...), "a coefficient")
        iszero(nu) && throw(ArgumentError("ν = $nu in cell $cell; ν = 1/μ must be nonzero"))
        MaxwellMaterial(; eps_r = a2 - a0 / ω^2, mu_r = 1 / nu, sigma = a1)
    end
    model = MaxwellModel(
        grid; materials, pec, eps0 = 1.0, mu0 = 1.0, length_scale = 1.0, quadrature = something(disc.quadrature, 2)
    )
    return (; model, omega = ω, load = _source_load(model, x -> _real_values(source_at(x...), "a source")))
end

function _real_values(values, what)
    all(v -> v isa Real && isfinite(v), values) ||
        throw(ArgumentError("$what of the Maxwell equation evaluates to $values; it must be real and finite"))
    return Float64.(values)
end

# ∫ f ⋅ Nᵢ for the source f(x) of the equation.
function _source_load(model, f)
    cv = model.cellvalues
    n = Ferrite.getnbasefunctions(cv)
    b = zeros(Ferrite.ndofs(model.dh))
    be = zeros(n)
    for cc in Ferrite.CellIterator(model.dh)
        Ferrite.reinit!(cv, cc)
        fill!(be, 0.0)
        x = Ferrite.getcoordinates(cc)
        for q in 1:Ferrite.getnquadpoints(cv)
            fq = Ferrite.Vec{3}(Tuple(f(Ferrite.spatial_coordinate(cv, q, x))))
            dΩ = Ferrite.getdetJdV(cv, q)
            for i in 1:n
                be[i] += dot(Ferrite.shape_value(cv, q, i), fq) * dΩ
            end
        end
        Ferrite.assemble!(b, Ferrite.celldofs(cc), be)
    end
    return b
end

# `discretize` with omega: the LinearProblem `A(ω) u = b` of the driven model.
function _maxwell_problem(pdesys, disc; kwargs...)
    (; model, omega, load) = maxwell_model(pdesys, disc)
    prob = driven_problem(model, omega; load)
    kwargs = merge(disc.kwargs, values(kwargs))
    return isempty(kwargs) ? prob : SciMLBase.LinearProblem(prob.A, prob.b; kwargs...)
end
