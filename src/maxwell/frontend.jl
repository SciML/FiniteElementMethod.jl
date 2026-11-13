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

# The symbolic outward unit normal of a boundary piece, `n(d)` for its component `d`, until PDEBase provides one
# (SciML/PDEBase.jl#164). It only builds a term: `_recognise_pec` reads it off the condition.
struct PieceNormal{D <: DomainSets.Domain} <: SymbolicUtils.Operator
    piece::D
end
Base.isequal(a::PieceNormal, b::PieceNormal) = isequal(a.piece, b.piece)
Base.:(==)(a::PieceNormal, b::PieceNormal) = isequal(a, b)
Base.hash(a::PieceNormal, h::UInt) = hash(a.piece, hash(PieceNormal, h))
SymbolicUtils.isbinop(::PieceNormal) = false
Base.show(io::IO, n::PieceNormal) = print(io, "n_", n.piece.name)

"""
    facet_normal(Γ::BoundaryPiece)

The symbolic outward unit normal of the boundary piece `Γ`, as a vector of its three components, for boundary
conditions such as the perfect conductor `cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0`. It is
`PDEBase.normal(Γ)` where PDEBase defines it (SciML/PDEBase.jl#164) and a term of this package otherwise; the
name avoids DomainSets' own exported `normal`. The normal itself is read off the facets of the mesh.
"""
function facet_normal(Γ::PDEBase.BoundaryPiece)
    hasmethod(PDEBase.normal, Tuple{typeof(Γ)}) && return PDEBase.normal(Γ)
    return [Num(SymbolicUtils.term(PieceNormal(Γ), d; type = Real)) for d in 1:3]
end

# PDEBase's facet normal operator, once it exists (SciML/PDEBase.jl#164).
const _FacetNormal = isdefined(PDEBase, :FacetNormal) ? Union{PieceNormal, getfield(PDEBase, :FacetNormal)} : PieceNormal

# The outward unit normal of every facet of the facet set `name`, with the facet.
function _facet_normals(grid, name)
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{RefTetrahedron}(1), Lagrange{RefTetrahedron, 1}())
    return map(collect(Ferrite.getfacetset(grid, name))) do facet
        Ferrite.reinit!(fv, Ferrite.getcells(grid, facet[1]), Ferrite.getcoordinates(grid, facet[1]), facet[2])
        (collect(Ferrite.getnormal(fv, 1)), facet)
    end
end

const _MAXWELL_FORM = "write the operator as curl(ν*curl(E)) with curl(F, (x, y, z)), as in curl(curl(E, X) / μ, X)"

# `Dt(E_c)` becomes `s₁ E_c` and `Dt(Dt(E_c))` `s₂ E_c` (the phasor of exp(+iωt) has s₁ = iω, s₂ = -ω²).
_phasor(ex, t, isfield, s) = substitute(ex, _phasor_rules!(Dict{Any, Any}(), unwrap(ex), t, isfield, s))

function _phasor_rules!(rules, ex, t, isfield, s, inside = false)
    SymbolicUtils.iscall(ex) || return rules
    op = SymbolicUtils.operation(ex)
    if op isa Differential && isequal(op.x, t)
        inside && throw(
            ArgumentError("the time derivative $(Num(ex)) is inside a spatial derivative; time derivatives may only multiply E, as in ε ∂ₜ²E + σ ∂ₜE")
        )
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
    foreach(a -> _phasor_rules!(rules, unwrap(a), t, isfield, s, inside || op isa Differential), SymbolicUtils.arguments(ex))
    return rules
end

# The unknown with the Nédélec space and the component of each flattened field.
function _maxwell_components(fields, replaced, disc)
    nedelec = [k for (k, ip) in disc.spaces if ip isa Ferrite.Nedelec]
    length(nedelec) == 1 && length(disc.spaces) == 1 || throw(
        ArgumentError("omega needs one vector unknown with a Nédélec space and no other spaces, as spaces = Dict(E => Nedelec{RefTetrahedron, 1}())")
    )
    E = only(nedelec)
    disc.spaces[E] == _NEDELEC ||
        throw(ArgumentError("omega supports the space Nedelec{RefTetrahedron, 1}() only, got $(disc.spaces[E])"))
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

_iszero_expr(ex) = isequal(Symbolics.simplify(Symbolics.expand(Num(ex))), Num(0))

# Field `c` of the analysis is component `c` of E. Equation `e` in slot form tests component `comps[e]`;
# returns ν, a₀, a₁, a₂ and the sources in slots, after checking at every cell centroid `xs`, and for the
# frequency slots `s` at (0, 0), (1, 0) and (0, 1), that the second-order part is curl(ν curl E) and the
# rest is λ E - f with the same scalar ν and λ in every equation. `fixed` substitutes the parameter values.
function _recognise_volume(an, s, fixed, xs)
    (; slots) = an
    Gs(c, b) = slots.G[(c - 1) * 3 + b]
    state = [slots.U; slots.G]
    nonlinear(e) = ArgumentError("the Maxwell front end discretizes linear equations only, but equation $e is nonlinear in E")
    timedep(what) = ArgumentError("$what of the Maxwell equation depends on time; with omega the coefficients and sources are time-independent, and a source is the phasor amplitude")
    B = [[Symbolics.derivative(an.flux[e][a], Gs(c, b)) for a in 1:3, c in 1:3, b in 1:3] for e in 1:3]
    dU = [[Symbolics.derivative(an.remainder[e], slots.U[c]) for c in 1:3] for e in 1:3]
    dG = [[Symbolics.derivative(an.remainder[e], g) for g in slots.G] for e in 1:3]
    dF = [[Symbolics.derivative(an.flux[e][a], slots.U[c]) for a in 1:3, c in 1:3] for e in 1:3]
    parts(e) = [vec(B[e]); dU[e]; dG[e]; vec(dF[e])]
    # Flux and remainder are then of degree one in U and G, so nothing else of them is left to discard.
    allslots = Dict(unwrap(g) => 0 for g in [slots.U; slots.G; slots.H; slots.Ud])
    for e in 1:3
        any(ex -> _depends(unwrap(ex), [slots.H; slots.Ud]), [an.flux[e]; an.remainder[e]; an.mass[e]]) && throw(
            ArgumentError("equation $e has a term the frequency-domain Maxwell path cannot place (a second derivative outside a divergence or a leftover time derivative); $_MAXWELL_FORM")
        )
        any(F -> !_iszero_expr(substitute(substitute(F, allslots), fixed)), an.flux[e]) && throw(
            ArgumentError("equation $e has a derivative of an expression without E, a source inside a divergence, which is not integrated by parts; write that source term out explicitly as a function of the coordinates")
        )
        any(ex -> _depends(unwrap(ex), state), parts(e)) && throw(nonlinear(e))
        any(ex -> _depends(unwrap(ex), [slots.T]), parts(e)) && throw(timedep("a coefficient"))
        any(ex -> _depends(unwrap(ex), s), [vec(B[e]); dG[e]; vec(dF[e])]) && throw(
            ArgumentError("equation $e has a time derivative inside a spatial derivative; time derivatives may only multiply E, as in ε ∂ₜ²E + σ ∂ₜE")
        )
        any(_depends(unwrap(Symbolics.derivative(ex, si)), s) for ex in dU[e], si in s) &&
            throw(ArgumentError("the coefficient of E is nonlinear in its time derivatives"))
    end
    sources = [-substitute(an.remainder[e], Dict(unwrap(g) => 0 for g in state)) for e in 1:3]
    any(f -> _depends(unwrap(f), [slots.T]), sources) && throw(timedep("a source"))

    values_at = Symbolics.build_function(
        [substitute(ex, fixed) for e in 1:3 for ex in parts(e)], slots.X..., s...; expression = Val(false)
    )[1]
    points = [(x, sv) for x in xs for sv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    function unpack(x, sv)
        v = _real_values(values_at(x..., sv...), "a coefficient")
        return map(0:2) do k
            w = v[(48k + 1):(48k + 48)]
            (B = reshape(w[1:27], 3, 3, 3), λU = w[28:30], dG = w[31:39], dF = reshape(w[40:48], 3, 3))
        end
    end
    comps = zeros(Int, 3)
    # Which component an equation tests: its Laplacian-like trace, or else its reaction, is nonzero (exactly: the choice
    # is not a tolerance on the size of the other coefficients).
    for (x, sv) in points, (e, q) in enumerate(unpack(x, sv))
        all(!iszero, comps) && break
        comps[e] == 0 || continue
        tr = [sum(q.B[a, c, a] for a in 1:3) for c in 1:3]
        comps[e] = maximum(abs, tr) > 0 ? argmax(abs.(tr)) : maximum(abs, q.λU) > 0 ? argmax(abs.(q.λU)) : 0
    end
    e0 = findfirst(iszero, comps)
    e0 === nothing || throw(ArgumentError("equation $e0 has neither a curl-curl nor a reaction term in E"))
    sort(comps) == 1:3 || throw(ArgumentError("the equations test the components $comps of E; write one equation per component"))
    # The structure of the second-order part, decided symbolically from its own coefficients: a term is there or not,
    # however small it is next to ν.
    νs = Any[]
    for e in 1:3
        ce = comps[e]
        other = mod1(ce + 1, 3)
        Be = [substitute(B[e][a, c, b], fixed) for a in 1:3, c in 1:3, b in 1:3]
        α, β, γ = Be[other, ce, other], Be[other, other, ce], Be[ce, other, other]
        _iszero_expr(γ) ||
            throw(ArgumentError("equation $e has a grad div E term, which has no meaning on Nédélec elements; $_MAXWELL_FORM"))
        expect = [α * (c == ce) * (a == b) + β * (b == ce) * (a == c) + γ * (a == ce) * (c == b) for a in 1:3, c in 1:3, b in 1:3]
        all(_iszero_expr(Be[a, c, b] - expect[a, c, b]) for a in 1:3, c in 1:3, b in 1:3) || throw(
            ArgumentError("the second-order part of equation $e is anisotropic or couples the components of E differently from curl(ν*curl(E)); $_MAXWELL_FORM")
        )
        _iszero_expr(β) && !_iszero_expr(α) && throw(
            ArgumentError("equation $e has the vector Laplacian of E, which is component-wise H¹ and not H(curl)-conforming; $_MAXWELL_FORM")
        )
        _iszero_expr(α + β) || throw(
            ArgumentError("the second-order part of equation $e adds a component-wise Laplacian of E to curl(ν*curl(E)), which is not H(curl)-conforming; $_MAXWELL_FORM")
        )
        push!(νs, β)
    end
    νsame = [_iszero_expr(νs[e] - νs[1]) for e in 1:3]
    for (x, sv) in points
        nus = [unpack(x, sv)[e].B[mod1(comps[e] + 1, 3), mod1(comps[e] + 1, 3), comps[e]] for e in 1:3]
        all(e -> νsame[e] || nus[e] == nus[1], 2:3) || throw(
            ArgumentError("ν differs between the component equations at x = $(Tuple(x)), so the medium is anisotropic; the coefficients must be scalars")
        )
    end
    _check_lower_order(an, s, fixed, xs, dU, dG, dF, comps)
    ce = comps[1]
    nu, λ = B[1][mod1(ce + 1, 3), mod1(ce + 1, 3), ce], dU[1][ce]
    a = (a0 = substitute(λ, Dict(unwrap(x) => 0 for x in s)), a1 = Symbolics.derivative(λ, s[1]), a2 = Symbolics.derivative(λ, s[2]))
    return (; nu, a..., sources = sources[invperm(comps)])
end

# The terms of lower order than the curl-curl, each judged on its own, with exact symbolic zero tests rather than
# tolerances relative to the other coefficients, which a large one would hide behind: there is no first-order term,
# the coefficients of E, ∂ₜE and ∂ₜ²E of an equation do not couple the components, and each is the same scalar
# function in every equation.
function _check_lower_order(an, s, fixed, xs, dU, dG, dF, comps)
    X = an.slots.X
    szero = Dict(unwrap(si) => 0 for si in s)
    orders = (ex -> substitute(ex, szero), ex -> Symbolics.derivative(ex, s[1]), ex -> Symbolics.derivative(ex, s[2]))
    # The indices of `exs` that are not zero: not symbolically zero and, at some point, not exactly zero. There is no
    # tolerance, since any tolerance relative to other coefficients (or to a unit) would depend on how the equation is
    # scaled and could hide a significant term; a coefficient that is zero only up to roundoff is rejected.
    function nonvanishing(exs)
        exs = [substitute(ex, fixed) for ex in exs]
        open = findall(ex -> !_iszero_expr(ex), exs)
        isempty(open) && return open
        g = Symbolics.build_function(exs[open], X...; expression = Val(false))[1]
        big = falses(length(open))
        for x in xs
            big .|= .!iszero.(_real_values(g(x...), "a coefficient"))
        end
        return open[big]
    end
    for e in 1:3
        isempty(nonvanishing(dF[e])) ||
            throw(ArgumentError("equation $e has a first-order term inside a derivative; $_MAXWELL_FORM"))
        isempty(nonvanishing(dG[e])) || throw(
            ArgumentError("equation $e has a first-order term in the derivatives of E, which curl(ν*curl(E)) does not have; $_MAXWELL_FORM")
        )
        off = [order(dU[e][c]) for c in setdiff(1:3, comps[e]) for order in orders]
        isempty(nonvanishing(off)) || throw(
            ArgumentError("the zeroth-order term of equation $e couples the components of E; the coefficients of E and its time derivatives must be scalars")
        )
    end
    diag = [orders[k](dU[e][comps[e]]) for e in 1:3, k in 1:3]
    f = Symbolics.build_function(substitute.(diag, Ref(fixed)), X...; expression = Val(false))[1]
    # Equal as expressions, or else equal as numbers: no tolerance, which would scale with coefficients (a large
    # reaction against a large inertia) that cancel at the requested frequency.
    same = [_iszero_expr(substitute(diag[e, k] - diag[1, k], fixed)) for e in 1:3, k in 1:3]
    for x in xs
        v = reshape(_real_values(f(x...), "a coefficient"), 3, 3)
        for k in 1:3, e in 2:3
            same[e, k] || v[e, k] == v[1, k] || throw(
                ArgumentError("the coefficient of E differs between the component equations at x = $(Tuple(x)), so the medium is anisotropic; the coefficients must be scalars")
            )
        end
    end
    return nothing
end

const _PEC_FORM = "write the perfect conductor as cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0"
const _ABC_FORM = "write the absorbing condition as cross(n, ν * curl(Eb, X)) .~ -sqrt(ε / μ) .* cross(n, cross(n, Dt.(Eb))) with n = facet_normal(Γ), Eb = [E(t, Γ)[c] for c in 1:3] and the ν = 1/μ and ε (the coefficient of ∂ₜ²E) of the cells next to Γ"

# The trace of the field in a boundary condition is E(t, Γ): the time variable and the boundary piece of the condition,
# nothing at a fixed or shifted time (an initial condition) and no other argument.
function _check_trace_arguments(ex, t, Γ, eq)
    args = [_unwrap_arg(a) for a in SymbolicUtils.arguments(ex)]
    length(args) == 2 && isequal(args[1], unwrap(t)) && args[2] isa PDEBase.BoundaryPiece && args[2] == Γ ||
        throw(ArgumentError("$eq has the trace $(Num(ex)); a boundary condition with omega must use E(t, Γ) at the time variable t and its boundary piece Γ, not a condition at one time"))
    return nothing
end

# Slot form of a boundary residual: the trace of E_c becomes U[c], ∂ₜ of it V[c], ∂ⱼ of it G[c, j] and component d of
# the facet normal N[d]. A derivative of anything but a trace, of order above one, or a normal derivative is an error.
function _boundary_rules!(rules, ex, eq, slots, ctx)
    SymbolicUtils.iscall(ex) || return rules
    op = SymbolicUtils.operation(ex)
    if op isa Differential
        inner = _unwrap_arg(only(SymbolicUtils.arguments(ex)))
        f = SymbolicUtils.iscall(inner) ? findfirst(isequal(SymbolicUtils.operation(inner)), ctx.ops) : nothing
        f !== nothing && op.order == 1 || throw(
            ArgumentError("$eq has the derivative $(Num(ex)); a boundary condition with omega may only take first derivatives of the components of E(t, Γ) themselves; $_ABC_FORM")
        )
        _check_trace_arguments(inner, ctx.t, ctx.piece, eq)
        j = findfirst(isequal(op.x), ctx.sivs)
        if isequal(op.x, ctx.t)
            rules[ex] = unwrap(slots.V[ctx.fields[f]])
        elseif j !== nothing
            rules[ex] = unwrap(slots.G[ctx.fields[f], j])
        else
            throw(ArgumentError("$eq has a derivative in $(op.x), which is neither time nor a coordinate"))
        end
    elseif op isa PDEBase.NormalDerivative
        throw(ArgumentError("$eq has a normal derivative; with omega the boundary conditions are the perfect conductor ($_PEC_FORM) and the first-order absorbing condition ($_ABC_FORM)"))
    elseif op isa _FacetNormal
        rules[ex] = unwrap(slots.N[Int(SymbolicUtils.unwrap_const(only(SymbolicUtils.arguments(ex))))])
    elseif (f = findfirst(isequal(op), ctx.ops)) !== nothing
        _check_trace_arguments(ex, ctx.t, ctx.piece, eq)
        rules[ex] = unwrap(slots.U[ctx.fields[f]])
    else
        foreach(a -> _boundary_rules!(rules, _unwrap_arg(a), eq, slots, ctx), SymbolicUtils.arguments(ex))
    end
    return rules
end

# The residual r of n × (ν curl E) = -c n × (n × ∂ₜE), as the 3×15 matrix acting on [vec(G); V; U].
function _absorbing_target(n, ν, c)
    return reduce(
        hcat, map(1:15) do k
            w = zeros(15)
            w[k] = 1
            G, V = reshape(w[1:9], 3, 3), w[10:12]
            curlE = [G[3, 2] - G[2, 3], G[1, 3] - G[3, 1], G[2, 1] - G[1, 2]]
            cross(n, ν .* curlE) + c .* cross(n, cross(n, V))
        end
    )
end

# The value of the coefficient expression `ex` at the normal `normal` (a Dict from the normal's slots to numbers) with
# a running forward bound on its rounding error: err(a ± b) = err(a) + err(b) + ε|a ± b|, err(a b) = |a| err(b) +
# |b| err(a) + ε|a b|, and likewise for quotients and powers. The normal and the constants of the equation are
# uncertain by a rounding each (a constant by half an ulp, the normal of a facet by an ulp), so a coefficient is as
# uncertain as its evaluation, not as large as its terms.
function _value_error(ex, normal)
    if SymbolicUtils.iscall(ex)
        op, args = SymbolicUtils.operation(ex), SymbolicUtils.arguments(ex)
        parts = [_value_error(a, normal) for a in args]
        if op === (+) || op === (-)
            v, e = parts[1]
            if op === (-) && length(parts) == 1
                return -v, e
            end
            for (w, f) in parts[2:end]
                s = op === (+) ? v + w : v - w
                v, e = s, e + f + eps() * abs(s)
            end
            return v, e
        elseif op === (*)
            v, e = parts[1]
            for (w, f) in parts[2:end]
                p = v * w
                v, e = p, abs(v) * f + abs(w) * e + eps() * abs(p)
            end
            return v, e
        elseif op === (/)
            (v, e), (w, f) = parts
            q = v / w
            return q, (e + abs(q) * f) / abs(w) + eps() * abs(q)
        elseif op === (^)
            (v, e), (k, _) = parts
            p = v^k
            return p, abs(k * v^(k - 1)) * e + eps() * abs(p)
        end
        f = op(first.(parts)...)
        return f, eps() * abs(f) + sum(last, parts) * max(1, abs(f))
    end
    value = SymbolicUtils.unwrap_const(ex)
    value isa Number && return Float64(value), eps() / 2 * abs(value)  # a decimal constant is rounded
    return normal[ex], eps() * abs(normal[ex])  # the normal of an oblique facet carries its own roundoff
end

# An orthonormal basis of the row space of M, as rows. Each row is scaled to unit length first, so that the result
# does not depend on how the equations are scaled; a row is zero only when all its entries are exactly zero, which
# the caller makes of the entries that are roundoff of their own cancelling terms (see `_magnitude`), so a tiny
# factor that is really there (1e-13, 1e-20) still makes a row.
function _rowspace(M)
    keep = findall(i -> norm(M[i, :]) > 0, axes(M, 1))
    M = M[keep, :] ./ norm.(eachrow(M[keep, :]))
    F = svd(M)
    return F.Vt[1:count(>(1.0e-10 * maximum(F.S; init = 0.0)), F.S), :]
end

# Generic unit directions, at which no row of a boundary condition vanishes for a reason of geometry.
const _GENERIC_NORMALS = [normalize([1.0, 2.0, 3.0]), normalize([-2.0, 1.0, 0.5]), normalize([0.3, -1.0, 2.0])]

# The boundary conditions on each piece: the perfect conductor n × E = 0 (the residuals are linear in the traces and
# the normal, tangential (C n = 0) and of rank 2 together) or, when they hold derivatives of the traces, the
# first-order absorbing condition, whose residuals must span exactly the rows of n × (ν curl E) + c n × (n × ∂ₜE)
# with ν and c = sqrt(ε ν) of the adjacent lossless cell. Both are checked at the actual normal of every facet of the
# piece. `cells` holds ν, a₀, a₁ and a₂ per cell, `ω` is the angular frequency.
function _recognise_boundaries(bcs, fields, pvals, ctx, domain, cells, ω)
    slots = (
        U = [Symbolics.variable(:U, c) for c in 1:3], V = [Symbolics.variable(:V, c) for c in 1:3],
        G = [Symbolics.variable(:G, c, j) for c in 1:3, j in 1:3], N = [Symbolics.variable(:N, d) for d in 1:3],
    )
    W = [vec(slots.G); slots.V; slots.U]
    byp = Dict{Symbol, Vector{Tuple{Equation, Num}}}()
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
        rules = _boundary_rules!(Dict{Any, Any}(), unwrap(res), eq, slots, (; ctx..., fields, piece = Γ))
        for (p, value) in zip(ctx.params, pvals)
            rules[p] = value
        end
        B = substitute(res, rules)
        # Coefficients may depend on the normal but not on the coordinates, so that each facet is checked whole.
        extra = [x for x in Symbolics.get_variables(B) if !any(isequal(x), unwrap.([W; slots.N]))]
        isempty(extra) || throw(ArgumentError("$eq depends on $(join(Num.(extra), ", ")); $_PEC_FORM, or $_ABC_FORM"))
        push!(get!(byp, Γ.name, Tuple{Equation, Num}[]), (eq, B))
    end
    check_facetsets(domain.mesh, collect(keys(byp)); disjoint = false)
    pec, absorbing = String[], String[]
    for (name, eqBs) in byp
        Bs = last.(eqBs)
        C = [Symbolics.derivative(B, w) for B in Bs, w in W]
        isabc = any(!_iszero_expr, C[:, 1:12])
        form = isabc ? _ABC_FORM : _PEC_FORM
        any(ex -> _depends(unwrap(ex), W), C) && throw(ArgumentError("the conditions on $name are nonlinear in E; $form"))
        any(B -> !_iszero_expr(substitute(B, Dict(unwrap(w) => 0 for w in W))), Bs) && throw(
            ArgumentError(isabc ? "the absorbing condition on $name has a source (a term without E), which is not supported; drive the field with a WavePort or a volume source; $form" : "the conditions on $name prescribe a nonzero tangential field, which is not supported; $form")
        )
        C_at = Symbolics.build_function(vec(C), slots.N...; expression = Val(false))[1]
        for (n, facet) in _facet_normals(domain.mesh, String(name))
            # The roundoff of a normal that lies in a coordinate plane is not a coefficient: snap it to exactly 0, so
            # that n[1] * E vanishes there while n[1] + 1e-16 does not.
            n = [abs(x) < 1.0e-14 ? 0.0 : x for x in n]
            Cn = reshape(_real_values(C_at(n...), "a coefficient of the condition"), length(Bs), 15)
            # An entry is zero when it is within a few ulps of its own evaluation error (see `_value_error`): a residue
            # of cancelling terms is no larger than that error, whereas a coefficient that is large only through its
            # terms (1e15 (n[1] + 0.25 n[3]) + 1) is as certain as its error and keeps its real part. The factor 4
            # allows for the order of the compiled evaluation differing from that of the bound.
            normal = Dict(unwrap(slots.N[d]) => n[d] for d in 1:3)
            for i in eachindex(Bs), j in 1:15
                value, error = _value_error(unwrap(C[i, j]), normal)
                abs(value) <= 4error && (Cn[i, j] = 0.0)
            end
            where = "on $name at facet $((facet[1], facet[2])) with normal $(Tuple(round.(n; digits = 6)))"
            if !isabc
                Cn = Cn[:, 13:15]
                # Each equation is judged by its own size, so that one written with a large coefficient hides no other.
                Cn = Cn[[norm(Cn[i, :]) > 0 for i in axes(Cn, 1)], :]
                Cn = Cn ./ norm.(eachrow(Cn))
                maximum(abs, Cn * n; init = 0.0) > 1.0e-10 && throw(
                    ArgumentError("the conditions $where constrain the normal component of E, unlike n × E = 0; $form")
                )
                rank(Cn) == 2 || throw(
                    ArgumentError("the conditions $where constrain $(rank(Cn)) of the two tangential components of E; $form")
                )
                continue
            end
            (; nu, a0, a1, a2) = cells[facet[1]]
            a0 == 0 && a1 == 0 && a2 > 0 && nu > 0 || throw(
                ArgumentError("the absorbing condition $where borders cell $(facet[1]) with ν = $nu, a₂ = $a2, a₁ = $a1 and a₀ = $a0; the time-domain absorbing condition needs a lossless cell (a₁ = a₀ = 0, a₂ = ε > 0, ν = 1/μ > 0), so build a MaxwellModel with `absorbing` for a lossy one")
            )
            c = sqrt(a2 * nu)
            # The columns act on G ~ k E, V = ∂ₜE ~ ω E and U = E; in these units the curl and impedance parts are of
            # the same size (ν k = c ω) whatever the unit system, where the raw coefficients can differ by many orders.
            scale = [fill(ω * sqrt(a2 / nu), 9); fill(ω, 3); ones(3)]'
            # A row that vanishes (all its entries zero above) is no equation; a term in E counts in a surviving row.
            has_e_term = any(i -> any(!iszero, Cn[i, 13:15]), eachindex(Bs))
            Q = _rowspace(Cn .* scale)
            # The absorbing condition has two equations, its tangential components.
            axis = argmin(abs.(n))
            t1 = normalize(cross(n, [k == axis for k in 1:3]))
            tangents = [t1'; cross(n, t1)']
            P = _rowspace((tangents * _absorbing_target(n, nu, c)) .* scale)
            size(Q, 1) == 2 || throw(
                ArgumentError("the conditions $where have $(size(Q, 1)) independent equations; the absorbing condition n × (ν curl E) = -sqrt(ε/μ) n × (n × ∂ₜE) has 2; $form")
            )
            # A term in E has the weight of E itself, which the field units of the derivative columns can make small.
            (has_e_term || norm(Q' * Q - P' * P) > 1.0e-8) && throw(
                ArgumentError("the conditions $where are not n × (ν curl E) = -sqrt(ε/μ) n × (n × ∂ₜE) of the adjacent cell (ν = $nu, sqrt(ε/μ) = $c): the coefficient of curl E, the impedance or the sign differs, or a term in E is added; $form")
            )
        end
        push!(isabc ? absorbing : pec, String(name))
    end
    # Overlapping perfect-conductor pieces only constrain the same dofs again; any other overlap would add a
    # boundary term twice or give a facet two conditions.
    grid = domain.mesh
    pieces = [sort(pec); sort(absorbing)]
    for i in eachindex(pieces), j in (i + 1):lastindex(pieces)
        a, b = pieces[i], pieces[j]
        (a in absorbing || b in absorbing) && !isdisjoint(Ferrite.getfacetset(grid, a), Ferrite.getfacetset(grid, b)) &&
            throw(ArgumentError("the pieces $a and $b share facets and one of them is absorbing; give each facet one condition"))
    end
    return (; pec = sort!(pec), absorbing = sort!(absorbing))
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
`∫ f ⋅ Nᵢ`. A complex permittivity has no real time-domain form here; build a `MaxwellModel` directly for it.

With `n = facet_normal(Γ)` and `Eb = [E(t, Γ)[c] for c in 1:3]`, the boundary conditions on a piece `Γ` are

  - the perfect conductor `cross(n, Eb) .~ 0`, the model's `pec`;
  - the first-order absorbing (Silver-Müller) condition `cross(n, ν * curl(Eb, X)) .~ -sqrt(ε / μ) .*
    cross(n, cross(n, Dt.(Eb)))`, with `ε = a₂` and `ν = 1/μ` of the cells next to `Γ`, the model's `absorbing`.
    `Dt` becomes `iω`, so this is the model's `n × (μr⁻¹ ∇ × E) = -i k0 sqrt(εr/μr) n × (n × E)` with `k0 = ω`.
    Any equivalent set of equations is accepted (a multiple, the other cross order, a linear combination):
    at every facet the conditions must span exactly the two tangential equations of this one. The cells next to
    `Γ` must be lossless (`a₁ = a₀ = 0`); a mismatched coefficient, impedance or sign is an error, since the
    model implements only the matched condition.

Pieces without a condition get the natural `n × (ν curl E) = 0`, except those carrying the
[`WavePort`](@ref)s of `disc.ports`, which must have no condition. With ports, `a₀` must be zero, so that the
model holds at every frequency and `solve_sweep(model, omegas)` is the sweep of the `PDESystem`.
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
    fixed = Dict{Any, Any}(unwrap(an.slots.P[j]) => pvals[j] for j in keep)
    centroids = [sum(Ferrite.getcoordinates(grid, cell)) / 4 for cell in 1:Ferrite.getncells(grid)]
    coeffs = _recognise_volume(an, slot_s, fixed, centroids)

    # Compile the coefficients and sources as functions of the mesh coordinate.
    X = an.slots.X
    compile(exs) = Symbolics.build_function([substitute(ex, fixed) for ex in exs], X...; expression = Val(false))[1]
    coefficient_at = compile([coeffs.nu, coeffs.a0, coeffs.a1, coeffs.a2])
    source_at = compile(coeffs.sources)
    cells = map(enumerate(centroids)) do (cell, x)
        nu, a0, a1, a2 = _real_values(coefficient_at(x...), "a coefficient")
        (iszero(nu) || iszero(1 / nu) || !isfinite(1 / nu)) &&
            throw(ArgumentError("ν = $nu in cell $cell; ν = 1/μ must be nonzero" * (iszero(nu) ? "" : " and 1/ν must be representable")))
        (; nu, a0, a1, a2)
    end
    (; pec, absorbing) = _recognise_boundaries(
        sys.bcs, comps, pvals[keep], (; ops, params = ctx.params[keep], sivs = ctx.sivs, t = ctx.t), disc.domain, cells, disc.omega
    )
    ports = disc.ports
    if !isempty(ports)
        # A port's modal data and the sweep use the cell materials at every frequency, so they must not depend on ω.
        cell = findfirst(c -> !iszero(c.a0), cells)
        cell === nothing || throw(
            ArgumentError("a₀ = $(cells[cell].a0) in cell $cell, but with wave ports the coefficient of E must be zero: eps_r = a₂ - a₀/ω² would depend on the frequency, which the ports and solve_sweep do not follow")
        )
        for port in ports, name in [pec; absorbing]
            _check_facetset(grid, port.facetset)
            isdisjoint(Ferrite.getfacetset(grid, port.facetset), Ferrite.getfacetset(grid, name)) || throw(
                ArgumentError("the wave port on $(port.facetset) is the boundary condition of its facets, but the PDESystem also has a condition on $name, which shares facets with it; remove that condition")
            )
        end
    end
    ω = disc.omega
    materials = map(enumerate(cells)) do (cell, c)
        # The net reaction a₀ - ω² a₂ is formed exactly (it cancels when the two are large), then divided once.
        net = big(c.a0) - big(ω)^2 * big(c.a2)
        eps_r = Float64(-net / big(ω)^2)
        # The coefficient and the reaction -ω² eps_r that assembly forms from it must neither overflow nor lose a nonzero
        # net reaction to underflow.
        reaction = -ω^2 * eps_r
        lost(x) = iszero(x) || issubnormal(x)
        isfinite(eps_r) && isfinite(reaction) && (iszero(net) || !(lost(eps_r) || lost(reaction))) || throw(
            ArgumentError("the reaction a₀ - ω² a₂ = $(Float64(net)) ($(c.a0) - $ω^2 $(c.a2)) in cell $cell is not representable as ε = a₂ - a₀/ω² at this ω; use a frequency nearer 1 or a MaxwellModel directly")
        )
        MaxwellMaterial(; eps_r, mu_r = 1 / c.nu, sigma = c.a1)
    end
    model = MaxwellModel(
        grid; materials, pec, absorbing, ports, eps0 = 1.0, mu0 = 1.0, length_scale = 1.0,
        quadrature = something(disc.quadrature, 2),
    )
    return (; model, omega = ω, load = _source_load(model, x -> _real_values(source_at(x...), "a source")))
end

"""
    solve_sweep(pdesys::PDESystem, disc::FEMDiscretization, omegas; kwargs...)

The wave-port scattering matrices of the `PDESystem` at the angular frequencies `omegas`: `solve_sweep` of
the model of [`maxwell_model`](@ref)`(pdesys, disc)`, which `disc.ports` drive. The `PDESystem` must have no
source, since the sweep excites the ports only; `disc.omega` builds the model but does not enter the result.
"""
function solve_sweep(pdesys::ModelingToolkit.PDESystem, disc::FEMDiscretization, omegas; kwargs...)
    isempty(disc.ports) &&
        throw(ArgumentError("solve_sweep needs wave ports; give them as FEMDiscretization(Ω, t; omega, spaces, ports = [WavePort(...)])"))
    (; model, load) = maxwell_model(pdesys, disc)
    any(!iszero, load) && throw(
        ArgumentError("solve_sweep drives the wave ports only, but the PDESystem has a source; solve a source with discretize(pdesys, disc) at each omega")
    )
    return solve_sweep(model, omegas; kwargs...)
end

"""
    WavePort(Γ::BoundaryPiece; a_direction, e_direction = nothing, deembed = 0.0)

The [`WavePort`](@ref) on the facet set of the boundary piece `Γ`, for `FEMDiscretization(…; ports)`.
"""
WavePort(Γ::PDEBase.BoundaryPiece; kwargs...) = WavePort(String(Γ.name); kwargs...)

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
