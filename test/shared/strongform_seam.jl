# Test-only stand-in for `PDEBase.StrongForm.analyse` (spec §4.4), with the data types of spec §5.2.
# Ported from the verified dauntless prototype (`split_time`, `split_divergence`, `to_placeholders`,
# `match_flux`). Swap for PDEBase.StrongForm when PDEBase releases it; FiniteElementMethod reads the
# analysis by field only.
module StrongFormSeam

using Symbolics
using Symbolics: unwrap, wrap, Differential, expand_derivatives, substitute
using Symbolics.SymbolicUtils: iscall, operation, arguments

"State, ∇state (field-major `G[(f-1)*dim+d]`), ∂ₜstate, Hessian, coords, time, normal, params."
struct StateSlots
    U::Vector{Num}
    G::Vector{Num}
    Ud::Vector{Num}
    H::Vector{Num}
    X::Vector{Num}
    T::Num
    N::Vector{Num}
    P::Vector{Num}
end
function StateSlots(nf, dim, np)
    return StateSlots(
        Symbolics.variables(:U, 1:nf), Symbolics.variables(:G, 1:(nf * dim)),
        Symbolics.variables(:Ud, 1:nf), Symbolics.variables(:H, 1:(nf * dim * dim)),
        Symbolics.variables(:X, 1:dim), Symbolics.variable(:T),
        Symbolics.variables(:N, 1:dim), Symbolics.variables(:P, 1:max(np, 1))
    )
end

struct EssentialBC
    piece::Any
    field::Int
    g::Num
end
# Periodic pairing of spec §5.2: `field` on `mirror` equals `field` on `master` translated by `shift`.
struct PeriodicBC
    master::Any
    mirror::Any
    field::Int
    shift::Any
end
struct NaturalBC
    piece::Any
    eq::Int
    Φ::Num
end

struct StrongFormAnalysis
    slots::StateSlots
    nfields::Int
    dim::Int
    strong::Vector{Num}
    mass::Vector{Num}
    flux::Vector{Vector{Num}}
    remainder::Vector{Num}
    conormal::Vector{Vector{Num}}
    essential::Vector{EssentialBC}
    natural::Vector{NaturalBC}
    periodic::Vector{Any}
end

# Integration-by-parts policies (spec §5.1, §5.2). `Weighted(w)` multiplies the equation by `w` before splitting.
struct ProductRule end
struct Weighted
    w::Any
end
weighted(eq, ::ProductRule) = eq
weighted(eq, policy::Weighted) = policy.w * eq.lhs ~ policy.w * eq.rhs

"Boundary condition `eq` on `piece`; `normal` is a constant outward normal or `nothing` (symbolic N)."
struct BC
    piece::Any
    eq::Equation
    normal::Union{Nothing, Vector{Float64}}
end
BC(piece, eq) = BC(piece, eq, nothing)

addterms(ex) = iscall(ex) && operation(ex) === (+) ? reduce(vcat, addterms.(arguments(ex))) : [ex]
function factors(ex)
    iscall(ex) && operation(ex) === (*) && return collect(arguments(ex))
    iscall(ex) && operation(ex) === (/) && return vcat(factors(arguments(ex)[1]), [1 / arguments(ex)[2]])
    return [ex]
end
isD(ex) = iscall(ex) && operation(ex) isa Differential
isD(ex, ivs) = isD(ex) && any(iv -> isequal(operation(ex).x, iv), ivs)
depends(ex, syms) = !isempty(intersect(Set(Symbolics.get_variables(ex)), Set(unwrap.(syms))))

function derivative_order(ex, sivs)
    iscall(ex) || return 0
    isD(ex, sivs) && return operation(ex).order + derivative_order(arguments(ex)[1], sivs)
    return maximum((derivative_order(a, sivs) for a in arguments(ex)); init = 0)
end
function has_time_derivative(ex, t)
    return iscall(ex) &&
        ((isD(ex) && isequal(operation(ex).x, t)) || any(a -> has_time_derivative(a, t), arguments(ex)))
end

"Stage a: per-field mass coefficients and the spatial terms of `e = lhs - rhs`."
function split_time(eq, depvars, t)
    e = unwrap(Symbolics.expand(eq.lhs - eq.rhs))
    mass = Any[0 for _ in depvars]
    spatial = Any[]
    for term in addterms(e)
        if has_time_derivative(term, t)
            dt(u) = unwrap(Differential(t)(u))
            fi = findfirst(u -> any(f -> isequal(f, dt(u)), factors(term)), depvars)
            fi === nothing &&
                error("time derivative in $term is not ∂ₜu × coefficient (nested/higher-order in time unsupported)")
            m = prod(f for f in factors(term) if !isequal(f, dt(depvars[fi])); init = 1)
            has_time_derivative(m, t) && error("nonlinear in ∂ₜu: $term")
            mass[fi] += m
        else
            push!(spatial, term)
        end
    end
    return mass, spatial
end

"Stage b: `c·D_i(w′) = D_i(c·w′) − (D_i c)·w′`, giving flux components `G` and remainder terms `R`."
function split_divergence(spatial, sivs)
    G = Any[0 for _ in sivs]
    R = Any[]
    for term in spatial
        fs = factors(term)
        idx = findall(f -> isD(f, sivs) && derivative_order(f, sivs) >= 2, fs)
        if isempty(idx)
            derivative_order(term, sivs) >= 2 && error("second derivative not in a divergence-compatible position: $term")
            push!(R, term)
            continue
        end
        length(idx) > 1 && error("product of two second-order factors: $term")
        Dw = fs[idx[1]]
        op = operation(Dw)
        w = arguments(Dw)[1]
        i = findfirst(iv -> isequal(op.x, iv), sivs)
        inner = op.order == 1 ? w : unwrap(Differential(op.x, op.order - 1)(w))
        derivative_order(inner, sivs) >= 2 && error("fourth-or-higher-order operator needs mixed/C¹ formulation: $term")
        c = prod(fs[j] for j in eachindex(fs) if j != idx[1]; init = 1)
        derivative_order(c, sivs) >= 1 && error("coefficient depends on ∇u, product rule would give 2nd derivatives: $term")
        G[i] += c * inner
        push!(R, -expand_derivatives(Differential(op.x)(c)) * inner)
    end
    return G, R
end

hasD(ex) = iscall(ex) && (operation(ex) isa Differential || any(hasD, arguments(ex)))

#     to_slots(ex, s, depvars, sivs, t, params; strict = true)
#
# `ex` in slots. With `strict`, unresolved symbols are an error, and so is a registered function without a
# derivative rule whose argument state reaches (spec §5.1 as clarified by PDEBase: opaque functions of the
# coordinates are fine). The strong residual is converted with `strict = false`, since it keeps `∂(opaque(x))`
# of a divergence-form coefficient that the weak form never needs.
function to_slots(ex, s, depvars, sivs, t, params; strict = true)
    dim = length(sivs)
    rules1 = Dict()
    rules2 = Dict()
    for (f, u) in enumerate(depvars)
        for d in 1:dim, e in 1:dim
            rules1[unwrap(Differential(sivs[e])(Differential(sivs[d])(u)))] = s.H[((f - 1) * dim + d - 1) * dim + e]
        end
        for d in 1:dim
            rules1[unwrap(Differential(sivs[d], 2)(u))] = s.H[((f - 1) * dim + d - 1) * dim + d]
            rules1[unwrap(Differential(sivs[d])(u))] = s.G[(f - 1) * dim + d]
        end
        rules1[unwrap(Differential(t)(u))] = s.Ud[f]
        rules2[unwrap(u)] = s.U[f]
    end
    for d in 1:dim
        rules2[unwrap(sivs[d])] = s.X[d]
    end
    rules2[unwrap(t)] = s.T
    for (j, p) in enumerate(params)
        rules2[unwrap(p)] = s.P[j]
    end
    out = substitute(substitute(expand_derivatives(wrap(ex)), rules1), rules2)
    strict || return Num(out)
    known = Set(unwrap.(vcat(s.U, s.G, s.Ud, s.H, s.X, [s.T], s.P, s.N)))
    leftover = filter(v -> !(v in known), collect(Symbolics.get_variables(out)))
    isempty(leftover) || error("unresolved symbols $(leftover) in $out (unregistered coefficient derivative?)")
    for slot in vcat(s.U, s.G, s.Ud, s.H)
        depends(out, [slot]) && hasD(unwrap(Symbolics.derivative(out, slot))) &&
            error("registered function without a derivative rule is applied to state in $out; register its derivative")
    end
    return Num(out)
end

"Stage d: the boundary flux Φ substituted for `G·N` when `B = 0` is proportional to the conormal, else `nothing`."
function match_flux(Φ, B, s)
    g = s.G
    aΦ = [Symbolics.derivative(Φ, gi) for gi in g]
    aB = [Symbolics.derivative(B, gi) for gi in g]
    any(a -> depends(a, g), [aΦ; aB]) && return nothing
    i = findfirst(a -> !isequal(Symbolics.simplify(a), 0), aΦ)
    i === nothing && return nothing
    λ = Symbolics.simplify(aB[i] / aΦ[i])
    all(j -> isequal(Symbolics.simplify(aB[j] - λ * aΦ[j]), 0), eachindex(g)) || return nothing
    isequal(λ, 0) && return nothing
    zero_g = Dict(gi => 0 for gi in g)
    return Symbolics.simplify(substitute(Φ, zero_g) - substitute(B, zero_g) / λ)
end

"""
    analyse(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict(), periodic = []) -> StrongFormAnalysis

Equation `e` is paired with field `e` and integrated by parts under `get(ibp, e, ProductRule())`. `periodic` is a
list of `PeriodicBC`, passed through: the seam does not infer pairings from trace equations. `mass[e]` is the whole time-derivative part `Σ_f m_ef·Ud_f`.
"""
function analyse(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict(), periodic = [])
    nf, dim = length(depvars), length(sivs)
    s = StateSlots(nf, dim, length(params))
    P(ex) = to_slots(ex, s, depvars, sivs, t, params)
    strong, mass, flux, remainder, conormal = Num[], Num[], Vector{Num}[], Num[], Vector{Num}[]
    for (e, eq) in enumerate(eqs)
        eq = weighted(eq, get(ibp, e, ProductRule()))
        m, spatial = split_time(eq, depvars, t)
        Gs, R = split_divergence(spatial, sivs)
        mslots = [P(mf) for mf in m]
        any(mf -> depends(mf, vcat(s.U, s.G)), mslots) && error("state-dependent mass: needs residual-form DAE")
        G = [P(gi) for gi in Gs]
        push!(strong, to_slots(unwrap(eq.lhs - eq.rhs), s, depvars, sivs, t, params; strict = false))
        push!(mass, sum(mslots .* s.Ud))
        push!(flux, G)
        push!(remainder, sum((P(r) for r in R); init = Num(0)))
        push!(conormal, [Symbolics.derivative(sum(G .* s.N), gi) for gi in s.G])
    end
    essential, natural = boundary_conditions(bcs, flux, s; depvars, t, sivs, params)
    return StrongFormAnalysis(s, nf, dim, strong, mass, flux, remainder, conormal, essential, natural, collect(Any, periodic))
end

# Stage d alone: essential data and matched natural fluxes of `bcs` against the per-equation `flux`, both in the
# slots `s` (any object with the `StateSlots` fields, such as PDEBase's). The Integration chain uses it on top of
# PDEBase.StrongForm.analyse, which does not analyse boundary conditions yet.
function boundary_conditions(bcs, flux, s; depvars, t, sivs, params = [])
    nf, dim = length(depvars), length(sivs)
    P(ex) = to_slots(ex, s, depvars, sivs, t, params)
    essential, natural = EssentialBC[], NaturalBC[]
    for bc in bcs
        B = P(unwrap(bc.eq.lhs - bc.eq.rhs))
        if !depends(B, s.G)
            f = findfirst(fi -> depends(B, [s.U[fi]]), 1:nf)
            a = Symbolics.derivative(B, s.U[f])
            depends(a, s.U) && error("nonlinear Dirichlet BC $(bc.eq)")
            push!(essential, EssentialBC(bc.piece, f, Symbolics.simplify(-substitute(B, Dict(s.U[f] => 0)) / a)))
            continue
        end
        nsub = bc.normal === nothing ? Dict() : Dict(s.N[d] => bc.normal[d] for d in 1:dim)
        e = findfirst(G -> match_flux(substitute(sum(G .* s.N), nsub), substitute(B, nsub), s) !== nothing, flux)
        e === nothing &&
            error("BC $(bc.eq) on $(bc.piece) is not a conormal (natural) condition for any equation's flux")
        Φ = match_flux(substitute(sum(flux[e] .* s.N), nsub), substitute(B, nsub), s)
        push!(natural, NaturalBC(bc.piece, e, Φ))
    end
    return essential, natural
end

end
