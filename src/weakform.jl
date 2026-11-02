# Stage e of the strong-to-weak transform (spec §5.1, §5.2): Galerkin projection of a strong-form
# analysis and classification of each additive term into :source, :bilinear or :nonlinear.
#
# The analysis is read duck-typed by field (`slots`, `dim`, `mass`, `flux`, `remainder`, `natural`), so
# `PDEBase.StrongForm.StrongFormAnalysis` can be passed in once it exists without changes here.

const SU = Symbolics.SymbolicUtils

"""
    TestSlots(V, W, hK)
    TestSlots(dim)

Symbolic test-function slots: the test function `V`, its gradient `W[1:dim]` and the element size `hK`.
"""
struct TestSlots
    V::Num
    W::Vector{Num}
    hK::Num
end
TestSlots(dim::Integer) = TestSlots(Symbolics.variable(:V), [Symbolics.variable(:W, i) for i in 1:dim], Symbolics.variable(:hK))

"""
    VolumeTerm(eq, kind, integrand, timedep, paramdep)

One volume integrand of equation `eq`, of `kind` `:mass`, `:bilinear`, `:source` or `:nonlinear`.
"""
struct VolumeTerm
    eq::Int
    kind::Symbol
    integrand::Num
    timedep::Bool
    paramdep::Bool
end

"""
    FacetTerm(eq, piece, kind, integrand, timedep, paramdep)

One facet integrand of equation `eq` on the boundary `piece` (an opaque key).
"""
struct FacetTerm{P}
    eq::Int
    piece::P
    kind::Symbol
    integrand::Num
    timedep::Bool
    paramdep::Bool
end

"""
    WeakForm(analysis, test, volume, facet)

Galerkin weak form with residual convention `Σ ∫ volume + Σ ∮ facet = 0`. Essential conditions stay in
`analysis.essential`.
"""
struct WeakForm{A}
    analysis::A
    test::TestSlots
    volume::Vector{VolumeTerm}
    facet::Vector{FacetTerm}
end

"""
    KernelSpec(eq, kind, piece, f, tangent, coeffs = nothing)

Quadrature-point kernel `f(U, G, Ud, V, W, X, T, N, P)` of equation `eq`; `piece` is `nothing` for volume
kernels and `tangent` is `nothing` unless `kind === :nonlinear`.

`coeffs`, for the linear kinds, lets the driver contract coefficients with the shape functions once per quadrature
point instead of calling `f` per test and trial function; `f` may then be `nothing`. Called as
`coeffs(X, T, N, P)`, it returns a tuple indexed by `(s - 1) * (1 + dim) + w` for the test slots `w` in `[V; W]`:
- `:source`: `∂I/∂w` (one trial "slot");
- `:bilinear`: `∂²I/∂s∂w` for the trial slots `s` in `[U; G]`;
- `:mass`: `∂²I/∂s∂w` for the trial slots `s` in `Ud`.
Entries may be `StructuralZero()`, which the driver drops at compile time.
"""
struct KernelSpec{F, T, C}
    eq::Int
    kind::Symbol
    piece::Any
    f::F
    tangent::T
    coeffs::C
end
KernelSpec(eq, kind, piece, f, tangent) = KernelSpec(eq, kind, piece, f, tangent, nothing)

addterms(ex) = SU.iscall(ex) && SU.operation(ex) === (+) ? reduce(vcat, addterms.(SU.arguments(ex))) : [ex]

function depends(ex, syms)
    vars = Set(Symbolics.get_variables(ex))
    return any(s -> Symbolics.unwrap(s) in vars, syms)
end

"The declared state slots of `slots`: `U`, `G` and, when the slot set has one, `H`."
function state_slots(slots)
    S = Num[slots.U; slots.G]
    if hasproperty(slots, :H) && slots.H !== nothing
        append!(S, slots.H)
    end
    return S
end

"""
    classify(integrand, slots, test)

Split the expanded `integrand` into additive terms and bucket each as `:source` (no state slot),
`:bilinear` (every derivative with respect to a state slot is state-free) or `:nonlinear`. State slots
are the declared `slots.U`, `slots.G` and `slots.H` (if present). Returns `(kind, integrand, timedep,
paramdep)` named tuples, one per non-empty bucket, where `timedep` means the bucket contains `slots.T`
and `paramdep` that it contains a parameter slot of `slots.P`.
"""
function classify(integrand, slots, test)
    S = state_slots(slots)
    buckets = Dict(:bilinear => Num(0), :source => Num(0), :nonlinear => Num(0))
    for term in addterms(Symbolics.unwrap(Symbolics.expand(integrand)))
        kind = !depends(term, S) ? :source :
            all(s -> !depends(Symbolics.derivative(term, s), S), S) ? :bilinear : :nonlinear
        buckets[kind] += term
    end
    out = @NamedTuple{kind::Symbol, integrand::Num, timedep::Bool, paramdep::Bool}[]
    for kind in (:bilinear, :source, :nonlinear)
        ex = buckets[kind]
        isequal(ex, 0) && continue
        push!(out, (; kind, integrand = ex, flags(ex, slots)...))
    end
    return out
end

flags(ex, slots) = (timedep = depends(ex, [slots.T]), paramdep = depends(ex, slots.P))

"""
    weak_form(analysis; test = TestSlots(analysis.dim))

Galerkin projection of a strong-form analysis: equation `e` contributes `mass[e]·V` (kind `:mass`),
the integrated-by-parts volume integrand `−G·W + R·V`, and `Φ·V` on each natural condition's piece.
"""
function weak_form(analysis; test = TestSlots(analysis.dim))
    slots, V, W = analysis.slots, test.V, test.W
    volume = VolumeTerm[]
    facet = FacetTerm[]
    for e in eachindex(analysis.flux)
        m = analysis.mass[e]
        if !isequal(m, 0)
            push!(volume, VolumeTerm(e, :mass, m * V, flags(m, slots)...))
        end
        G = analysis.flux[e]
        integrand = -sum(G[d] * W[d] for d in eachindex(G)) + analysis.remainder[e] * V
        for c in classify(integrand, slots, test)
            push!(volume, VolumeTerm(e, c.kind, c.integrand, c.timedep, c.paramdep))
        end
    end
    for bc in analysis.natural
        for c in classify(bc.Φ * V, slots, test)
            push!(facet, FacetTerm(bc.eq, bc.piece, c.kind, c.integrand, c.timedep, c.paramdep))
        end
    end
    return WeakForm(analysis, test, volume, facet)
end
