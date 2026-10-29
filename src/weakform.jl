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

# `expand` leaves a quotient whole, so `(U·V + g·V) / r` (a Robin condition whose data has the
# denominator `r`, as normals on a circle do) is split over its numerator's terms.
function addterms(ex)
    SU.iscall(ex) || return [ex]
    op = SU.operation(ex)
    op === (+) && return reduce(vcat, addterms.(SU.arguments(ex)))
    if op === (/)
        num, den = SU.arguments(ex)
        return [term / den for term in addterms(num)]
    end
    return [ex]
end

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
    AbstractTestSpace

How `weak_form` tests each equation (spec §1 goal 4, §5.2): [`Galerkin`](@ref) or [`SUPG`](@ref).
"""
abstract type AbstractTestSpace end

"Galerkin testing: equation `e` is tested with the test function `v` of its paired field."
struct Galerkin <: AbstractTestSpace end

"""
    SUPG(τ; b = nothing)

Streamline-upwind Petrov–Galerkin testing: the Galerkin form plus `τ (b·∇v)` times the strong residual on every
cell, with the mass term tested by `v + τ b·∇v` too. `b` is the advection velocity, by default read from the
analysis as the coefficient of the equation's own gradient in its remainder, `b_d = ∂R/∂(∂_d u)`, which must be
free of the state (otherwise pass `b`, a vector of numbers or expressions). `τ` is a number, an expression, or a
function `τ(hK, b)` of the element size `hK` and `b`, e.g. `(hK, b) -> hK / (2 * sqrt(sum(abs2, b)))`.
Boundary terms stay Galerkin. The stabilised terms need the Hessian slots `H` and `hK`, so the driver's
`CellValues` must be built with `update_hessians = true`.
"""
struct SUPG{T, B} <: AbstractTestSpace
    τ::T
    b::B
end
SUPG(τ; b = nothing) = SUPG(τ, b)

# The extra test weight of a test space for equation `e` as `(τ, b·W)`, so the weight is `τ (b·W)`; `nothing` for
# Galerkin. `τ` is kept apart because it may hold a quotient that `expand` does not distribute (e.g. `hK / |b|`), so
# terms are classified without it and scaled by it afterwards; it is free of the state and of the test slots.
test_weight(::Galerkin, analysis, e, test) = nothing
function test_weight(space::SUPG, analysis, e, test)
    b = map(fold, space.b === nothing ? advection(analysis, e) : space.b)
    τ = space.τ isa Function ? space.τ(test.hK, b) : space.τ
    depends(Num(τ), [state_slots(analysis.slots); test.V; test.W]) &&
        throw(ArgumentError("SUPG: τ = $τ depends on the state or the test function"))
    return τ, sum(b[d] * test.W[d] for d in 1:(analysis.dim))
end

# A constant expression as a number, so `sqrt(sum(abs2, b))` of a constant `b` folds.
function fold(x)
    v = Symbolics.unwrap(x)
    return SU.isconst(v) ? Float64(SU.unwrap_const(v)) : x
end

function advection(analysis, e)
    slots, dim = analysis.slots, analysis.dim
    b = [Symbolics.derivative(analysis.remainder[e], slots.G[(e - 1) * dim + d]) for d in 1:dim]
    any(bd -> depends(bd, state_slots(slots)), b) && throw(
        ArgumentError(
            "SUPG: the advection velocity ∂R/∂∇u of equation $e is $b, which depends on the state; pass SUPG(τ; b = ...)"
        )
    )
    return b
end

"""
    weak_form(analysis; test = TestSlots(analysis.dim), test_space = Galerkin())

Projection of a strong-form analysis: equation `e` contributes `mass[e]·V` (kind `:mass`), the integrated-by-parts
volume integrand `−G·W + R·V`, and `Φ·V` on each natural condition's piece. With [`SUPG`](@ref), the volume
integrand gains `τ (b·W)(strong[e] − mass[e])` and the mass term `mass[e]·τ (b·W)`.
"""
function weak_form(analysis; test = TestSlots(analysis.dim), test_space::AbstractTestSpace = Galerkin())
    slots, V, W = analysis.slots, test.V, test.W
    volume = VolumeTerm[]
    facet = FacetTerm[]
    for e in eachindex(analysis.flux)
        m = analysis.mass[e]
        weight = test_weight(test_space, analysis, e, test)
        if !isequal(m, 0)
            mass = weight === nothing ? m * V : m * V + weight[1] * m * weight[2]
            push!(volume, VolumeTerm(e, :mass, mass, flags(mass, slots)...))
        end
        G = analysis.flux[e]
        integrand = -sum(G[d] * W[d] for d in eachindex(G)) + analysis.remainder[e] * V
        for c in classify(integrand, slots, test)
            push!(volume, VolumeTerm(e, c.kind, c.integrand, c.timedep, c.paramdep))
        end
        weight === nothing && continue
        τ, bW = weight
        for c in classify(bW * (analysis.strong[e] - m), slots, test)
            stabilised = τ * c.integrand
            push!(volume, VolumeTerm(e, c.kind, stabilised, flags(stabilised, slots)...))
        end
    end
    for bc in analysis.natural
        for c in classify(bc.Φ * V, slots, test)
            push!(facet, FacetTerm(bc.eq, bc.piece, c.kind, c.integrand, c.timedep, c.paramdep))
        end
    end
    return WeakForm(analysis, test, volume, facet)
end
