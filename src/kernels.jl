# Stage f of the strong-to-weak transform (spec §5.1, §5.2): quadrature-point kernels from a WeakForm.

kernel_slots(wf::WeakForm) = kernel_slots(wf.analysis.slots, wf.test)
kernel_slots(s, test) = (s.U, s.G, s.Ud, test.V, test.W, s.X, s.T, s.N, s.P)

compile(ex, slots) = Symbolics.build_function(ex, slots...; expression = Val(false), expression_module = @__MODULE__)

"The first unevaluated `Differential` call in `ex`, or `nothing`."
function find_differential(ex)
    SU.iscall(ex) || return nothing
    SU.operation(ex) isa Symbolics.Differential && return ex
    for arg in SU.arguments(ex)
        found = find_differential(arg)
        found === nothing || return found
    end
    return nothing
end

"The innermost function call in `ex` whose derivative with respect to `slot` stays unevaluated, or `nothing`."
function underivable_function(ex, slot)
    SU.iscall(ex) || return nothing
    for arg in SU.arguments(ex)
        found = underivable_function(arg, slot)
        found === nothing || return found
    end
    op = SU.operation(ex)
    (op in (+, -, *, /, ^) || op isa Symbolics.Differential) && return nothing
    return find_differential(Symbolics.unwrap(Symbolics.derivative(Symbolics.wrap(ex), slot))) === nothing ? nothing : op
end

# `Symbolics.derivative` leaves `Differential(x)(f(x))` unevaluated for a registered `f` without a
# derivative rule, and `build_function` compiles that to 0.
function assert_differentiated(ex, term = ex)
    found = find_differential(Symbolics.unwrap(ex))
    found === nothing && return ex
    inner = only(SU.arguments(found))
    slot = SU.operation(found).x
    fname = underivable_function(inner, slot)
    what = fname === nothing ? "cannot differentiate $inner with respect to $slot: register the derivative of the functions it calls" :
        "cannot differentiate $fname with respect to $slot: register the derivative for $fname"
    throw(ArgumentError("$what (U = field value, G = field gradient, field-major; term: $term)"))
end
checked_derivative(ex, x) = assert_differentiated(Symbolics.derivative(ex, x), ex)

"""
    StructuralZero()

An entry of a `coeffs` tuple that is zero for every argument. It is known from the tuple's type, so the driver's
contractions drop it at compile time; arithmetic with numbers treats it as an exact zero.
"""
struct StructuralZero end
Base.:+(::StructuralZero, x::Number) = x
Base.:+(x::Number, ::StructuralZero) = x
Base.:+(::StructuralZero, ::StructuralZero) = StructuralZero()
Base.:-(x::Number, ::StructuralZero) = x
Base.:*(::StructuralZero, ::Number) = StructuralZero()
Base.:*(::Number, ::StructuralZero) = StructuralZero()
Base.:*(::StructuralZero, ::StructuralZero) = StructuralZero()

# A tuple-valued function of `slots`, allocation-free with `Vec` and tuple arguments. Constant entries are
# compiled as `Float64`, constant zeros as `zero`.
function compile_tuple(exs, slots; zero = 0.0)
    entries = map(exs) do ex
        x = Symbolics.unwrap(ex)
        SU.isconst(x) || return x
        value = SU.unwrap_const(x)
        return iszero(value) ? zero : Float64(value)
    end
    return Symbolics.build_function(SU.Code.MakeTuple(entries), slots...; expression = Val(false), expression_module = @__MODULE__)
end

compile_tangent(ex, s, slots) = compile_tuple([checked_derivative(ex, x) for x in [s.U; s.G]], slots)

"""
    compile_coeffs(kind, ex, slots, test)

The coefficient form of a `:mass`, `:bilinear` or `:source` integrand `ex` (see `KernelSpec`), or `nothing`, in
which case the kernel keeps the per-shape-function `f`. It is `nothing` when
- `kind` is `:nonlinear`: per quadrature point the coefficients cost more than calling `f` and `tangent` per test
  function (measured on case (iii), 1.43× against 1.20× of hand-written);
- `ex` is not the sum of its coefficients times the trial and test slots, e.g. a term with a part free of the
  trial slots or of the test slots, or one nonlinear in them;
- a coefficient depends on a trial or test slot, i.e. on `U`, `G`, `Ud`, `V` or `W`.
Terms built by `weak_form` always have a coefficient form; the fallback serves hand-built integrands.
"""
function compile_coeffs(kind, ex, s, test)
    kind === :nonlinear && return nothing
    w = [test.V; test.W]
    trial = kind === :mass ? s.Ud : [s.U; s.G]
    first_order = [checked_derivative(ex, wi) for wi in w]
    if kind === :source
        exs = first_order
        linear = sum(first_order .* w)
    else
        exs = [checked_derivative(d, si) for si in trial for d in first_order]
        linear = sum(exs .* [si * wi for si in trial for wi in w])
    end
    isequal(Symbolics.expand(ex - linear), 0) || return nothing
    any(e -> depends(e, [s.U; s.G; s.Ud; w]), exs) && return nothing
    return compile_tuple(exs, (s.X, s.T, s.N, s.P); zero = StructuralZero())
end

# `f` is compiled only when the driver will call it: without `coeffs`, or for `:nonlinear` terms.
function kernel(eq, kind, piece, integrand, wf::WeakForm; coeffs = true)
    slots = kernel_slots(wf)
    s = wf.analysis.slots
    assert_differentiated(integrand)
    c = coeffs ? compile_coeffs(kind, integrand, s, wf.test) : nothing
    tangent = kind === :nonlinear ? compile_tangent(integrand, s, slots) : nothing
    f = c === nothing ? compile(integrand, slots) : nothing
    return KernelSpec(eq, kind, piece, f, tangent, c)
end

"The terms of `wf` summed per `(eq, kind, piece)`, volume first, in order of first appearance."
function fused_terms(wf::WeakForm)
    keyed = [[(term.eq, term.kind, nothing) => term.integrand for term in wf.volume]; [(term.eq, term.kind, term.piece) => term.integrand for term in wf.facet]]
    keys = Tuple{Int, Symbol, Any}[]
    sums = Dict{Tuple{Int, Symbol, Any}, Num}()
    for (key, integrand) in keyed
        haskey(sums, key) || push!(keys, key)
        sums[key] = get(sums, key, Num(0)) + integrand
    end
    return [(key..., sums[key]) for key in keys]
end

"""
    emit_kernels(wf::WeakForm; coeffs = true)

One `KernelSpec` per equation, kind and piece of `wf` (the terms sharing them are summed), as a tuple. Linear
kinds carry `coeffs`, the coefficient form the driver contracts with the shape functions, and then no `f`
(see [`compile_coeffs`](@ref) for the exceptions). `:nonlinear` kernels carry `f(U, G, Ud, V, W, X, T, N, P)`,
the scalar integrand, and `tangent`, with the same arguments, returning the tuple `∂I/∂[U; G]` (length
`nf + nf*dim`). `coeffs = false` emits `f` for every kernel instead. Array slots accept `Vec`, tuples or vectors.
"""
function emit_kernels(wf::WeakForm; coeffs = true)
    return Tuple(kernel(eq, kind, piece, integrand, wf; coeffs) for (eq, kind, piece, integrand) in fused_terms(wf))
end

"""
    emit_dirichlet(analysis)

One `(piece, field, g)` per essential condition of `analysis`, with `g(X, T, P)` the prescribed value.
"""
function emit_dirichlet(analysis)
    s = analysis.slots
    return Tuple((bc.piece, bc.field, compile(assert_differentiated(bc.g), (s.X, s.T, s.P))) for bc in analysis.essential)
end
