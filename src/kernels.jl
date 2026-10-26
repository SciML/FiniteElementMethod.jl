# Stage f of the strong-to-weak transform (spec §5.1, §5.2): quadrature-point kernels from a WeakForm.

kernel_slots(wf::WeakForm) = kernel_slots(wf.analysis.slots, wf.test)
kernel_slots(s, test) = (s.U, s.G, s.Ud, test.V, test.W, s.X, s.T, s.N, s.P)

compile(ex, slots) = Symbolics.build_function(ex, slots...; expression = Val(false))

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

# One scalar function per component, returned as a tuple: the array form of `build_function` allocates
# and cannot build its output from `Vec` or tuple arguments.
function compile_tangent(ex, s, slots)
    parts = Tuple(compile(checked_derivative(ex, x), slots) for x in [s.U; s.G])
    # Explicit arguments: a pass-through `args...` is not specialised and allocates.
    return (U, G, Ud, V, W, X, T, N, P) -> map(part -> part(U, G, Ud, V, W, X, T, N, P), parts)
end

function kernel(term, piece, wf::WeakForm)
    slots = kernel_slots(wf)
    tangent = term.kind === :nonlinear ? compile_tangent(term.integrand, wf.analysis.slots, slots) : nothing
    f = compile(assert_differentiated(term.integrand), slots)
    return KernelSpec(term.eq, term.kind, piece, f, tangent)
end

"""
    emit_kernels(wf::WeakForm)

One `KernelSpec` per volume and facet term of `wf`, as a tuple. Each `f(U, G, Ud, V, W, X, T, N, P)`
returns the scalar integrand; a `:nonlinear` term's `tangent` takes the same arguments and returns the
tuple `∂I/∂[U; G]` (length `nf + nf*dim`). Array slots accept `Vec`, tuples or vectors.
"""
function emit_kernels(wf::WeakForm)
    return (
        map(term -> kernel(term, nothing, wf), wf.volume)...,
        map(term -> kernel(term, term.piece, wf), wf.facet)...,
    )
end

"""
    emit_dirichlet(analysis)

One `(piece, field, g)` per essential condition of `analysis`, with `g(X, T, P)` the prescribed value.
"""
function emit_dirichlet(analysis)
    s = analysis.slots
    return Tuple((bc.piece, bc.field, compile(assert_differentiated(bc.g), (s.X, s.T, s.P))) for bc in analysis.essential)
end
