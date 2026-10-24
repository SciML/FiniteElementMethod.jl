# Stage f of the strong-to-weak transform (spec §5.1, §5.2): quadrature-point kernels from a WeakForm.

kernel_slots(wf::WeakForm) = kernel_slots(wf.analysis.slots, wf.test)
kernel_slots(s, test) = (s.U, s.G, s.Ud, test.V, test.W, s.X, s.T, s.N, s.P)

compile(ex, slots) = Symbolics.build_function(ex, slots...; expression = Val(false))

# One scalar function per component, returned as a tuple: the array form of `build_function` allocates
# and cannot build its output from `Vec` or tuple arguments.
function compile_tangent(ex, s, slots)
    parts = Tuple(compile(Symbolics.derivative(ex, x), slots) for x in [s.U; s.G])
    # Explicit arguments: a pass-through `args...` is not specialised and allocates.
    return (U, G, Ud, V, W, X, T, N, P) -> map(part -> part(U, G, Ud, V, W, X, T, N, P), parts)
end

function kernel(term, piece, wf::WeakForm)
    slots = kernel_slots(wf)
    tangent = term.kind === :nonlinear ? compile_tangent(term.integrand, wf.analysis.slots, slots) : nothing
    return KernelSpec(term.eq, term.kind, piece, compile(term.integrand, slots), tangent)
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
    return Tuple((bc.piece, bc.field, compile(bc.g, (s.X, s.T, s.P))) for bc in analysis.essential)
end
