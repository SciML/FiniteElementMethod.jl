# The vector-calculus operators of Symbolics (`Grad`, `Divergence`, `Curl`, `Laplacian`, written `grad = Grad(vars)`,
# `grad ⋅ u`, `grad × u`) rewritten to the `Differential` form that PDEBase and the kernels classify. The eager forms
# (`Grad{true}` and friends) are `Differential`s already; only the lazy ones, which build a term whose operation is the
# operator itself, are rewritten here. The product rule is not applied, so `∇⋅(ν∇u)` becomes `Dx(ν Dx(u)) + Dy(ν Dy(u))`,
# the divergence form the flux analysis reads.

const _LAZY_VECTOR_OPS = Union{Symbolics.Grad{false}, Symbolics.Divergence{false}, Symbolics.Curl{false}}

function _vector_ops(ex)
    ops = []
    _walk(e -> SymbolicUtils.iscall(e) && SymbolicUtils.operation(e) isa _LAZY_VECTOR_OPS && push!(ops, SymbolicUtils.operation(e)), ex)
    return ops
end

_has_vector_op(ex) = !isempty(_vector_ops(ex))

# The components of an array-valued expression in column-major order, as a 1-based array. `scalarize` indexes from 1,
# so an array over other axes (a field `A[0:1]`) is indexed by its own axes instead.
function _components(ex)
    axes_ = SymbolicUtils.shape(unwrap(ex))
    wrapped = Symbolics.wrap(ex)
    all(r -> first(r) == 1, axes_) || return reshape([wrapped[i...] for i in Iterators.product(axes_...)], length.(axes_)...)
    comps = Symbolics.scalarize(wrapped)
    return reshape(collect(Iterators.flatten(comps)), size(comps))
end

function _apply_vector_op(op::Symbolics.Grad, s)
    return SymbolicUtils.Const{SymbolicUtils.vartype(unwrap(s))}(unwrap.([Differential(x)(s) for x in op.vars]))
end

function _apply_vector_op(op::Symbolics.Divergence, v)
    comps = _components(v)
    return sum(i -> unwrap(Differential(op.vars[i])(comps[i])), eachindex(op.vars))
end

function _apply_vector_op(op::Symbolics.Curl{false, N}, v) where {N}
    comps = _components(v)
    D = Differential.(op.vars)
    N == 2 && return unwrap(D[1](comps[2]) - D[2](comps[1]))
    return SymbolicUtils.Const{SymbolicUtils.vartype(first(op.vars))}(
        unwrap.([D[2](comps[3]) - D[3](comps[2]), D[3](comps[1]) - D[1](comps[3]), D[1](comps[2]) - D[2](comps[1])])
    )
end

# Innermost first: a lazy term whose argument has no lazy operator is replaced, and the pass repeats on the result.
# `substitute` skips the arguments of a `Differential` unless told otherwise, and an operator can sit inside one
# (`Dx(∇(u)[1])`), so the filter admits every term; a pass that changes nothing is an error rather than a loop.
function _lower_vector_op(ex)
    ex = _unwrap_arg(ex)
    while true
        inner = nothing
        _walk(ex) do e
            inner === nothing && SymbolicUtils.iscall(e) && SymbolicUtils.operation(e) isa _LAZY_VECTOR_OPS &&
                !_has_vector_op(only(SymbolicUtils.arguments(e))) && (inner = e)
        end
        inner === nothing && return ex
        lowered = _apply_vector_op(SymbolicUtils.operation(inner), only(SymbolicUtils.arguments(inner)))
        next = _unwrap_arg(substitute(ex, Dict(inner => lowered); filterer = Returns(true)))
        isequal(next, ex) && throw(ArgumentError("cannot rewrite the vector operator $inner in $ex to Differentials"))
        ex = next
    end
    return
end

_is_array_valued(ex) = !isempty(SymbolicUtils.shape(unwrap(ex)))

# The first array term of `ex` other than the base of an index (`A[1]` of an array variable), which the analysis cannot read.
function _array_term(ex)
    ex = _unwrap_arg(ex)
    SymbolicUtils.iscall(ex) || return nothing
    args = SymbolicUtils.arguments(ex)
    SymbolicUtils.operation(ex) === getindex && (args = args[2:end])
    isempty(SymbolicUtils.shape(ex)) || return ex
    for a in args
        term = _array_term(a)
        term === nothing || return term
    end
    return nothing
end

# A scalar expression with reductions over the lowered component arrays (`sum(∇u)`, `∇u ⋅ ∇u`) expanded to scalar arithmetic,
# which is all the strong-form analysis reads. `scalarize` is left out of expressions with no array term, since it indexes
# array variables from 1 and fails on a field `A[0:1]`.
function _scalarized(ex)
    ex = _unwrap_arg(ex)
    # `scalarize` does not look inside a `Differential`, so each one gets its argument scalarized first.
    derivatives = Any[]
    _walk(e -> SymbolicUtils.iscall(e) && SymbolicUtils.operation(e) isa Differential && push!(derivatives, e), ex)
    rules = Dict(d => SymbolicUtils.operation(d)(_scalarized(only(SymbolicUtils.arguments(d)))) for d in derivatives)
    isempty(rules) || (ex = _unwrap_arg(substitute(ex, rules; filterer = Returns(true))))
    _array_term(ex) === nothing && return ex
    ex = _unwrap_arg(Symbolics.scalarize(Symbolics.wrap(ex)))
    term = _array_term(ex)
    term === nothing || throw(ArgumentError("cannot rewrite to scalar Differentials: the array term $term remains in $ex"))
    return ex
end

# The scalar equations that `eq` stands for: itself, or one per component when its sides are vectors (a 3D curl
# equation, written with a lazy or an eager operator). The sides must have the same number of components.
function _scalar_equations(eq::Equation)
    lazy = _has_vector_op(eq.lhs) || _has_vector_op(eq.rhs)
    lazy || _is_array_valued(eq.lhs) || _is_array_valued(eq.rhs) || return [eq]
    lhs, rhs = _lower_vector_op(eq.lhs), _lower_vector_op(eq.rhs)
    if _is_array_valued(lhs) || _is_array_valued(rhs)
        (_is_array_valued(lhs) && _is_array_valued(rhs) && size(_components(lhs)) == size(_components(rhs))) ||
            throw(ArgumentError("equation $eq has sides with different shapes, so it cannot be split into scalar equations"))
        pairs = zip(vec(_components(lhs)), vec(_components(rhs)))
    else
        pairs = [(lhs, rhs)]
    end
    return Equation[Symbolics.wrap(_scalarized(a)) ~ Symbolics.wrap(_scalarized(b)) for (a, b) in pairs]
end

_flat_equations(eq::Equation) = [eq]
_flat_equations(eqs) = reduce(vcat, map(_flat_equations, eqs); init = Equation[])

# The equations of `eqs` with the lazy vector operators rewritten to `Differential`s, vector equations split into
# components. Anything else is returned as it is.
lower_vector_ops(eqs) = reduce(vcat, map(_scalar_equations, _flat_equations(eqs)); init = Equation[])

# The `pdesys` itself when no equation or condition uses a lazy operator, so a system in the `Differential` form
# reaches PDEBase untouched.
function lower_vector_ops(pdesys::ModelingToolkit.PDESystem)
    eqs = _flat_equations([pdesys.eqs; pdesys.bcs])
    flat = all(e -> e isa Equation, pdesys.eqs) && all(e -> e isa Equation, pdesys.bcs)
    ops = reduce(vcat, [_vector_ops(e.lhs) for e in eqs]; init = []) ∪ reduce(vcat, [_vector_ops(e.rhs) for e in eqs]; init = [])
    isempty(ops) && flat && !any(e -> _is_array_valued(e.lhs) || _is_array_valued(e.rhs), eqs) && return pdesys
    # Symbolics differentiates by any variable without complaint, which would leave a stray `Differential` for PDEBase.
    for op in ops, x in op.vars
        any(iv -> isequal(unwrap(iv), x), pdesys.ivs) ||
            throw(ArgumentError("the operator $op is taken in $(Num.(collect(op.vars))), but $(Num(x)) is not an independent variable of the PDESystem, whose are $(Num.(pdesys.ivs))"))
    end
    return ModelingToolkit.PDESystem(
        lower_vector_ops(pdesys.eqs), lower_vector_ops(pdesys.bcs), pdesys.domain, pdesys.ivs, pdesys.dvs, pdesys.ps;
        initial_conditions = pdesys.initial_conditions, name = nameof(pdesys)
    )
end

# An equation key of `ibp` or `pairing`, lowered like the equations it names; other keys are left as they are. An
# integer key indexes the flattened input equations, so it names every component of a vector equation: `spans[i]` is
# the range of lowered equations that input equation `i` became.
_lowered_keys(key::Equation, spans) = lower_vector_ops([key])
function _lowered_keys(key::Integer, spans)
    spans === nothing && return [key]
    (key isa Bool || !(key in eachindex(spans))) && throw(ArgumentError("the integer key $key is not the index of an equation of the system, which has $(length(spans))"))
    return collect(spans[key])
end
_lowered_keys(key, spans) = [key]

function _lowering_spans(eqs)
    stop = 0
    return map(_flat_equations(eqs)) do eq
        start = stop + 1
        stop += length(_scalar_equations(eq))
        start:stop
    end
end

# The options of `ibp` or `pairing` with their equation keys lowered. Two keys that lower to one equation, a lazy
# equation and its expansion, must carry the same value (`same`), or one of them would win by dictionary order.
function _lowered_options(options, same, name, spans = nothing)
    lowered = Dict{Any, Any}()
    origin = Dict{Any, Any}()
    for (key, value) in options, k in _lowered_keys(key, spans)
        if haskey(lowered, k)
            same(lowered[k], value) || throw(
                ArgumentError("the $name keys $(origin[k]) and $key are the same equation once the vector operators are rewritten, but their values $(lowered[k]) and $value differ")
            )
        else
            lowered[k], origin[k] = value, key
        end
    end
    return lowered
end

# Two integration-by-parts policies are the same when they are of one type and, for `Weighted`, have equal weights.
_same_policy(a, b) = typeof(a) == typeof(b) && (!(a isa Weighted) || isequal(a.w, b.w))
