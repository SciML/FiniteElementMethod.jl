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

_components(ex) = Symbolics.scalarize(Symbolics.wrap(ex))

function _apply_vector_op(op::Symbolics.Grad, s)
    return SymbolicUtils.Const{SymbolicUtils.SymReal}(unwrap.([Differential(x)(s) for x in op.vars]))
end

function _apply_vector_op(op::Symbolics.Divergence, v)
    comps = _components(v)
    return sum(i -> unwrap(Differential(op.vars[i])(comps[i])), eachindex(op.vars))
end

function _apply_vector_op(op::Symbolics.Curl{false, N}, v) where {N}
    comps = _components(v)
    D = Differential.(op.vars)
    N == 2 && return unwrap(D[1](comps[2]) - D[2](comps[1]))
    return SymbolicUtils.Const{SymbolicUtils.SymReal}(
        unwrap.([D[2](comps[3]) - D[3](comps[2]), D[3](comps[1]) - D[1](comps[3]), D[1](comps[2]) - D[2](comps[1])])
    )
end

function _lower_vector_op(ex)
    ex = _unwrap_arg(ex)
    _has_vector_op(ex) || return ex
    op = SymbolicUtils.operation(ex)
    args = [_lower_vector_op(a) for a in SymbolicUtils.arguments(ex)]
    op isa _LAZY_VECTOR_OPS && return _apply_vector_op(op, only(args))
    return SymbolicUtils.maketerm(typeof(ex), op, args, nothing)
end

_is_array_valued(ex) = !isempty(SymbolicUtils.shape(unwrap(ex)))

# The scalar equations that `eq` stands for: itself, or one per component when its sides are vectors (a 3D curl
# equation; Symbolics requires both sides to have the same shape).
function _scalar_equations(eq::Equation)
    (_has_vector_op(eq.lhs) || _has_vector_op(eq.rhs)) || return [eq]
    lhs, rhs = _lower_vector_op(eq.lhs), _lower_vector_op(eq.rhs)
    _is_array_valued(lhs) || return [Symbolics.wrap(lhs) ~ Symbolics.wrap(rhs)]
    return Equation[a ~ b for (a, b) in zip(_components(lhs), _components(rhs))]
end

_flat_equations(eq::Equation) = [eq]
_flat_equations(eqs) = reduce(vcat, map(_flat_equations, eqs); init = Equation[])

"""
The equations of `eqs` with the lazy vector operators rewritten to `Differential`s, vector equations split into
components. Anything else is returned as it is.
"""
lower_vector_ops(eqs) = reduce(vcat, map(_scalar_equations, _flat_equations(eqs)); init = Equation[])

# The `pdesys` itself when no equation or condition uses a lazy operator, so a system in the `Differential` form
# reaches PDEBase untouched.
function lower_vector_ops(pdesys::ModelingToolkit.PDESystem)
    eqs = _flat_equations([pdesys.eqs; pdesys.bcs])
    ops = reduce(vcat, [_vector_ops(e.lhs) for e in eqs]; init = []) ∪ reduce(vcat, [_vector_ops(e.rhs) for e in eqs]; init = [])
    isempty(ops) && return pdesys
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
