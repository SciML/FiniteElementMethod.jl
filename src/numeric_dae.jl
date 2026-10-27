"""
    FEMOp(A::SparseMatrixCSC)

Opaque wrapper for an assembled operator, used as a non-tunable `System` parameter so that
ModelingToolkit neither flattens it into the tunable vector nor scans it element by element.
"""
struct FEMOp
    A::SparseMatrixCSC{Float64, Int}
end
Base.show(io::IO, ::FEMOp) = print(io, "FEMOp")

"""
    FEMLoad

The right-hand side `F(t)` of `M du + K u = F` with the Dirichlet rows `F[d] = g_d(t)`.
`timedep` is `true` when the volume load or the Dirichlet data is a function. It holds no
buffers: [`load!`](@ref) writes into the caller's vector, so problems that share a `FEMLoad`
(remakes, ensembles) never share mutable state through it.
"""
struct FEMLoad
    load::Any
    prescribed::Vector{Int}
    values::Any
    timedep::Bool
    n::Int
end
Base.show(io::IO, ::FEMLoad) = print(io, "FEMLoad")

function FEMLoad(load, prescribed, values, n)
    load isa AbstractVector && length(load) != n &&
        throw(DimensionMismatch("the load vector has length $(length(load)), the operators have $n rows"))
    values isa AbstractVector && length(values) != length(prescribed) &&
        throw(DimensionMismatch("dirichlet_values has length $(length(values)) for $(length(prescribed)) prescribed dofs"))
    timedep = !(load isa AbstractVector) || !(values isa Union{Number, AbstractVector})
    return FEMLoad(load, prescribed, values, timedep, n)
end

"""
    load!(out, F::FEMLoad, t)

Write the load vector at `t` into `out`, with `g(t)` on the prescribed rows, and return `out`.
"""
function load!(out, F::FEMLoad, t)
    if F.load isa AbstractVector
        copyto!(out, F.load)
    else
        F.load(out, t)
    end
    dirichlet_values!(view(out, F.prescribed), F, t)
    return out
end

function dirichlet_values!(out, F::FEMLoad, t)
    g = F.values
    if g isa Union{Number, AbstractVector}
        out .= g
    else
        g(out, t)
    end
    return out
end

"""
    dirichlet_rates(F::FEMLoad, t)

`g'(t)` on the prescribed dofs: zero for constant data, `ForwardDiff` through the callback otherwise.
"""
function dirichlet_rates(F::FEMLoad, t)
    F.values isa Union{Number, AbstractVector} && return zeros(length(F.prescribed))
    return ForwardDiff.derivative(τ -> dirichlet_values!(zeros(typeof(τ), length(F.prescribed)), F, τ), float(t))
end

fem_apply(op::FEMOp, rows::UnitRange{Int}, cols::UnitRange{Int}, x::AbstractVector) = view(op.A, rows, cols) * x
@register_array_symbolic fem_apply(op::FEMOp, rows::UnitRange{Int}, cols::UnitRange{Int}, x::AbstractVector) begin
    size = (length(rows),)
    eltype = promote_type(Float64, eltype(x))
    ndims = 1
end

fem_load(op::FEMLoad, rows::UnitRange{Int}, t) = load!(zeros(op.n), op, t)[rows]
@register_array_symbolic fem_load(op::FEMLoad, rows::UnitRange{Int}, t) begin
    size = (length(rows),)
    eltype = Float64
    ndims = 1
end

"""
    FEMNonlinear(op, p)

Opaque wrapper for the nonlinear part `N(u, t)` of `M du + K u + N(u, t) = F(t)`, used as a
non-tunable `System` parameter like [`FEMOp`](@ref): `op` is a [`NonlinearOperator`](@ref)
and `p` the parameter values its kernels are evaluated with. [`FEMOperators`](@ref) marks the
prescribed dofs, whose rows of `N` and of its Jacobian are then zero. `op` is untyped so that
the parameter has one concrete type; each call dispatches on it once. The operator's buffers
are shared by every problem holding this wrapper (remakes, ensembles), so those problems must
not be solved concurrently.
"""
struct FEMNonlinear
    op::Any
    p::Vector{Float64}
    prescribed::BitVector
end
FEMNonlinear(op, p) = FEMNonlinear(op, collect(Float64, p), falses(0))
Base.show(io::IO, ::FEMNonlinear) = print(io, "FEMNonlinear")

function nonlinear_residual!(r, N::FEMNonlinear, u, t)
    residual!(r, N.op, u, N.p, t)
    r[N.prescribed] .= 0
    return r
end

function nonlinear_jacobian!(N::FEMNonlinear, u, t)
    J = jacobian!(N.op.J, N.op, u, N.p, t)
    rows = rowvals(J)
    for k in eachindex(rows)
        N.prescribed[rows[k]] && (nonzeros(J)[k] = 0)
    end
    return J
end

# `fem_nonlinear(op, rows, t, x)` is rows `rows` of `N(x, t)`, with `x` the whole state built from the
# field arrays by `fem_cat`. A registered function gets a method for every symbolic/numeric combination
# of its arguments, so it takes the state as one array rather than one argument per field (which made
# about 150·5ⁿ⁻¹ methods for n fields); `fem_cat` keeps the fields whole arrays instead of the scalars
# that `vcat` of symbolic arrays gives.
fem_cat(a::AbstractVector, b::AbstractVector) = vcat(a, b)
@register_array_symbolic fem_cat(a::AbstractVector, b::AbstractVector) begin
    size = (length(a) + length(b),)
    eltype = promote_type(eltype(a), eltype(b))
    ndims = 1
end

fem_nonlinear(op::FEMNonlinear, rows::UnitRange{Int}, t, x::AbstractVector) =
    nonlinear_residual!(zeros(promote_type(Float64, eltype(x)), length(x)), op, x, t)[rows]
@register_array_symbolic fem_nonlinear(op::FEMNonlinear, rows::UnitRange{Int}, t, x::AbstractVector) begin
    size = (length(rows),)
    eltype = Float64
    ndims = 1
end

"""
    FEMOperators(M, K, load, fields; prescribed = Int[], dirichlet_values = 0.0, nonlinear = nothing)

Assembled operators of the semi-discrete system `M du + K u + N(u, t) = F(p, t)`, with Dirichlet
conditions applied by algebraic row replacement: for each prescribed dof `d`, row `d` of `M`
is zeroed, row `d` of `K` becomes `e_d` and `F[d] = g_d(t)`. `M` and `K` are copied, not
mutated, and must share one sparsity pattern (for example from `Ferrite.allocate_matrix(dh, ch)`)
that contains the diagonal entry of every prescribed dof.

  - `load`: a vector, or a function `(out, t) -> out` writing the full load vector. Loads do
    not see the parameters; parameter-dependent operators are reassembled instead (spec §8.3).
  - `fields`: `name => range` pairs in state order; the ranges must tile `1:size(M, 1)`.
  - `dirichlet_values`: a number, a vector over `prescribed`, or `(out, t) -> out`.
  - `nonlinear`: `nothing`, or a [`FEMNonlinear`](@ref) whose Jacobian has the sparsity
    pattern of `M` and `K`. Its rows at prescribed dofs are dropped, so the Dirichlet rows
    stay `u_d = g_d(t)`.

The fields `M`, `K`, `load`, `fields`, `prescribed`, `differential_vars` and `jac_prototype`
are read by [`numeric_dae_problem`](@ref). `differential_vars` is `false` on prescribed dofs
and on every row where `M` is zero, which covers algebraic equations such as `0 ~ Dxx(v) + f`.
"""
struct FEMOperators
    M::SparseMatrixCSC{Float64, Int}
    K::SparseMatrixCSC{Float64, Int}
    load::FEMLoad
    fields::Vector{Pair{Symbol, UnitRange{Int}}}
    prescribed::Vector{Int}
    differential_vars::BitVector
    jac_prototype::SparseMatrixCSC{Float64, Int}
    nonlinear::Union{Nothing, FEMNonlinear}
end

function FEMOperators(
        M::SparseMatrixCSC, K::SparseMatrixCSC, load, fields;
        prescribed = Int[], dirichlet_values = 0.0, nonlinear = nothing
    )
    n = size(K, 1)
    size(M) == size(K) == (n, n) || throw(DimensionMismatch("M and K must be square of the same size; got $(size(M)) and $(size(K))"))
    M.colptr == K.colptr && M.rowval == K.rowval ||
        throw(ArgumentError("M and K must share one sparsity pattern, e.g. both from Ferrite.allocate_matrix(dh, ch)"))
    fields = Pair{Symbol, UnitRange{Int}}[Symbol(name) => r for (name, r) in fields]
    stop = 0
    for (name, r) in fields
        first(r) == stop + 1 || throw(ArgumentError("field ranges must tile 1:$n in state order; field $name starts at $(first(r)), expected $(stop + 1)"))
        stop = last(r)
    end
    stop == n || throw(ArgumentError("field ranges cover 1:$stop but the operators have $n rows"))
    prescribed = collect(Int, prescribed)
    M, K = copy(SparseMatrixCSC{Float64, Int}(M)), copy(SparseMatrixCSC{Float64, Int}(K))
    replace_dirichlet_rows!(M, K, prescribed)
    differential_vars = falses(n)
    for (k, i) in enumerate(rowvals(M))
        iszero(nonzeros(M)[k]) || (differential_vars[i] = true)
    end
    F = FEMLoad(load, prescribed, dirichlet_values, n)
    if nonlinear !== nothing
        J = nonlinear.op.J
        J.colptr == K.colptr && J.rowval == K.rowval ||
            throw(ArgumentError("the nonlinear operator's Jacobian must have the sparsity pattern of M and K, e.g. NonlinearOperator(ctx, kernels; ch) with the ConstraintHandler of allocate_matrix(dh, ch)"))
        nonlinear = FEMNonlinear(nonlinear.op, nonlinear.p, isprescribed_mask(n, prescribed))
    end
    return FEMOperators(M, K, F, fields, prescribed, differential_vars, copy(K), nonlinear)
end

function isprescribed_mask(n, prescribed)
    mask = falses(n)
    mask[prescribed] .= true
    return mask
end

# One pass over the stored entries; the Jacobian then needs no row zeroing.
function replace_dirichlet_rows!(M, K, prescribed)
    isprescribed = isprescribed_mask(size(K, 1), prescribed)
    rows = rowvals(K)
    for j in axes(K, 2), k in nzrange(K, j)
        i = rows[k]
        isprescribed[i] || continue
        nonzeros(M)[k] = 0
        nonzeros(K)[k] = i == j
    end
    for d in prescribed
        isone(K[d, d]) || throw(ArgumentError("the sparsity pattern has no diagonal entry for prescribed dof $d"))
    end
    return M, K
end

"""
    check_finite_initial_condition(u0, fields; coordinate = nothing)

Throw an `ArgumentError` naming the first dof of `u0` that is not finite, its field and,
if `coordinate(dof)` is given, its node coordinate. A non-finite initial value (for example
`sin(r)/r` at `r = 0`) would otherwise give a `NaN` `du0` and an `Unstable` return at step 1.
"""
function check_finite_initial_condition(u0, fields; coordinate = nothing)
    for (d, val) in pairs(u0)
        isfinite(val) && continue
        name = first(fields[findfirst(f -> d in last(f), fields)])
        at = coordinate === nothing ? "" : " at coordinate $(coordinate(d))"
        throw(ArgumentError("the initial condition is not finite for dof $d of field $name$at: u0[$d] = $val"))
    end
    return nothing
end

"""
    fem_system(ops::FEMOperators, iv; unknowns = nothing, metadata = nothing, name = :fem)

The completed array-residual `System`: one array unknown per field, the operators as opaque
non-tunable parameters `Mop`, `Kop`, `Fop` (and `Nop` for `ops.nonlinear`), and one block
equation per field, `F_i(p, t) ~ Σ_j M_ij D(u_j) + K_ij u_j (+ N_i(u, t))`. `parameters` are `symbol => value` pairs
added as parameters with those values, which the operators were assembled with. `unknowns` defaults to `name(iv)[1:n]` per field.
Returns `(sys, unknowns, parameter_map)`.
"""
function fem_system(ops::FEMOperators, iv; unknowns = nothing, metadata = nothing, name = :fem, parameters = ())
    D = Differential(iv)
    us = unknowns === nothing ?
        [only(@variables $field(iv)[1:length(r)]) for (field, r) in ops.fields] : collect(unknowns)
    @parameters Mop::FEMOp [tunable = false] Kop::FEMOp [tunable = false] Fop::FEMLoad [tunable = false]
    ranges = last.(ops.fields)
    linear(ri) = sum(fem_apply(Mop, ri, rj, D(uj)) + fem_apply(Kop, ri, rj, uj) for (rj, uj) in zip(ranges, us))
    ps = Any[Mop, Kop, Fop]
    append!(ps, first.(parameters))
    if ops.nonlinear === nothing
        eqs = [fem_load(Fop, ri, iv) ~ linear(ri) for ri in ranges]
    else
        @parameters Nop::FEMNonlinear [tunable = false]
        push!(ps, Nop)
        state = foldl(fem_cat, us)
        eqs = [fem_load(Fop, ri, iv) ~ linear(ri) + fem_nonlinear(Nop, ri, iv, state) for ri in ranges]
    end
    md = metadata === nothing ? [] : [ProblemTypeCtx => metadata]
    sys = complete(System(eqs, iv, us, ps; name, metadata = md))
    for (u, r) in zip(us, ranges)
        variable_index(sys, u) == r || error("System reordered the unknowns: $u is at $(variable_index(sys, u)), expected $r")
    end
    pmap = Dict{Any, Any}(Mop => FEMOp(ops.M), Kop => FEMOp(ops.K), Fop => ops.load)
    ops.nonlinear === nothing || (pmap[last(ps)] = ops.nonlinear)
    foreach(((q, v),) -> pmap[q] = v, parameters)
    return sys, us, pmap
end

"""
    consistent_du0(ops::FEMOperators, u0, t0)

`du0` with `du_d = g'(t0)` on prescribed dofs, zero on the other algebraic rows, and
`M_ff du_f = (F(t0) - K u0 - N(u0, t0) - M_fd g'(t0))_f` on the differential rows. Throws an
`ArgumentError` when a non-prescribed algebraic variable has a nonzero column in `M`: its
`du` is then not zero, cannot be found without index reduction, and must be passed as `du0`.
"""
function consistent_du0(ops::FEMOperators, u0, t0)
    dd = ops.differential_vars
    isprescribed = falses(length(u0))
    isprescribed[ops.prescribed] .= true
    for j in axes(ops.M, 2)
        dd[j] || isprescribed[j] || all(iszero, view(nonzeros(ops.M), nzrange(ops.M, j))) ||
            throw(ArgumentError("variable $j is algebraic (its row of M is zero) but its column of M is not, so a consistent du0 cannot be computed; pass du0 explicitly"))
    end
    du0 = zeros(promote_type(eltype(u0), eltype(ops.M)), length(u0))
    du0[ops.prescribed] = dirichlet_rates(ops.load, t0)
    any(dd) || return du0
    rhs = load!(zeros(length(u0)), ops.load, t0) - ops.K * u0 - ops.M * du0
    ops.nonlinear === nothing || (rhs .-= nonlinear_residual!(zeros(length(u0)), ops.nonlinear, u0, t0))
    du0[dd] = ops.M[dd, dd] \ rhs[dd]
    return du0
end

function check_pattern(J, A, name)
    (A.colptr === J.colptr && A.rowval === J.rowval) || (A.colptr == J.colptr && A.rowval == J.rowval) ||
        throw(ArgumentError("$name no longer has the sparsity pattern of the Jacobian prototype; an operator with a new pattern (a remeshed problem) needs a new problem from numeric_dae_problem, not remake"))
    return nothing
end

# The operators hold the parameter values they were assembled with; a `remake` with other values would
# otherwise be solved silently with the old ones.
function assembled_parameter_check(sys, parameters)
    isempty(parameters) && return Returns(nothing)
    syms = [q for (q, _) in parameters]
    assembled = [v for (_, v) in parameters]
    get = getp(sys, syms)
    return function (p)
        current = get(p)
        for (q, a, c) in zip(syms, assembled, current)
            isequal(a, c) || throw(
                ArgumentError(
                    "parameter $q was $a when the operators were assembled but is $c in this problem; operators are not reassembled for new parameter values, so build a new problem with DAEProblem(pdesys, disc)"
                )
            )
        end
        return nothing
    end
end

default_initializealg() = BrownFullBasicInit(nlsolve = NewtonRaphson(autodiff = AutoForwardDiff()))

"""
    numeric_dae_problem(ops::FEMOperators, u0, tspan, metadata = nothing;
        iv = ModelingToolkit.t_nounits, unknowns = nothing, du0 = nothing,
        initializealg = BrownFullBasicInit(nlsolve = NewtonRaphson(autodiff = AutoForwardDiff())),
        coordinate = nothing, name = :fem, kwargs...)

Build the `DAEProblem` for `M du + K u + N(u, t) - F(p, t) = 0` with a numeric residual, an
analytic sparse Jacobian `K + γ M + ∂N/∂u` and the completed `System` of [`fem_system`](@ref) attached for
symbolic indexing. No initialization problem is built and nothing calls `mtkcompile`.

`parameters` (`symbol => value` pairs) become parameters of the `System` at the values the
operators were assembled with; the residual and Jacobian throw an `ArgumentError` naming the
parameter if a `remake` has changed one. `metadata` becomes the problem's `problem_type`; a `SciMLBase.AbstractDiscretizationMetadata`
first gets the `System` through `PDEBase.add_metadata!`. `du0` defaults to the consistent one of
[`consistent_du0`](@ref). `coordinate(dof)` is used in the error for a non-finite `u0`.
Remaining keywords are passed to `DAEProblem`.
"""
function numeric_dae_problem(
        ops::FEMOperators, u0, tspan, metadata = nothing;
        iv = ModelingToolkit.t_nounits, unknowns = nothing, du0 = nothing,
        initializealg = default_initializealg(), coordinate = nothing, name = :fem, parameters = (), kwargs...
    )
    length(u0) == size(ops.K, 1) || throw(DimensionMismatch("u0 has length $(length(u0)), the operators have $(size(ops.K, 1)) rows"))
    check_finite_initial_condition(u0, ops.fields; coordinate)
    sys, _, pmap = fem_system(ops, iv; unknowns, metadata, name, parameters)
    p = MTKParameters(sys, pmap)
    check_assembled = assembled_parameter_check(sys, parameters)
    getM, getK, getF = getp(sys, sys.Mop), getp(sys, sys.Kop), getp(sys, sys.Fop)
    getN = ops.nonlinear === nothing ? Returns(nothing) : getp(sys, sys.Nop)
    nbuf = zeros(length(u0))
    function residual!(res, du, u, p, t)
        check_assembled(p)
        load!(res, getF(p), t)
        res .= .-res
        mul!(res, getM(p).A, du, true, true)
        mul!(res, getK(p).A, u, true, true)
        N = getN(p)
        N === nothing || (res .+= nonlinear_residual!(nbuf, N, u, t))
        return nothing
    end
    # The Dirichlet rows of K + γM are already e_d and those of N's Jacobian zero, so the
    # stored values are added as they are.
    function jac!(J, du, u, p, γ, t)
        check_assembled(p)
        Kp, Mp = getK(p).A, getM(p).A
        check_pattern(J, Kp, "Kop")
        check_pattern(J, Mp, "Mop")
        nonzeros(J) .= nonzeros(Kp) .+ γ .* nonzeros(Mp)
        N = getN(p)
        if N !== nothing
            JN = nonlinear_jacobian!(N, u, t)
            check_pattern(J, JN, "Nop")
            nonzeros(J) .+= nonzeros(JN)
        end
        return nothing
    end
    u0 = collect(u0)
    du0 = du0 === nothing ? consistent_du0(ops, u0, first(tspan)) : collect(du0)
    f = DAEFunction{true}(residual!; jac = jac!, jac_prototype = ops.jac_prototype, sys)
    md = getmetadata(sys, ProblemTypeCtx, nothing)
    md isa SciMLBase.AbstractDiscretizationMetadata && PDEBase.add_metadata!(md, sys)
    differential_vars = copy(ops.differential_vars)
    return md === nothing ?
        DAEProblem(f, du0, u0, tspan, p; differential_vars, initializealg, kwargs...) :
        DAEProblem(f, du0, u0, tspan, p, md; differential_vars, initializealg, kwargs...)
end
