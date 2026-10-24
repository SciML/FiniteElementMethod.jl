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

The right-hand side `F(p, t)` of `M du + K u = F` with the Dirichlet rows `F[d] = g_d(t)`
already written in. `timedep` is `true` when the volume load or the Dirichlet data is a
function, in which case the load is re-evaluated on every call; otherwise it is evaluated once.
"""
struct FEMLoad
    load::Any
    prescribed::Vector{Int}
    values::Any
    timedep::Bool
    buffer::Vector{Float64}
    gbuffer::Vector{Float64}
end
Base.show(io::IO, ::FEMLoad) = print(io, "FEMLoad")

function FEMLoad(load, prescribed, values, n)
    timedep = !(load isa AbstractVector) || !(values isa Union{Number, AbstractVector})
    F = FEMLoad(load, prescribed, values, timedep, zeros(n), zeros(length(prescribed)))
    timedep || fill_load!(F, nothing, 0.0)
    return F
end

function fill_load!(F::FEMLoad, p, t)
    if F.load isa AbstractVector
        copyto!(F.buffer, F.load)
    else
        F.load(F.buffer, p, t)
    end
    g = F.values
    if g isa Union{Number, AbstractVector}
        F.buffer[F.prescribed] .= g
    else
        g(F.gbuffer, t)
        F.buffer[F.prescribed] .= F.gbuffer
    end
    return F.buffer
end

"""
    evaluate_load!(F::FEMLoad, p, t)

Return the load vector at `(p, t)`, writing into `F`'s buffer when the load is time dependent.
"""
evaluate_load!(F::FEMLoad, p, t) = F.timedep ? fill_load!(F, p, t) : F.buffer

fem_apply(op::FEMOp, rows::UnitRange{Int}, cols::UnitRange{Int}, x::AbstractVector) = view(op.A, rows, cols) * x
@register_array_symbolic fem_apply(op::FEMOp, rows::UnitRange{Int}, cols::UnitRange{Int}, x::AbstractVector) begin
    size = (length(rows),)
    eltype = promote_type(Float64, eltype(x))
    ndims = 1
end

fem_load(op::FEMLoad, rows::UnitRange{Int}, t) = evaluate_load!(op, nothing, t)[rows]
@register_array_symbolic fem_load(op::FEMLoad, rows::UnitRange{Int}, t) begin
    size = (length(rows),)
    eltype = Float64
    ndims = 1
end

"""
    FEMOperators(M, K, load, fields; prescribed = Int[], dirichlet_values = 0.0)

Assembled operators of the semi-discrete system `M du + K u = F(p, t)`, with Dirichlet
conditions applied by algebraic row replacement: for each prescribed dof `d`, row `d` of `M`
is zeroed, row `d` of `K` becomes `e_d` and `F[d] = g_d(t)`. `M` and `K` are copied, not
mutated, and must share one sparsity pattern (for example from `Ferrite.allocate_matrix(dh, ch)`)
that contains the diagonal entry of every prescribed dof.

  - `load`: a vector, or a function `(out, p, t) -> out` writing the full load vector.
  - `fields`: `name => range` pairs in state order; the ranges must tile `1:size(M, 1)`.
  - `dirichlet_values`: a number, a vector over `prescribed`, or `(out, t) -> out`.

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
end

function FEMOperators(
        M::SparseMatrixCSC, K::SparseMatrixCSC, load, fields;
        prescribed = Int[], dirichlet_values = 0.0
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
    return FEMOperators(M, K, F, fields, prescribed, differential_vars, copy(K))
end

# One pass over the stored entries; the Jacobian then needs no row zeroing.
function replace_dirichlet_rows!(M, K, prescribed)
    isprescribed = falses(size(K, 1))
    isprescribed[prescribed] .= true
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
non-tunable parameters `Mop`, `Kop`, `Fop`, and one block equation per field,
`F_i(p, t) ~ Σ_j M_ij D(u_j) + K_ij u_j`. `unknowns` defaults to `name(iv)[1:n]` per field.
Returns `(sys, unknowns, parameter_map)`.
"""
function fem_system(ops::FEMOperators, iv; unknowns = nothing, metadata = nothing, name = :fem)
    D = Differential(iv)
    us = unknowns === nothing ?
        [only(@variables $field(iv)[1:length(r)]) for (field, r) in ops.fields] : collect(unknowns)
    @parameters Mop::FEMOp [tunable = false] Kop::FEMOp [tunable = false] Fop::FEMLoad [tunable = false]
    ranges = last.(ops.fields)
    eqs = [
        fem_load(Fop, ri, iv) ~ sum(
            fem_apply(Mop, ri, rj, D(uj)) + fem_apply(Kop, ri, rj, uj)
                for (rj, uj) in zip(ranges, us)
        )
            for ri in ranges
    ]
    md = metadata === nothing ? [] : [ProblemTypeCtx => metadata]
    sys = complete(System(eqs, iv, us, [Mop, Kop, Fop]; name, metadata = md))
    for (u, r) in zip(us, ranges)
        variable_index(sys, u) == r || error("System reordered the unknowns: $u is at $(variable_index(sys, u)), expected $r")
    end
    pmap = Dict{Any, Any}(Mop => FEMOp(ops.M), Kop => FEMOp(ops.K), Fop => ops.load)
    return sys, us, pmap
end

"""
    consistent_du0(ops::FEMOperators, u0, p, t0)

`du0` with `M_dd du_d = (F(p, t0) - K u0)_d` on the differential rows and zero elsewhere.
"""
function consistent_du0(ops::FEMOperators, u0, p, t0)
    dd = ops.differential_vars
    du0 = zeros(promote_type(eltype(u0), eltype(ops.M)), length(u0))
    any(dd) || return du0
    rhs = evaluate_load!(ops.load, p, t0) - ops.K * u0
    du0[dd] = ops.M[dd, dd] \ rhs[dd]
    return du0
end

default_initializealg() = BrownFullBasicInit(nlsolve = NewtonRaphson(autodiff = AutoForwardDiff()))

"""
    numeric_dae_problem(ops::FEMOperators, u0, tspan, metadata = nothing;
        iv = ModelingToolkit.t_nounits, unknowns = nothing, du0 = nothing,
        initializealg = BrownFullBasicInit(nlsolve = NewtonRaphson(autodiff = AutoForwardDiff())),
        coordinate = nothing, name = :fem, kwargs...)

Build the `DAEProblem` for `M du + K u - F(p, t) = 0` with a numeric residual, an analytic
sparse Jacobian `K + γ M` and the completed `System` of [`fem_system`](@ref) attached for
symbolic indexing. No initialization problem is built and nothing calls `mtkcompile`.

`metadata` becomes the problem's `problem_type`; a `SciMLBase.AbstractDiscretizationMetadata`
first gets the `System` through `PDEBase.add_metadata!`. `du0` defaults to the consistent one of
[`consistent_du0`](@ref). `coordinate(dof)` is used in the error for a non-finite `u0`.
Remaining keywords are passed to `DAEProblem`.
"""
function numeric_dae_problem(
        ops::FEMOperators, u0, tspan, metadata = nothing;
        iv = ModelingToolkit.t_nounits, unknowns = nothing, du0 = nothing,
        initializealg = default_initializealg(), coordinate = nothing, name = :fem, kwargs...
    )
    length(u0) == size(ops.K, 1) || throw(DimensionMismatch("u0 has length $(length(u0)), the operators have $(size(ops.K, 1)) rows"))
    check_finite_initial_condition(u0, ops.fields; coordinate)
    sys, _, pmap = fem_system(ops, iv; unknowns, metadata, name)
    p = MTKParameters(sys, pmap)
    getM, getK, getF = getp(sys, sys.Mop), getp(sys, sys.Kop), getp(sys, sys.Fop)
    function residual!(res, du, u, p, t)
        mul!(res, getM(p).A, du)
        mul!(res, getK(p).A, u, true, true)
        res .-= evaluate_load!(getF(p), p, t)
        return nothing
    end
    # J shares the operators' pattern, and the Dirichlet rows of K + γM are already e_d.
    function jac!(J, du, u, p, γ, t)
        nonzeros(J) .= nonzeros(getK(p).A) .+ γ .* nonzeros(getM(p).A)
        return nothing
    end
    u0 = collect(u0)
    du0 = du0 === nothing ? consistent_du0(ops, u0, p, first(tspan)) : collect(du0)
    f = DAEFunction{true}(residual!; jac = jac!, jac_prototype = ops.jac_prototype, sys)
    md = getmetadata(sys, ProblemTypeCtx, nothing)
    md isa SciMLBase.AbstractDiscretizationMetadata && PDEBase.add_metadata!(md, sys)
    differential_vars = copy(ops.differential_vars)
    return md === nothing ?
        DAEProblem(f, du0, u0, tspan, p; differential_vars, initializealg, kwargs...) :
        DAEProblem(f, du0, u0, tspan, p, md; differential_vars, initializealg, kwargs...)
end
