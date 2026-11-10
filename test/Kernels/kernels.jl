# Stage f (spec §5.1, §5.2): kernels evaluate their symbolic integrands, tangents match central
# differences, every KernelSpec is concrete, and calls with the driver's slot types do not allocate.
import FiniteElementMethod
using FiniteElementMethod: weak_form, emit_kernels, emit_dirichlet, WeakForm, VolumeTerm, FacetTerm, TestSlots,
    StructuralZero, emit_strong, build
using FiniteElementMethod: AssemblyContext, assemble_linear
using Ferrite: Ferrite, Vec, Triangle, RefTriangle, Lagrange, DofHandler, CellValues, FacetValues, QuadratureRule,
    FacetQuadratureRule, generate_grid, add!, close!
using LinearAlgebra: norm
using Symbolics
using Test

include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse, BC, StateSlots

opaque(z) = 1 + z^2
@register_symbolic opaque(z)

@variables t x y k α β h Du Dw χ a b
@variables u(..) w(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U, Wf = u(t, x, y), w(t, x, y)
const RIGHT, TOP = [1.0, 0.0], [0.0, 1.0]

function case_i()
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    bcs = [BC("left", U ~ 0.1 * y), BC("right", Dx(U) ~ h, RIGHT), BC("top", k * Dy(U) + α * U ~ β, TOP)]
    return analyse([eq], bcs; depvars = [U], t, sivs = [x, y], params = [k, α, β, h])
end
function case_iii()
    ustar = exp(x) * sin(y) / 2
    kstar = 1 + ustar^2
    fsym = expand_derivatives(-(Dx(kstar * Dx(ustar)) + Dy(kstar * Dy(ustar))))
    eq = 0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + fsym
    bcs = [BC("left", U ~ ustar), BC("right", Dx(U) ~ expand_derivatives(Dx(ustar)), RIGHT)]
    return analyse([eq], bcs; depvars = [U], t, sivs = [x, y])
end
function case_iib()
    axy = 1 + x^2 + y
    eq = Dt(U) ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) - 0.7 * Dx(U) - 0.3 * Dy(U) - 2.5 * U + x * y + 1
    bcs = [BC("left", U ~ 0.1 * y), BC("right", Dx(U) ~ h, RIGHT), BC("top", axy * Dy(U) + α * U ~ β, TOP)]
    return analyse([eq], bcs; depvars = [U], t, sivs = [x, y], params = [k, α, β, h])
end
function two_field()
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a - U + U^2 * Wf,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U^2 * Wf,
    ]
    bcs = [BC("right", Du * Dx(U) + χ * Dx(Wf) ~ 0.1, RIGHT), BC("right", Dx(Wf) ~ 0.2, RIGHT)]
    return analyse(eqs, bcs; depvars = [U, Wf], t, sivs = [x, y], params = [Du, Dw, χ, a, b])
end
const CASES = ("(i)" => case_i, "(iii)" => case_iii, "two-field" => two_field)

randtuple(n) = ntuple(_ -> 0.5 + rand(), n)
randvec(n) = Vec{n}(randtuple(n))
"Random kernel arguments with the assembly driver's slot types: NTuple U, G, Ud; Vec W, X, N; Vector P."
function random_args(an)
    nf, dim, s = an.nfields, an.dim, an.slots
    return (
        randtuple(nf), randtuple(nf * dim), randtuple(nf), rand(), randvec(dim), randvec(dim), rand(), randvec(dim),
        0.5 .+ rand(length(s.P)),
    )
end
function symbolic_value(ex, wf, args)
    s = wf.analysis.slots
    vars = (s.U, s.G, s.Ud, [wf.test.V], wf.test.W, s.X, [s.T], s.N, s.P)
    vals = Dict(v => val for (vs, as) in zip(vars, args) for (v, val) in zip(vs, as))
    return Float64(Symbolics.value(substitute(ex, vals; fold = Val(true))))
end
setslot(args, i, j, δ) = Base.setindex(args, Base.setindex(args[i], args[i][j] + δ, j), i)
# One typed barrier for every arity: the call is specialised on `F` and the argument types, so `@allocated`
# sees no boxing of the arguments (needed on Julia 1.10).
function measure_alloc(f::F, args::Vararg{Any, N}) where {F, N}
    f(args...)
    return @allocated f(args...)
end
allocations(f, args) = measure_alloc(f, args...)

@testset "kernels of case $name" for (name, case) in CASES
    wf = weak_form(case())
    kernels = emit_kernels(wf; coeffs = false)
    @test kernels isa Tuple && isconcretetype(typeof(kernels))
    @test length(kernels) == length(wf.volume) + length(wf.facet)
    @test all(kernel -> isconcretetype(typeof(kernel)), kernels)
    terms = [wf.volume; wf.facet]
    @test [kernel.piece for kernel in kernels] == [fill(nothing, length(wf.volume)); [term.piece for term in wf.facet]]
    for (kernel, term) in zip(kernels, terms)
        args = random_args(wf.analysis)
        @test kernel.eq == term.eq && kernel.kind === term.kind
        @test isapprox(kernel.f(args...), symbolic_value(term.integrand, wf, args); rtol = 1.0e-12)
        @test allocations(kernel.f, args) == 0
        @test (kernel.tangent === nothing) == (kernel.kind !== :nonlinear)
        kernel.tangent === nothing && continue
        # central differences over the U then G slots, in that order
        ε = 1.0e-6
        fd = [
            (kernel.f(setslot(args, i, j, ε)...) - kernel.f(setslot(args, i, j, -ε)...)) / 2ε
                for i in (1, 2) for j in eachindex(args[i])
        ]
        tangent = kernel.tangent(args...)
        @test tangent isa Tuple && length(tangent) == length(fd)
        @test sqrt(sum(abs2, collect(tangent) .- fd)) <= 1.0e-8 * sqrt(sum(abs2, fd))
        @test allocations(kernel.tangent, args) == 0
    end
end

@testset "kernels accept plain vectors for array slots" begin
    wf = weak_form(two_field())
    args = random_args(wf.analysis)
    for kernel in emit_kernels(wf; coeffs = false)
        vecargs = map(a -> a isa Number ? a : collect(a), args)
        @test kernel.f(vecargs...) ≈ kernel.f(args...)
        kernel.tangent === nothing || @test collect(kernel.tangent(vecargs...)) ≈ collect(kernel.tangent(args...))
    end
end

@testset "Dirichlet data" begin
    an = case_i()
    dirichlet = emit_dirichlet(an)
    @test length(dirichlet) == 1
    piece, field, g = only(dirichlet)
    @test piece == "left" && field == 1
    X = Vec(0.3, 0.7)
    @test g(X, 0.0, [2.0, 3.0, 0.5, 1.5]) ≈ 0.1 * X[2]
    @test allocations(g, (X, 0.0, [2.0, 3.0, 0.5, 1.5])) == 0
    X1 = an.slots.X[1]
    @test_throws "register the derivative for opaque" emit_dirichlet((; an.slots, essential = [(; piece = "left", field = 1, g = Differential(X1)(opaque(X1)))]))
end

@testset "a function without a derivative rule is an error" begin
    # Built directly: the seam rejects this input before it reaches emit_kernels.
    s = StateSlots(1, 2, 1)
    test = TestSlots(2)
    term = VolumeTerm(1, :nonlinear, -s.G[1] * test.W[1] * opaque(s.U[1]), false, false)
    wf = WeakForm((; slots = s), test, [term], FacetTerm[])
    @test_throws "register the derivative for opaque" emit_kernels(wf)
    # An unevaluated Differential already in the integrand would also compile to 0.
    stray = VolumeTerm(1, :bilinear, Differential(s.U[1])(opaque(s.U[1])) * test.V, false, false)
    @test_throws "register the derivative for opaque" emit_kernels(WeakForm((; slots = s), test, [stray], FacetTerm[]))
    compound = VolumeTerm(1, :bilinear, Differential(s.U[1])(s.U[1] * opaque(s.U[1])) * test.V, false, false)
    @test_throws "register the derivative for opaque" emit_kernels(WeakForm((; slots = s), test, [compound], FacetTerm[]))
end

# The coefficient form (KernelSpec docstring): with w = [V; W] and trial slots s ([U; G] for :bilinear, Ud for
# :mass, none for :source), I = Σ c[(s - 1) * (1 + dim) + w] · s · w.
function from_coeffs(kernel, args)
    U, G, Ud, V, W, X, T, N, P = args
    c = kernel.coeffs(X, T, N, P)
    w = (V, Tuple(W)...)
    trial = kernel.kind === :source ? (1.0,) : kernel.kind === :mass ? Ud : (U..., G...)
    return sum(number(c[(s - 1) * length(w) + j]) * trial[s] * w[j] for s in eachindex(trial) for j in eachindex(w))
end
number(x) = x isa StructuralZero ? 0.0 : x

@testset "coefficient form of case $name" for (name, case) in (CASES..., "(ii-b)" => case_iib)
    wf = weak_form(case())
    kernels = emit_kernels(wf)
    # every linear kernel of these cases is contracted, so a change that sends one to the `f` path fails here
    @test all(kernels) do kernel
        kernel.kind === :nonlinear ? (kernel.coeffs === nothing && kernel.f !== nothing) :
            (kernel.coeffs !== nothing && kernel.f === nothing)
    end
    plain = emit_kernels(wf; coeffs = false)
    for (kernel, reference) in zip(kernels, plain)
        kernel.coeffs === nothing && continue
        args = random_args(wf.analysis)
        X, T, N, P = args[6:9]
        @test isapprox(from_coeffs(kernel, args), reference.f(args...); rtol = 1.0e-12)
        @test (@inferred kernel.coeffs(X, T, N, P)) isa Tuple
        @test measure_alloc(kernel.coeffs, X, T, N, P) == 0
    end
end

# A long raw source (the residual of a manufactured solution) is linear in V, but `expand(ex - linear)` does not
# cancel its quotients term by term, so it used to fall back to `f` once per test function and quadrature point.
@testset "a long raw symbolic source takes the coefficient path" begin
    uex = exp(-t) * sum(1:4) do j
        sin((1 + 0.3j) * x + t) * cos((1 + 0.2j) * y) * exp(-0.05j * x * y) * (1 + x^2 * y)^(0.5 + 0.1j) / j
    end
    fsym = expand_derivatives(Dt(uex) - Dx(Dx(uex)) - Dy(Dy(uex)))
    an = analyse([Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + fsym], [BC("left", U ~ 0)]; depvars = [U], t, sivs = [x, y])
    wf = weak_form(an)
    kernels = emit_kernels(wf)
    plain = emit_kernels(wf; coeffs = false)
    source = only(filter(kernel -> kernel.kind === :source, kernels))
    @test source.coeffs !== nothing

    grid = generate_grid(Triangle, (4, 4))
    ip = Lagrange{RefTriangle, 2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ctx = AssemblyContext(dh, [:u], CellValues(QuadratureRule{RefTriangle}(6), ip), FacetValues(FacetQuadratureRule{RefTriangle}(6), ip))
    F = assemble_linear(ctx, kernels, Float64[]; t = 0.3).F
    Fref = assemble_linear(ctx, plain, Float64[]; t = 0.3).F
    @test norm(F) > 0
    @test norm(F - Fref) ≤ 1.0e-12 * norm(Fref)
end

@testset "sources that are not linear in the test slots keep the per-test-function path" begin
    wf = weak_form(case_i())
    s, test = wf.analysis.slots, wf.test
    V, W = test.V, test.W[1]
    X = s.X[1]
    # `sign` has a zero derivative and vanishes at 0 but is not linear
    for ex in (V^2, V * W, V * X + 1, V * s.U[1], X + V * X, V + sign(V) - substitute(sign(V), V => 0))
        @test FiniteElementMethod.compile_coeffs(:source, ex, s, test) === nothing
    end
    @test FiniteElementMethod.compile_coeffs(:source, V * (X + 1)^2 / (1 + X^3)^1.5, s, test) !== nothing
end

@testset "one kernel per equation, kind and piece" begin
    # two natural conditions of equation 1 on `right` are summed into one facet kernel per kind
    an = analyse(
        [Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U))], [BC("right", Dx(U) ~ h, RIGHT), BC("right", Dx(U) + α * U ~ β, RIGHT)];
        depvars = [U], t, sivs = [x, y], params = [k, α, β, h]
    )
    wf = weak_form(an)
    kernels = emit_kernels(wf)
    keys = [(kernel.eq, kernel.kind, kernel.piece) for kernel in kernels]
    @test allunique(keys)
    @test Set(keys) == Set([(term.eq, term.kind, nothing) for term in wf.volume]) ∪ Set([(term.eq, term.kind, term.piece) for term in wf.facet])
    @test length(wf.facet) > length(filter(kernel -> kernel.piece !== nothing, kernels))
end

@testset "generated functions are RuntimeGeneratedFunctions owned by FiniteElementMethod" begin
    # The cache tag is the second type parameter of a RuntimeGeneratedFunction.
    owner(g) = parentmodule(typeof(g).parameters[2])
    functions = [g for k in emit_kernels(weak_form(two_field())) for g in (k.f, k.tangent, k.coeffs) if g !== nothing]
    @test !isempty(functions)
    @test all(g -> owner(g) === FiniteElementMethod, functions)
    _, _, g = only(emit_dirichlet(case_i()))
    @test owner(g) === FiniteElementMethod
end

@testset "expression form of case $name round-trips" for (name, case) in CASES
    wf = weak_form(case())
    compiled, generated = emit_kernels(wf), emit_kernels(wf; expression = Val(true))
    @test length(generated) == length(compiled)
    for (k, e) in zip(compiled, generated)
        @test (e.eq, e.kind, e.piece) == (k.eq, k.kind, k.piece)
        args = random_args(wf.analysis)
        for (g, ex, call_args) in ((k.f, e.f, args), (k.tangent, e.tangent, args), (k.coeffs, e.coeffs, args[6:9]))
            @test (g === nothing) == (ex === nothing)
            g === nothing && continue
            @test ex isa Expr
            evaluated = Core.eval(@__MODULE__, ex)
            @test isequal(Base.invokelatest(evaluated, call_args...), g(call_args...))
        end
    end
end

@testset "collocation: the strong residual of case (iii) vanishes at the exact solution" begin
    # spec §10.1 collocation row; scratch/dauntless/test_testspace.jl (a)
    ustar = exp(x) * sin(y) / 2
    r = only(emit_strong(case_iii()))
    D = (Dx, Dy)
    fields = [ustar; [expand_derivatives(D[d](ustar)) for d in 1:2]; [expand_derivatives(D[e](D[d](ustar))) for d in 1:2 for e in 1:2]]
    values = [Symbolics.build_function(f, x, y; expression = Val(false)) for f in fields]
    worst = maximum(1:200) do _
        X = (rand(), rand())
        v = [g(X...) for g in values]
        abs(r((v[1],), (v[2], v[3]), (0.0,), Tuple(v[4:7]), X, 0.0, [0.0]))
    end
    @test worst ≤ 1.0e-13
end

# The name of the function a generated call applies: Symbolics embeds function objects and global references.
function callee_name(x)
    x isa Expr && x.head === :call || return nothing
    callee = x.args[1]
    return callee isa Function ? nameof(callee) : callee isa GlobalRef ? callee.name : callee
end
# Calls of `^` or `NaNMath.pow` in a generated `Expr` whose exponent is a temporary bound to a number, which is what
# common-subexpression elimination makes of a literal exponent (`literal_pow` becomes a libm `pow` call).
function hoisted_exponents(ex)
    numbers = Set{Symbol}()
    found = Any[]
    walk(x) = x
    function walk(x::Expr)
        x.head === :(=) && x.args[1] isa Symbol && x.args[2] isa Number && push!(numbers, x.args[1])
        powers = callee_name(x) in (:^, :pow)
        powers && length(x.args) == 3 && x.args[3] isa Symbol && push!(found, x)
        foreach(walk, x.args)
        return x
    end
    walk(ex)
    return [call for call in found if call.args[3] in numbers]
end
count_calls(ex, name) = ex isa Expr ? (callee_name(ex) === name) + sum(a -> count_calls(a, name), ex.args; init = 0) : 0

@testset "no kernel turns a literal power into a pow call on a hoisted exponent" begin
    for (name, case) in ("(iii)" => case_iii, "two-field" => two_field)
        kernels = emit_kernels(weak_form(case()); expression = Val(true))
        @test any(k -> k.kind === :nonlinear, kernels)
        for k in kernels, ex in (k.f, k.tangent, k.coeffs)
            ex === nothing && continue
            @test isempty(hoisted_exponents(ex))
        end
        # not vacuous: the squares are there, as `literal_pow` calls
        powers = sum(count_calls(ex, :literal_pow) for k in kernels if k.kind === :nonlinear for ex in (k.f, k.tangent) if ex !== nothing)
        @test powers > 0
    end
    @variables ξ η
    squares = ξ^2 * η + 2 * ξ * η + ξ^2
    for cse in (false, true)
        code = build(squares, (ξ, η), Val(true); cse)
        @test isempty(hoisted_exponents(code))
        @test count_calls(code, :literal_pow) + count_calls(code, :^) > 0
    end
end

@testset "common-subexpression elimination of a source agrees with the plain code generator" begin
    @variables ξ η τ p1 p2
    slots = (ξ, η, τ, p1, p2)
    plain(ex) = Symbolics.build_function(ex, slots...; expression = Val(false), cse = false, expression_module = FiniteElementMethod)
    manufactured = exp(-τ) * sum(1:8) do j
        sin((1 + 0.3j) * ξ + τ) * cos((1 + 0.2j) * η) * exp(-0.05j * ξ * η) * (1 + ξ^2 * η)^(0.5 + 0.1j)
    end
    long = expand_derivatives(Differential(τ)(manufactured) - Differential(ξ)(Differential(ξ)(manufactured)) - Differential(η)(Differential(η)(manufactured)))
    zoo = (
        "ordinary" => sin(π * ξ) * sin(π * η) * exp(-τ) + 1,
        "long" => long,
        "parameters and time" => p1 * (ξ * η + τ)^2 + p2 * sin(ξ * τ) / (1 + η^2) - (ξ * η + τ)^2,
        "squares" => (1 - ξ)^2 * sin(τ)^2 + η^2 * cos(ξ * η)^2 + (1 - η)^2 * p1^2 * exp(-ξ^2),
    )
    for (name, ex) in zoo
        cse, reference = build(ex, slots, Val(false); cse = true), plain(ex)
        for _ in 1:20
            point = (0.5 + rand(), 0.5 + rand(), rand(), 0.5 + rand(), 0.5 + rand())
            @test isapprox(cse(point...), reference(point...); rtol = 1.0e-12)
        end
    end
    # it does share: the long source evaluates fewer sin calls
    shared, plain_code = build(long, slots, Val(true); cse = true), Symbolics.build_function(long, slots...; expression = Val{true}, cse = false, expression_module = FiniteElementMethod)
    @test count_calls(shared, :sin) < count_calls(plain_code, :sin)
    # through the emitters: a long source gives the same value as the symbolic integrand
    an = analyse([Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + sum(j -> sin(j * x + t) * cos(j * y)^2 * exp(-j * x * y) * (1 + x^2 * y)^(0.5 + 0.1j), 1:4)], BC[]; depvars = [U], t, sivs = [x, y], params = [k, α, β, h])
    wf = weak_form(an)
    args = random_args(an)
    for kernel in emit_kernels(wf; coeffs = false)
        kernel.kind === :source || continue
        term = only(filter(term -> term.eq == kernel.eq && term.kind === :source, wf.volume))
        @test isapprox(kernel.f(args...), symbolic_value(term.integrand, wf, args); rtol = 1.0e-12)
    end
end

guarded_root(z) = sqrt(z)
@register_symbolic guarded_root(z)

@testset "a guarded source keeps its conditional branches lazy" begin
    guarded = ifelse(x > 0, guarded_root(x), 0.0) + 2 * ifelse(x > 0, guarded_root(x), 0.0) * y
    an = analyse([Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + guarded], BC[]; depvars = [U], t, sivs = [x, y], params = [k, α, β, h])
    X = Vec(-1.0, 0.3)
    for kernel in emit_kernels(weak_form(an)), kind in (:source,)
        kernel.kind === kind || continue
        @test all(c -> c isa StructuralZero || iszero(c), kernel.coeffs(X, 0.0, Vec(1.0, 0.0), [2.0, 3.0, 0.5, 1.5]))
    end
    for kernel in emit_kernels(weak_form(an); coeffs = false)
        kernel.kind === :source || continue
        args = random_args(an)
        @test kernel.f(args[1:5]..., X, args[7:9]...) == 0.0
    end
end

module Custom
    pow(x, y) = x + y
end
@register_symbolic Custom.pow(x, y)

@testset "only NaNMath.pow is rewritten into literal_pow" begin
    @variables ξ
    ex = Custom.pow(ξ, 2) + ξ^2 * Custom.pow(ξ, 3)
    f = build(ex, (ξ,), Val(false); cse = true)
    @test f(3.0) == Custom.pow(3.0, 2) + 3.0^2 * Custom.pow(3.0, 3)
end
