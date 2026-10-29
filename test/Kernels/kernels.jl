# Stage f (spec §5.1, §5.2): kernels evaluate their symbolic integrands, tangents match central
# differences, every KernelSpec is concrete, and calls with the driver's slot types do not allocate.
using FiniteElementMethod: weak_form, emit_kernels, emit_dirichlet, WeakForm, VolumeTerm, FacetTerm, TestSlots,
    StructuralZero
using Ferrite: Vec
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
