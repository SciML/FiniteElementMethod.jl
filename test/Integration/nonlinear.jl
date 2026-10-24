# End to end, nonlinear single-field classes (spec §10.1 case (iii); Appendix A nonlinear diffusion, nonlinear
# advection, nonlinear natural conditions): the chain's residual K u − F + r(u) against a hand-written residual at
# a random state, its Jacobian K + J against central differences, Newton convergence order, and nodal rates of a
# manufactured solution.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y
@variables u(..)
Dx, Dy = Differential(x), Differential(y)
U = u(t, x, y)
const RIGHT = [1.0, 0.0]
const P0 = [0.0]

# Steady `0 ~ ∇·(k(u)∇u) + f` for the non-harmonic `u* = shift + sin(πx)cos(πy) + x²y` (red cell: Dauntless's
# `eˣ sin y` has Δu* = 0), Dirichlet `u*` on left, bottom and top, and `∂ₙu = ∂ₓu*` on right, written as the
# user would (not the conormal), so the chain rescales it into a nonlinear facet kernel.
function diffusion_problem(ksym, knum; shift = 0.0)
    ustar = shift + sin(pi * x) * cos(pi * y) + x^2 * y
    kstar = ksym(ustar)
    fsym = expand_derivatives(-(Dx(kstar * Dx(ustar)) + Dy(kstar * Dy(ustar))))
    hsym = expand_derivatives(Dx(ustar))
    eq = 0 ~ Dx(ksym(U) * Dx(U)) + Dy(ksym(U) * Dy(U)) + fsym
    bcs = [BC("left", U ~ ustar), BC("bottom", U ~ ustar), BC("top", U ~ ustar), BC("right", Dx(U) ~ hsym, RIGHT)]
    c = chain([eq], bcs; depvars = [U], t, sivs = [x, y])
    f, hf, exact = numeric(fsym, [x, y]), numeric(hsym, [x, y]), numeric(ustar, [x, y])
    reference(setup, uvec) = residual_reference(
        setup, uvec; flux = (e, Uq, ∇Uq, X) -> knum(Uq[1]) * ∇Uq[1], source = (e, Uq, ∇Uq, X) -> -f(X),
        facets = [("right", 1, (Uq, X) -> -knum(Uq[1]) * hf(X))]
    )
    return (; c, reference, exact, shift)
end

# Residual and tangent at a random state, then Newton and nodal errors on `sizes`.
function check_diffusion(prob, sizes)
    (; c, reference, exact, shift) = prob
    @test any(v -> v.kind === :nonlinear, c.wf.volume)
    @test any(f -> f.kind === :nonlinear && f.piece == "right", c.wf.facet)
    setup = mesh_setup(Triangle, 1, 16)
    u_rand = shift .+ randn(Xoshiro(1), ndofs(setup.dh))
    R(v) = chain_residual(setup, c.kernels, v, P0)[1]
    rg, Jg = chain_residual(setup, c.kernels, u_rand, P0)
    @test relerr(rg, reference(setup, u_rand)) ≤ 1.0e-14
    @test tangent_error(R, Jg, u_rand) ≤ 1.0e-9
    errs = Float64[]
    for nel in sizes
        setup = mesh_setup(Triangle, 1, nel)
        ch = constraints(setup, c.dirichlet, P0)
        uh, history = newton(v -> chain_residual(setup, c.kernels, v, P0), ch, fill(shift, ndofs(setup.dh)))
        @test newton_converged(history)
        push!(errs, nodal_error(setup, uh, exact))
    end
    return errs
end

@testset "nonlinear diffusion k(u) = 1 + u²" begin
    prob = diffusion_problem(v -> 1 + v^2, v -> 1 + v^2)
    errs = check_diffusion(prob, (8, 16, 32, 64))
    @test rates(errs)[end] ≥ 1.9                       # red cell: 1.71, 1.83, 1.94
end

@testset "nonlinear diffusion k(u) = u², solution bounded away from zero" begin
    prob = diffusion_problem(v -> v^2, v -> v^2; shift = 3.0)
    errs = check_diffusion(prob, (8, 16, 32))
    @test rates(errs)[end] ≥ 1.8
end

@testset "nonlinear natural condition ∂ₙu = −(u⁴ − 1)" begin
    c = chain(
        [0 ~ Dx(Dx(U)) + Dy(Dy(U)) + 1], [BC("left", U ~ 1.0), BC("right", Dx(U) ~ -(U^4 - 1), RIGHT)];
        depvars = [U], t, sivs = [x, y]
    )
    @test Set(f.kind for f in c.wf.facet) == Set([:nonlinear, :source])
    setup = mesh_setup(Triangle, 1, 12)
    ref(v) = residual_reference(
        setup, v; flux = (e, Uq, ∇Uq, X) -> ∇Uq[1], source = (e, Uq, ∇Uq, X) -> -1.0,
        facets = [("right", 1, (Uq, X) -> Uq[1]^4 - 1)]
    )
    u_rand = 1 .+ 0.3 .* randn(Xoshiro(2), ndofs(setup.dh))
    rg, Jg = chain_residual(setup, c.kernels, u_rand, P0)
    @test relerr(rg, ref(u_rand)) ≤ 1.0e-14
    @test tangent_error(v -> chain_residual(setup, c.kernels, v, P0)[1], Jg, u_rand) ≤ 1.0e-9
    ch = constraints(setup, c.dirichlet, P0)
    uh, history = newton(v -> chain_residual(setup, c.kernels, v, P0), ch, ones(ndofs(setup.dh)))
    @test newton_converged(history)
    @test norm(ref(uh)[setdiff(1:ndofs(setup.dh), ch.prescribed_dofs)]) < 1.0e-10
end

@testset "nonlinear advection −u·∂ₓu and −∂ₓ(u²), 1D" begin
    @variables v1(..)
    V1 = v1(t, x)
    setup = mesh_setup(Line, 2, 10)
    for (advection, coefficient) in ((-V1 * Dx(V1), 1.0), (-Dx(V1^2), 2.0))
        c = chain([Differential(t)(V1) ~ Dx(Dx(V1)) + advection], [BC("left", V1 ~ 1.0)]; depvars = [V1], t, sivs = [x])
        @test any(v -> v.kind === :nonlinear, c.wf.volume)
        u_rand = randn(Xoshiro(4), ndofs(setup.dh))
        rg, Jg = chain_residual(setup, c.kernels, u_rand, P0)
        ref = residual_reference(
            setup, u_rand; flux = (e, Uq, ∇Uq, X) -> ∇Uq[1], source = (e, Uq, ∇Uq, X) -> coefficient * Uq[1] * ∇Uq[1][1]
        )
        @test relerr(rg, ref) ≤ 1.0e-14
        @test tangent_error(v -> chain_residual(setup, c.kernels, v, P0)[1], Jg, u_rand) ≤ 1.0e-9
    end
end
