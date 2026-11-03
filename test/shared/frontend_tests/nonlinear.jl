# Nonlinear terms through the front end (spec §8.1): PDESystem -> DAEProblem -> solve against manufactured
# solutions, the analytic Jacobian against finite differences, the Dirichlet rows, the remake policy and the
# nonlinear classes that stay rejected.
#
# Every manufactured solution is u* = (1 + x + 2y)(1 + t), which lies in the P1 space. Each source is written so
# that it cancels the nonlinear term pointwise at u*, so u* solves the discrete equations exactly whatever the
# quadrature, and the nodal error is at the level of the time-integration tolerance.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Triangle
using LinearAlgebra: I, norm
using SciMLBase: remake
using NonlinearSolveFirstOrder: NewtonRaphson

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@parameters k
@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const U = u(t, x, y)
const tol = 1.0e-8

uexact(t_, x_, y_) = (1 + x_ + 2y_) * (1 + t_)
const ustar = (1 + x + 2y) * (1 + t)
const ic = u(0, x, y) ~ 1 + x + 2y
square() = MeshDomain(rectangle_grid(Triangle, (4, 4)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
dirichlet(Ω, Γs = sides(Ω)) = [u(t, Γ) ~ ustar for Γ in Γs]

function pdesystem(eqs, bcs, Ω; ps = [])
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], ps)
    return pdesys
end

# Central differences of the DAE residual in u, plus γ times those in du.
function fd_jacobian(prob, du, u0, γ, t0; ε = 1.0e-6)
    n = length(u0)
    J = zeros(n, n)
    r₊, r₋ = zeros(n), zeros(n)
    for j in 1:n
        e = zeros(n)
        e[j] = ε
        prob.f(r₊, du, u0 + e, prob.p, t0)
        prob.f(r₋, du, u0 - e, prob.p, t0)
        J[:, j] .= (r₊ - r₋) / 2ε
        prob.f(r₊, du + γ * e, u0, prob.p, t0)
        prob.f(r₋, du - γ * e, u0, prob.p, t0)
        J[:, j] .+= (r₊ - r₋) / 2ε
    end
    return J
end

@testset verbose = true "nonlinear terms" begin
    @testset "nonlinear diffusion −∇·((1 + u²)∇u)" begin
        # ∇·((1 + u*²)∇u*) = 2u*|∇u*|² = 10u*(1 + t)²; the source is time dependent.
        Ω = square()
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + (1 + x + 2y) - 10 * ustar * (1 + t)^2
        sol, prob = fem_solve([eq], [dirichlet(Ω); ic], Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
        @test any(p -> nameof(p) === :Nop, ModelingToolkit.parameters(prob.f.sys))
    end

    @testset "nonlinear reaction −u²" begin
        Ω = square()
        eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) - U^2 + (1 + x + 2y) + ustar^2
        sol, _ = fem_solve([eq], [dirichlet(Ω); ic], Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
    end

    @testset "algebraic field fixed by a nonlinear term in itself (index 1) keeps its abstol" begin
        # 0 = v² − u − s with v* = 1 + x + 2y: v's linear block is zero, but v² determines v, so the system has
        # index 1 and algebraic_abstol = :auto must not put v out of error control.
        @variables v(..)
        V = v(t, x, y)
        Ω = square()
        parabolic = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + (1 + x + 2y)
        constraint = 0 ~ V^2 - U - ((1 + x + 2y)^2 - ustar)
        bcs = [dirichlet(Ω); ic; v(0, x, y) ~ 1 + x + 2y]
        sol, prob = fem_solve(
            [parabolic, constraint], bcs, Ω, (x, y), t, [U, V]; pairing = Dict(constraint => v),
            abstol = 1.0e-10, solve_kwargs = (; reltol = 1.0e-10)
        )
        @test !any(prob.differential_vars[FiniteElementMethod.field_range(sol.disc_data.space, :v)])
        @test prob.kwargs[:abstol] === 1.0e-10
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
        @test nodal_error(sol, V, (_, x_, y_) -> 1 + x_ + 2y_) <= tol
        @named pdesys = PDESystem([parabolic, constraint], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U, V], [])
        @test !haskey(discretize(pdesys, FEMDiscretization(Ω, t; pairing = Dict(constraint => v))).kwargs, :abstol)
    end

    @testset "Burgers-type convection u(∂ₓu + ∂ᵧu)" begin
        # u*(∂ₓu* + ∂ᵧu*) = 3u*(1 + t).
        Ω = square()
        eq = Dt(U) + U * Dx(U) + U * Dy(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + (1 + x + 2y) + 3 * ustar * (1 + t)
        sol, _ = fem_solve([eq], [dirichlet(Ω); ic], Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
    end

    @testset "nonlinear natural condition ∂ₙu = −u² + h" begin
        # On the right side x = 1, ∂ₙu* = ∂ₓu* = 1 + t, so h = (1 + t) + u*².
        Ω = square()
        Γl, Γr, Γb, Γt = sides(Ω)
        Dn = NormalDerivative(Γr)
        eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + (1 + x + 2y)
        bcs = [dirichlet(Ω, (Γl, Γb, Γt)); Dn(u(t, Γr)) ~ -u(t, Γr)^2 + (1 + t) + ustar^2; ic]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
    end

    @testset "Jacobian against finite differences, Dirichlet rows" begin
        Ω = square()
        eq = Dt(U) + U * Dx(U) ~ Dx((1 + U^2) * Dx(U)) + Dy(Dy(U)) - U^2 + (1 + x + 2y)
        prob = DAEProblem(pdesystem([eq], [dirichlet(Ω); ic], Ω), FEMDiscretization(Ω, t))
        n = length(prob.u0)
        rng_u = [0.3 + 0.1 * sin(3i) for i in 1:n]
        rng_du = [0.2 * cos(5i) for i in 1:n]
        γ, t0 = 0.7, 0.4
        J = copy(prob.f.jac_prototype)
        prob.f.jac(J, rng_du, rng_u, prob.p, γ, t0)
        Jfd = fd_jacobian(prob, rng_du, rng_u, γ, t0)
        @test norm(Matrix(J) - Jfd, Inf) <= 1.0e-7 * norm(Jfd, Inf)
        # Prescribed rows are u_d − g_d(t): identity rows of the Jacobian, unaffected by N(u).
        space = prob.problem_type.space
        dofs = space.ch.prescribed_dofs
        @test Matrix(J)[dofs, :] == Matrix(1.0 * I(n))[dofs, :]
        r = zeros(n)
        prob.f(r, rng_du, rng_u, prob.p, t0)
        g = FiniteElementMethod.dirichlet_values!(zeros(length(dofs)), space, t0)
        @test r[dofs] ≈ rng_u[dofs] - g atol = 1.0e-14
    end

    @testset "inconsistent Dirichlet u0 through the default initializer" begin
        # BrownFullBasicInit's NewtonRaphson differentiates the residual with ForwardDiff, so the load and N(u)
        # see Dual states; the time-dependent source exercises the load path too.
        Ω = square()
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + (1 + x + 2y) - 10 * ustar * (1 + t)^2
        prob = DAEProblem(pdesystem([eq], [dirichlet(Ω); ic], Ω), FEMDiscretization(Ω, t))
        dofs = prob.problem_type.space.ch.prescribed_dofs
        u0 = copy(prob.u0)
        u0[dofs] .+= 0.5
        sol = solve(remake(prob; u0), EXACT_ROWS; abstol = 1.0e-10, reltol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error(sol, U, uexact) <= tol
    end

    @testset "remakes solved concurrently match serial solves" begin
        # Every evaluation borrows its own copy of the nonlinear operator's buffers and values objects.
        Ω = MeshDomain(rectangle_grid(Triangle, (16, 16)))
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + (1 + x + 2y) - 10 * ustar * (1 + t)^2
        prob = DAEProblem(pdesystem([eq], [dirichlet(Ω); ic], Ω), FEMDiscretization(Ω, t))
        probs = [remake(prob; u0 = s * prob.u0) for s in (1.0, 1.3, 0.7)]
        final(q) = solve(q, EXACT_ROWS; abstol = 1.0e-8, reltol = 1.0e-8).original_sol.u[end]
        serial = final.(probs)
        @test serial[1] != serial[2]
        concurrent = fetch.([Threads.@spawn final(q) for q in probs])
        @test concurrent == serial
        Threads.nthreads() > 1 || @info "the remakes were solved on one thread, so this test cannot see a race; run with JULIA_NUM_THREADS > 1"
    end

    @testset "states other than Float64 and Dual{Float64} are rejected" begin
        Ω = square()
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy(Dy(U)) + (1 + x + 2y) - 2 * ustar * (1 + t)^2
        prob = DAEProblem(pdesystem([eq], [dirichlet(Ω); ic], Ω), FEMDiscretization(Ω, t))
        n = length(prob.u0)
        Dual = FiniteElementMethod.ForwardDiff.Dual
        states = (
            Float32.(prob.u0), big.(prob.u0), Dual{Nothing}.(Float32.(prob.u0), 1.0f0),
            Dual{:outer}.(Dual{:inner}.(prob.u0, 1.0), Dual{:inner}(1.0, 0.0)),
        )
        for u in states
            T = eltype(u)
            msg = "the nonlinear operator evaluates Float64 states and first-order ForwardDiff Duals of Float64; got $T"
            @test_throws ArgumentError(msg) prob.f(zeros(T, n), zeros(T, n), u, prob.p, 0.0)
        end
        @test_throws ArgumentError("the nonlinear operator evaluates Float64 states and first-order ForwardDiff Duals of Float64; got Float32") prob.f(
            zeros(Float32, n), zeros(Float32, n), Float32.(prob.u0), prob.p, 0.0
        )
        @test_throws ArgumentError("the nonlinear operator's Jacobian evaluates Float64 states; got Float32") prob.f.jac(
            copy(prob.f.jac_prototype), zeros(n), Float32.(prob.u0), prob.p, 1.0, 0.0
        )
        # The supported states still evaluate: Float64 and a first-order Dual of Float64.
        r = zeros(n)
        prob.f(r, zeros(n), prob.u0, prob.p, 0.0)
        rd = zeros(Dual{Nothing, Float64, 1}, n)
        prob.f(rd, zeros(n), Dual{Nothing}.(prob.u0, 1.0), prob.p, 0.0)
        @test FiniteElementMethod.ForwardDiff.value.(rd) ≈ r
    end

    @testset "steady nonlinear diffusion, dirichlet = $dirichlet" for dirichlet in (:algebraic, :eliminate)
        # ∇·((1 + u*²)∇u*) = 2u*|∇u*|² = 10u* for u* = 1 + x + 2y, which lies in the P1 space.
        Ω = square()
        us = 1 + x + 2y
        eq = Dx((1 + u(x, y)^2) * Dx(u(x, y))) + Dy((1 + u(x, y)^2) * Dy(u(x, y))) ~ 10 * us
        @named steady = PDESystem([eq], [u(Γ) ~ us for Γ in sides(Ω)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
        prob = discretize(steady, FEMDiscretization(Ω; dirichlet))
        sol = solve(prob, NewtonRaphson(); abstol = 1.0e-12)
        @test SciMLBase.successful_retcode(sol)
        @test sol.stats.nsteps > 1
        pts = range(0.05, 0.95; length = 7)
        @test maximum(abs(sol(px, py; dv = u(x, y)) - (1 + px + 2py)) for px in pts, py in pts) <= tol
    end

    @testset "dirichlet = :eliminate keeps the nonlinear terms" begin
        # u* = 1 + x + 2y is steady, so its Dirichlet data are constant in time; the sources cancel the diffusion
        # (10u*) and reaction (u*²) terms at u*, so u stays u* only if N is evaluated.
        Ω = square()
        us = 1 + x + 2y
        eq = Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) - U^2 + us^2 - 10 * us
        bcs = [[u(t, Γ) ~ us for Γ in sides(Ω)]; u(0, x, y) ~ us]
        exact(t_, x_, y_) = 1 + x_ + 2y_
        sole, probe = fem_solve([eq], bcs, Ω, (x, y), t, [U]; dirichlet = :eliminate)
        sola, _ = fem_solve([eq], bcs, Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sole)
        @test nodal_error(sole, U, exact) <= tol
        @test maximum(abs, sole[U][end, :] - sola[U][end, :]) <= tol
        @test any(p -> nameof(p) === :Nop, ModelingToolkit.parameters(probe.f.sys))
        # The analytic Jacobian on the free dofs against central differences of the reduced residual.
        n = length(probe.u0)
        rng_u = [1.0 + 0.1 * sin(3i) for i in 1:n]
        rng_du = [0.2 * cos(5i) for i in 1:n]
        J = copy(probe.f.jac_prototype)
        probe.f.jac(J, rng_du, rng_u, probe.p, 0.7, 0.4)
        Jfd = fd_jacobian(probe, rng_du, rng_u, 0.7, 0.4)
        @test norm(Matrix(J) - Jfd, Inf) <= 1.0e-7 * norm(Jfd, Inf)
    end

    @testset "a parameter changed by remake is rejected, not ignored" begin
        # The operators, the nonlinear one included, hold the values they were assembled with (spec §8.3).
        Ω = square()
        eq = Dt(U) ~ Dx((1 + k * U^2) * Dx(U)) + Dy(Dy(U)) + (1 + x + 2y) - 2k * ustar * (1 + t)^2
        prob = DAEProblem(pdesystem([eq], [dirichlet(Ω); ic], Ω; ps = [k => 1.0]), FEMDiscretization(Ω, t))
        sol = solve(prob, EXACT_ROWS; abstol = 1.0e-10, reltol = 1.0e-10)
        @test nodal_error(sol, U, uexact) <= tol
        @test SciMLBase.successful_retcode(solve(remake(prob; p = [k => 1.0]), EXACT_ROWS))
        msg = "parameter k was 1.0 when the operators were assembled but is 3.0 in this problem; operators are not reassembled for new parameter values, so build a new problem with DAEProblem(pdesys, disc)"
        @test_throws ArgumentError(msg) solve(remake(prob; p = [k => 3.0]), EXACT_ROWS)
    end
end

@testset verbose = true "still-rejected nonlinear classes" begin
    Ω = square()
    lower(eqs, bcs = [dirichlet(Ω); ic]) = DAEProblem(pdesystem(eqs, bcs, Ω), FEMDiscretization(Ω, t))
    # Raised by PDEBase's strong-form analysis with the messages of spec Appendix A.
    @test_throws ErrorException("state-dependent mass in Differential(t, 1)(u(t, x, y))*(u(t, x, y)^2) needs residual-form DAE") lower(
        [(1 + U^2) * Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))]
    )
    @test_throws ErrorException("time derivative in Differential(t, 1)(u(t, x, y))^2 is not ∂ₜu × coefficient") lower(
        [Dt(U)^2 ~ Dx(Dx(U)) + Dy(Dy(U))]
    )
    @test_throws ErrorException(
        "time derivative in -Differential(x, 1)(Differential(t, 1)(Differential(x, 1)(u(t, x, y)))) is not ∂ₜu × coefficient"
    ) lower([Dt(U) ~ Dx(Dt(Dx(U))) + Dy(Dy(U))])
    @test_throws ErrorException("second derivative not in a divergence-compatible position: -sin(Differential(x, 2)(u(t, x, y)))") lower(
        [Dt(U) ~ sin(Dx(Dx(U))) + Dy(Dy(U))]
    )
    # The printed order of a product's factors differs between Julia versions, so these two match the
    # message up to the term.
    @test_throws "second derivative not in a divergence-compatible position: -" lower([Dt(U) ~ Dx(Dx(U)) * Dy(Dy(U))])
    @test_throws "second derivative not in a divergence-compatible position: -" lower([Dt(U) ~ Dx(U) * Dx(Dx(U)) + Dy(Dy(U))])
    Γl = first(sides(Ω))
    @test_throws ErrorException("Dirichlet data must be linear in one field: u(t, Γ_left)^2 ~ 1 on Γ_left") lower(
        [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [[u(t, Γl)^2 ~ 1]; dirichlet(Ω, sides(Ω)[2:end]); ic]
    )
    # The front end's own: a time-dependent coefficient in a linear operator would need reassembly.
    @test_throws ArgumentError(
        "time-dependent coefficients in the mass or stiffness operator are not supported: equation 1 has G₁*W₁ + G₁*T*W₁"
    ) lower([Dt(U) ~ Dx((1 + t) * Dx(U))])
end
