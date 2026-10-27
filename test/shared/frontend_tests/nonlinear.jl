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
        "time-dependent coefficients in the mass or stiffness operator land in M3: equation 1 has G₁*W₁ + G₁*T*W₁"
    ) lower([Dt(U) ~ Dx((1 + t) * Dx(U))])
end
