# End-to-end SUPG through the PDESystem front end (spec §1 goal 4, §10.1 SUPG row, §11.1 M4):
# FEMDiscretization(Ω, t; test_space = SUPG(τ)) on a convection-dominated problem, against Galerkin.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle, Quadrilateral, Line, Vec, RefQuadrilateral
using LinearAlgebra: norm, tr, ⋅

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, expand_derivatives

@variables x y u(..) w(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

# u_t = εu″ − u′ on (0, 1), u(0) = 0, u(1) = 1, u(0, x) = x, marched to its steady state, the boundary layer
# (e^{x/ε} − 1)/(e^{1/ε} − 1). With 20 cells the cell Péclet number is 2.5.
const EPS = 0.01
layer(x_) = expm1(x_ / EPS) / expm1(1 / EPS)
optimal_τ(hK, b) = (Pe = abs(b[1]) * hK / (2EPS); hK / (2abs(b[1])) * (coth(Pe) - 1 / Pe))

function boundary_layer(test_space; nel = 20)
    Ω = MeshDomain(rectangle_grid(Line, (nel,)))
    lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
    eq = Dt(u(t, x)) ~ EPS * Dx(Dx(u(t, x))) - Dx(u(t, x))
    sol, _ = fem_solve([eq], [u(t, lo) ~ 0.0, u(t, hi) ~ 1.0, u(0, x) ~ x], Ω, (x,), t, [u(t, x)]; tspan = (0.0, 5.0), test_space)
    @test SciMLBase.successful_retcode(sol)
    steady = sol[u(t, x)][end, :]
    return steady, maximum(abs.(steady .- layer.(sol[x])))
end

# M and K of ∂ₜu = ε Δu − b·∇u on a P1 quadrilateral grid, assembled with Ferrite as the SUPG weak form
# ∫ (v + τ b·∇v)(u_t) + ε ∇v·∇u + v b·∇u + τ (b·∇v)(−ε Δu + b·∇u), with the cell diameter hK and the front end's
# quadrature order 2. A τ of 0 is Galerkin.
function supg_reference(grid, τ, b)
    ip = Ferrite.Lagrange{RefQuadrilateral, 1}()
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{RefQuadrilateral}(2), ip; update_hessians = true)
    M, K = Ferrite.allocate_matrix(dh), Ferrite.allocate_matrix(dh)
    n = Ferrite.getnbasefunctions(cv)
    Me, Ke = zeros(n, n), zeros(n, n)
    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Me, 0)
        fill!(Ke, 0)
        xs = Ferrite.getcoordinates(cell)
        hK = maximum(norm(xs[i] - xs[j]) for i in eachindex(xs) for j in eachindex(xs))
        for q in 1:Ferrite.getnquadpoints(cv)
            dΩ = Ferrite.getdetJdV(cv, q)
            bq = b(Ferrite.spatial_coordinate(cv, q, xs))
            τq = τ(hK, bq)
            for i in 1:n
                v, ∇v = Ferrite.shape_value(cv, q, i), Ferrite.shape_gradient(cv, q, i)
                for j in 1:n
                    ψ, ∇ψ, Hψ = Ferrite.shape_value(cv, q, j), Ferrite.shape_gradient(cv, q, j), Ferrite.shape_hessian(cv, q, j)
                    Me[i, j] += (v + τq * (bq ⋅ ∇v)) * ψ * dΩ
                    Ke[i, j] += (EPS * (∇v ⋅ ∇ψ) + v * (bq ⋅ ∇ψ) + τq * (bq ⋅ ∇v) * (-EPS * tr(Hψ) + bq ⋅ ∇ψ)) * dΩ
                end
            end
        end
        dofs = Ferrite.celldofs(cell)
        M[dofs, dofs] += Me
        K[dofs, dofs] += Ke
    end
    return M, K
end

@testset "SUPG operators on quadrilaterals against a hand assembly" begin
    Ω = MeshDomain(rectangle_grid(Quadrilateral, (8, 8)))
    U = u(t, x, y)
    bcs = [[u(t, g) ~ 0 for g in boundarypieces(Ω)]; u(0, x, y) ~ 0]
    dom = [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)]
    # coordinate-dependent advection velocity (1 + y, x), and a constant one (1, 1/2)
    @named varying = PDESystem(
        [Dt(U) ~ EPS * (Dx(Dx(U)) + Dy(Dy(U))) - (1 + y) * Dx(U) - x * Dy(U)], bcs, dom, [t, x, y], [U], []
    )
    @named constant = PDESystem(
        [Dt(U) ~ EPS * (Dx(Dx(U)) + Dy(Dy(U))) - Dx(U) - 0.5 * Dy(U)], bcs, dom, [t, x, y], [U], []
    )
    b_varying(X) = Vec(1 + X[2], X[1])
    b_constant(X) = Vec(1.0, 0.5)
    quotient(hK, b) = hK / (2 * sqrt(b[1]^2 + b[2]^2))
    # (name, system, test space, τ(hK, b), b(X)); every case discretizes and matches its hand assembly.
    cases = [
        ("Galerkin", varying, Galerkin(), (hK, b) -> 0.0, b_varying),
        ("τ = hK / 2, b from the equation", varying, SUPG((hK, b) -> 0.5 * hK), (hK, b) -> 0.5 * hK, b_varying),
        ("τ = hK / 2|b|, constant b", constant, SUPG(quotient), quotient, b_constant),
        ("explicit b holding the coordinates", varying, SUPG(0.05; b = [1 + y, x]), (hK, b) -> 0.05, b_varying),
        ("τ = hK / 2|b|, b from the equation", varying, SUPG(quotient), quotient, b_varying),
        ("number τ, constant b", constant, SUPG(0.05), (hK, b) -> 0.05, b_constant),
        ("τ function without hK", varying, SUPG((hK, b) -> 0.05), (hK, b) -> 0.05, b_varying),
    ]
    @testset "$name" for (name, pdesys, space, τ, b) in cases
        disc = FEMDiscretization(Ω, t; test_space = space)
        @test discretize(pdesys, disc) isa SciMLBase.DAEProblem
        ops = FiniteElementMethod.operators(pdesys, disc)
        M, K = supg_reference(Ω.mesh, τ, b)
        @test norm(ops.M - M) <= 1.0e-12 * norm(M)
        @test norm(ops.K - K) <= 1.0e-12 * norm(K)
    end
end

@testset verbose = true "SUPG front end" begin
    @testset "boundary layer: Galerkin oscillates, SUPG is monotone" begin
        steady, err = boundary_layer(Galerkin())
        @test minimum(steady) < -0.4
        @test err > 0.4
        # The optimal τ makes the steady state nodally exact in 1D.
        steady, err = boundary_layer(SUPG(optimal_τ))
        @test minimum(steady) >= -1.0e-10 && maximum(steady) <= 1 + 1.0e-10
        @test err <= 1.0e-9
        # τ = h/(2|b|) is monotone too, with the layer smeared over a cell.
        steady, err = boundary_layer(SUPG((hK, b) -> hK / (2abs(b[1]))))
        @test minimum(steady) >= -1.0e-10 && maximum(steady) <= 1 + 1.0e-10
        @test err < 0.2
        # An explicit b and a constant τ give the same as the inferred b with τ = h/(2|b|) at h = 1/20.
        steady_b, _ = boundary_layer(SUPG(1 / 40; b = [1.0]))
        @test steady_b ≈ steady atol = 1.0e-8
    end

    @testset "smooth manufactured solution, $(ct) P$(order): rate p + 1/2 or better" for (ct, order, rate) in
        ((Quadrilateral, 1, 1.4), (Triangle, 2, 1.9))
        # u* = sin πx sin πy with b = (1, 1/2), ε = 0.01: the cell Péclet number falls from 7 to 1.7 over the
        # refinements. The optimal τ tends to h²/(12ε) once diffusion dominates a cell; τ = h/(2|b|) would not.
        U = u(t, x, y)
        us = sin(pi * x) * sin(pi * y)
        f = expand_derivatives(-EPS * (Dx(Dx(us)) + Dy(Dy(us))) + Dx(us) + 0.5 * Dy(us))
        eq = Dt(U) ~ EPS * (Dx(Dx(U)) + Dy(Dy(U))) - Dx(U) - 0.5 * Dy(U) + f
        nb = sqrt(1.25)
        τ(hK, b) = (Pe = nb * hK / (2EPS); hK / (2nb) * (coth(Pe) - 1 / Pe))
        shape = ct === Triangle ? RefTriangle : RefQuadrilateral
        errs = map((8, 16, 32)) do n
            Ω = MeshDomain(rectangle_grid(ct, (n, n)))
            sol, _ = fem_solve(
                [eq], [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ us], Ω, (x, y), t, [U];
                tspan = (0.0, 0.5), test_space = SUPG(τ), spaces = Dict(u => Lagrange{shape, order}())
            )
            @test SciMLBase.successful_retcode(sol)
            pts = [Vec(a, c) for a in range(0.01, 0.99, 40) for c in range(0.01, 0.99, 40)]
            vals = sol(0.5, pts; dv = U)
            sqrt(sum(abs2, vals .- [sin(pi * q[1]) * sin(pi * q[2]) for q in pts]) / length(pts))
        end
        @test all(>(rate), [log2(errs[i] / errs[i + 1]) for i in 1:2])
    end

    @testset "an equation without advection stays Galerkin" begin
        # u is advected, w only diffuses: SUPG(τ(hK, b)) leaves w's equation as Galerkin, even though τ(hK, 0) is
        # not finite, and the two fields are uncoupled, so w matches a Galerkin solve.
        Ω = MeshDomain(rectangle_grid(Line, (20,)))
        lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
        eqs = [Dt(u(t, x)) ~ EPS * Dx(Dx(u(t, x))) - Dx(u(t, x)), Dt(w(t, x)) ~ Dx(Dx(w(t, x))) + 1]
        bcs = [u(t, lo) ~ 0.0, u(t, hi) ~ 1.0, w(t, lo) ~ 0.0, w(t, hi) ~ 0.0, u(0, x) ~ x, w(0, x) ~ 0]
        solve_with(ts) = first(fem_solve(eqs, bcs, Ω, (x,), t, [u(t, x), w(t, x)]; tspan = (0.0, 5.0), test_space = ts))
        supg, galerkin = solve_with(SUPG(optimal_τ)), solve_with(Galerkin())
        @test supg[w(t, x)][end, :] ≈ galerkin[w(t, x)][end, :] atol = 1.0e-8
        @test maximum(abs.(supg[u(t, x)][end, :] .- layer.(supg[x]))) <= 1.0e-9
    end

    @testset "error paths" begin
        Ω = MeshDomain(rectangle_grid(Line, (10,)))
        lo, hi = boundarypiece.(Ω, (:x1_lo, :x1_hi))
        @parameters c k
        bcs = [u(t, lo) ~ 0.0, u(t, hi) ~ 1.0, u(0, x) ~ x]
        lower(eq, ts; ps = []) = discretize(
            PDESystem([eq], bcs, [x ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)], ps; name = :sys),
            FEMDiscretization(Ω, t; test_space = ts)
        )
        @test_throws ArgumentError("test_space must be Galerkin() or SUPG(τ; b = nothing), got :supg") FEMDiscretization(
            Ω, t; test_space = :supg
        )
        advected(β) = Dt(u(t, x)) ~ EPS * Dx(Dx(u(t, x))) - β * Dx(u(t, x))
        W1 = Symbolics.variable(:W, 1)
        @test_throws ArgumentError(
            "SUPG: equation 1 has τ = 0.05 and b·∇v = $(c * W1), which reference parameters of the PDESystem; τ and b " *
                "must be numbers or expressions in the coordinates, time and hK, so write the parameter's value"
        ) lower(advected(c), SUPG(0.05; b = [c]); ps = [c => 1.0])
        # The same error with b inferred from the equation, so τ and b·∇v hold the analysis' parameter slot, whose
        # printed form is the analysis' business: the fixed text around them is checked.
        msg = errmsg(() -> lower(advected(c), SUPG((hK, b) -> hK / (2abs(b[1]))); ps = [c => 1.0]))
        @test startswith(msg, "ArgumentError: SUPG: equation 1 has τ = ")
        @test endswith(
            msg, "which reference parameters of the PDESystem; τ and b must be numbers or expressions in the " *
                "coordinates, time and hK, so write the parameter's value"
        )
        @test_throws ArgumentError(
            "SUPG: equation 1 has τ = k and b·∇v = $W1, which reference parameters of the PDESystem; τ and b must be " *
                "numbers or expressions in the coordinates, time and hK, so write the parameter's value"
        ) lower(advected(1), SUPG(k); ps = [k => 0.05])
        # A parameter elsewhere in the equation is fine.
        sol = solve(lower(Dt(u(t, x)) ~ k * Dx(Dx(u(t, x))) - Dx(u(t, x)), SUPG(0.05); ps = [k => 0.01]), EXACT_ROWS)
        @test SciMLBase.successful_retcode(sol)
        # b = [U₁] in the analysis' state slot, printed as the error prints it (module-qualified under Pkg.test).
        b = Num[Symbolics.variable(:U, 1)]
        @test_throws ArgumentError(
            "SUPG: the advection velocity ∂R/∂∇u of equation 1 is $b, which depends on the state; pass SUPG(τ; b = ...)"
        ) lower(Dt(u(t, x)) ~ EPS * Dx(Dx(u(t, x))) - u(t, x) * Dx(u(t, x)), SUPG(0.05))
        # Nonlinear terms are allowed with Galerkin, but SUPG stabilises linear terms only.
        @test_throws ArgumentError("SUPG stabilises linear terms only, but the system has a nonlinear term; use test_space = Galerkin()") lower(
            Dt(u(t, x)) ~ EPS * Dx(Dx(u(t, x))) - Dx(u(t, x)) - u(t, x)^2, SUPG(0.05)
        )
    end
end
