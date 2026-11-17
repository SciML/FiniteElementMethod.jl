# The vector-calculus operators of Symbolics (`Grad`, `Divergence`, `Curl`, `Laplacian`, written `grad = Grad(vars)`,
# `grad ⋅ u`, `grad × u`) in the strong form. Each case builds the equation both ways, with the operators and with the
# hand-expanded `Differential`s, and compares the assembled Galerkin operators, so the operators are accepted and give
# the same discrete system rather than merely a similar one.
using FiniteElementMethod, ModelingToolkit, DomainSets, NonlinearSolveFirstOrder, Test
using FiniteElementMethod: operators, lower_vector_ops
using Ferrite: generate_grid, Triangle, Tetrahedron, Vec
using LinearAlgebra: ⋅, ×
using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, Grad, Divergence, Curl, Laplacian

@variables x y z u(..) v(..) w(..) p(..) E1(..) E2(..) E3(..)
Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)

X = [x, y]
grad = Grad(X)
k(x_, y_) = 1 + x_ + y_^2

square(n = 3) = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
field(dv) = Symbolics.operation(Symbolics.unwrap(dv))

# The steady system `eqs` with Dirichlet data `g` on every side, on P2 triangles.
function steady_system(eqs, dvs, g; n = 3)
    Ω = square(n)
    bcs = [field(dv)(Γ) ~ g for dv in dvs for Γ in sides(Ω)]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω], [x, y], dvs, [])
    disc = FEMDiscretization(Ω; spaces = Dict(field(dv) => Lagrange{RefTriangle, 2}() for dv in dvs))
    return pdesys, disc
end

steady_operators(args...) = operators(steady_system(args...)...)

function transient_operators(eq, ic)
    Ω = square()
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ ic]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    return operators(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}())))
end

same_operators(a, b) = a.M ≈ b.M && a.K ≈ b.K && a.F ≈ b.F && a.u0 ≈ b.u0

@testset verbose = true "vector-calculus operators in the strong form" begin
    ut = u(t, x, y)
    f = sin(pi * x) * y

    @testset "$name heat equation matches the hand-expanded form" for (name, rhs) in (
            ("lazy ∇⋅(k ∇u)", grad ⋅ (k(x, y) * grad(ut)) + f),
            ("eager ∇⋅(k ∇u)", Grad{true}(X) ⋅ (k(x, y) * Grad{true}(X)(ut)) + f),
            ("lazy Laplacian{false}", Laplacian{false}(X)(ut) + f),
            ("eager Laplacian{true}", Laplacian{true}(X)(ut) + f),
            ("lazy ∇⋅∇", (grad ⋅ grad)(ut) + f),
        )
        divergence_form = occursin("k", name)
        hand = divergence_form ? Dt(ut) ~ Dx(k(x, y) * Dx(ut)) + Dy(k(x, y) * Dy(ut)) + f : Dt(ut) ~ Dx(Dx(ut)) + Dy(Dy(ut)) + f
        @test same_operators(transient_operators(Dt(ut) ~ rhs, x * y), transient_operators(hand, x * y))
    end

    @testset "lowering produces the Differential form and leaves other systems alone" begin
        lowered = only(lower_vector_ops([Dt(ut) ~ grad ⋅ (k(x, y) * grad(ut))]))
        @test isequal(lowered.rhs, Dx(k(x, y) * Dx(ut)) + Dy(k(x, y) * Dy(ut)))
        @test isequal(only(lower_vector_ops([0 ~ Laplacian(X)(ut)])).rhs, Dx(Dx(ut)) + Dy(Dy(ut)))
        Ω = square()
        plain = Dt(ut) ~ Dx(Dx(ut)) + Dy(Dy(ut))
        @named pdesys = PDESystem([plain], [u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [ut], [])
        @test lower_vector_ops(pdesys) === pdesys
    end

    @testset "Stokes system with ∇⋅[u, v] and ∇p matches the hand-expanded form" begin
        ν = 0.1
        U, V, P = u(x, y), v(x, y), p(x, y)
        lap(s) = Dx(Dx(s)) + Dy(Dy(s))
        hand = [0 ~ ν * lap(U) - Dx(P), 0 ~ ν * lap(V) - Dy(P), 0 ~ Dx(U) + Dy(V)]
        lazy = [0 ~ ν * Laplacian(X)(U) - grad(P)[1], 0 ~ ν * Laplacian(X)(V) - grad(P)[2], 0 ~ grad ⋅ [U, V]]
        eager = [0 ~ ν * Laplacian{true}(X)(U) - Grad{true}(X)(P)[1], 0 ~ ν * Laplacian{true}(X)(V) - Grad{true}(X)(P)[2], 0 ~ Divergence{true}(X)([U, V])]
        a = steady_operators(hand, [U, V, P], 0.0)
        for eqs in (lazy, eager)
            @test same_operators(steady_operators(eqs, [U, V, P], 0.0), a)
        end
        # The transport term (u⋅∇)u written with the operator.
        advect = [0 ~ ν * Laplacian(X)(U) - ([U, V] ⋅ grad)(U) - Dx(P), 0 ~ ν * Laplacian(X)(V) - ([U, V] ⋅ grad)(V) - Dy(P), 0 ~ Dx(U) + Dy(V)]
        hand_advect = [0 ~ ν * lap(U) - U * Dx(U) - V * Dy(U) - Dx(P), 0 ~ ν * lap(V) - U * Dx(V) - V * Dy(V) - Dy(P), 0 ~ Dx(U) + Dy(V)]
        steady_prob(eqs) = discretize(steady_system(eqs, [U, V, P], 0.0)...)
        pa, pb = steady_prob(advect), steady_prob(hand_advect)
        @test pa.u0 == pb.u0
        res_a, res_b = similar(pa.u0), similar(pb.u0)
        pa.f(res_a, pa.u0 .+ 0.1 .* (1:length(pa.u0)) ./ length(pa.u0), pa.p)
        pb.f(res_b, pb.u0 .+ 0.1 .* (1:length(pb.u0)) ./ length(pb.u0), pb.p)
        @test res_a ≈ res_b
    end

    @testset "scalar curl in two dimensions" begin
        U, V, W = u(x, y), v(x, y), w(x, y)
        hand = [0 ~ Dx(Dx(U)) + Dy(Dy(U)) + Dx(V), 0 ~ Dx(Dx(V)) + Dy(Dy(V)) - Dy(U), 0 ~ Dx(V) - Dy(U) - W]
        lazy = [hand[1], hand[2], 0 ~ Curl(X)([U, V]) - W]
        eager = [hand[1], hand[2], 0 ~ Curl{true}(X)([U, V]) - W]
        a = steady_operators(hand, [U, V, W], 0.0)
        @test same_operators(steady_operators(lazy, [U, V, W], 0.0), a)
        @test same_operators(steady_operators(eager, [U, V, W], 0.0), a)
    end

    @testset "curl curl in three dimensions, as vector equations and with ∇ ×" begin
        X3 = [x, y, z]
        grad3 = Grad(X3)
        Ω = MeshDomain(generate_grid(Tetrahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0)))
        faces = boundarypiece.(Ω, (:left, :right, :bottom, :top, :front, :back))
        E = [E1(x, y, z), E2(x, y, z), E3(x, y, z)]
        rhs = [x * y, y * z, z + x]
        scalar_curl(F) = [Dy(F[3]) - Dz(F[2]), Dz(F[1]) - Dx(F[3]), Dx(F[2]) - Dy(F[1])]
        hand = [c + E[i] ~ rhs[i] for (i, c) in enumerate(scalar_curl(scalar_curl(E)))]
        function system(eqs)
            bcs = [field(dv)(Γ) ~ 0 for dv in E for Γ in faces]
            @named pdesys = PDESystem(eqs, bcs, [(x, y, z) ∈ Ω], [x, y, z], E, [])
            return operators(pdesys, FEMDiscretization(Ω; spaces = Dict(field(e) => Lagrange{RefTetrahedron, 2}() for e in E)))
        end
        ref = system(hand)
        @testset "$name" for (name, eqs) in (
                ("Curl(Curl(E)) as a vector equation", Curl(X3)(Curl(X3)(E)) + E ~ rhs),
                ("∇ × (∇ × E) with cross", grad3 × (grad3 × E) + E ~ rhs),
                ("eager Curl{true}", Curl{true}(X3)(Curl{true}(X3)(E)) + E .~ rhs),
            )
            @test same_operators(system(eqs), ref)
        end
    end

    @testset "unknown given as an array variable" begin
        @variables A(x, y)[1:2]
        function system(eqs)
            @named pdesys = PDESystem(eqs, Equation[], [(x, y) ∈ square()], [x, y], [A], [])
            return operators(pdesys, FEMDiscretization(pdesys.domain[1].domain; boundary_check = :off))
        end
        hand = [0 ~ Dx(A[1]) + Dy(A[2]) + A[1], 0 ~ Dx(Dx(A[1])) + Dy(Dy(A[2])) + A[2]]
        @test same_operators(system([0 ~ grad ⋅ A + A[1], hand[2]]), system(hand))
    end

    @testset "the two-dimensional cross product grad × [u, v], where Symbolics has it" begin
        U, V, W = u(x, y), v(x, y), w(x, y)
        cross2 = try
            grad × [U, V]
        catch
            nothing   # Symbolics defines `×` for three variables only; `Curl(X)([u, v])` is the 2D curl
        end
        if cross2 !== nothing
            eqs(c) = [0 ~ Dx(Dx(U)) + Dy(Dy(U)) + Dx(V), 0 ~ Dx(Dx(V)) + Dy(Dy(V)) - Dy(U), 0 ~ c - W]
            @test same_operators(steady_operators(eqs(cross2), [U, V, W], 0.0), steady_operators(eqs(Dx(V) - Dy(U)), [U, V, W], 0.0))
        end
    end

    @testset "a lazy equation and its expansion as two option keys must agree" begin
        U, V = u(x, y), v(x, y)
        lazy = 0 ~ Laplacian(X)(U)
        hand = 0 ~ Dx(Dx(U)) + Dy(Dy(U))
        other = 0 ~ Laplacian(X)(V) + U
        function system(options)
            Ω = square()
            bcs = [field(dv)(Γ) ~ 0 for dv in (U, V) for Γ in sides(Ω)]
            @named pdesys = PDESystem([lazy, other], bcs, [(x, y) ∈ Ω], [x, y], [U, V], [])
            return operators(pdesys, FEMDiscretization(Ω; options...))
        end
        agreeing = system((; ibp = Dict(lazy => Weighted(1 + x), hand => Weighted(1 + x)), pairing = Dict(lazy => u, hand => u)))
        @test same_operators(agreeing, system((; ibp = Dict(hand => Weighted(1 + x)), pairing = Dict(hand => u))))
        message(options) = try
            system(options)
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("ibp keys", message((; ibp = Dict(lazy => Weighted(1 + x), hand => Weighted(1 + y)))))
        @test occursin("pairing keys", message((; pairing = Dict(lazy => u, hand => v))))
    end

    @testset "Navier–Stokes in the operator form lowers to exactly the hand-written equations" begin
        @parameters ν
        U, V, P = u(t, x, y), v(t, x, y), p(t, x, y)
        lazy = [
            Dt(U) ~ grad ⋅ (ν * grad(U)) - ([U, V] ⋅ grad)(U) - grad(P)[1],
            Dt(V) ~ grad ⋅ (ν * grad(V)) - ([U, V] ⋅ grad)(V) - grad(P)[2],
            0 ~ grad ⋅ [U, V],
        ]
        hand = [
            Dt(U) ~ Dx(ν * Dx(U)) + Dy(ν * Dy(U)) - U * Dx(U) - V * Dy(U) - Dx(P),
            Dt(V) ~ Dx(ν * Dx(V)) + Dy(ν * Dy(V)) - U * Dx(V) - V * Dy(V) - Dy(P),
            0 ~ Dx(U) + Dy(V),
        ]
        @test all(isequal.(lower_vector_ops(lazy), hand))
    end

    @testset "operators nested in Differentials, in operators and in functions lower without looping" begin
        U = u(x, y)
        Dxx_Dyy = Dx(Dx(U)) + Dy(Dy(U))
        @test !FiniteElementMethod._has_vector_op(only(lower_vector_ops([0 ~ Dx(grad(U)[1])])).rhs)
        @test isequal(only(lower_vector_ops([0 ~ Dy(grad ⋅ grad(U))])).rhs, Dy(Dxx_Dyy))
        @test isequal(only(lower_vector_ops([0 ~ grad ⋅ [Dx(U), Dy(U)]])).rhs, Dxx_Dyy)
        @test isequal(only(lower_vector_ops([0 ~ grad ⋅ (Grad(X)(Laplacian(X)(U)))])).rhs, Dx(Dx(Dxx_Dyy)) + Dy(Dy(Dxx_Dyy)))
        @test isequal(only(lower_vector_ops([0 ~ sin(Laplacian(X)(U))])).rhs, sin(Dxx_Dyy))
        poisson(eq) = steady_operators([eq], [U], 0.0)
        @test same_operators(poisson(0 ~ Dx(grad(U)[1]) + Dy(grad(U)[2])), poisson(0 ~ Dxx_Dyy))
    end

    @testset "equation-keyed ibp and pairing options accept the operator form of the equation" begin
        U, V = u(x, y), v(x, y)
        w = 1 + x
        lazy = 0 ~ Laplacian(X)(U)
        hand = 0 ~ Dx(Dx(U)) + Dy(Dy(U))
        other = 0 ~ Laplacian(X)(V) + U
        operators_with(eq, key) = begin
            Ω = square()
            bcs = [field(dv)(Γ) ~ 0 for dv in (U, V) for Γ in sides(Ω)]
            @named pdesys = PDESystem([eq, other], bcs, [(x, y) ∈ Ω], [x, y], [U, V], [])
            disc = FEMDiscretization(Ω; ibp = Dict(key => Weighted(w)), pairing = Dict(key => u))
            operators(pdesys, disc)
        end
        @test same_operators(operators_with(lazy, lazy), operators_with(hand, hand))
    end

    @testset "an operator in a variable that is not an independent variable of the system is an error" begin
        Ω = square()
        @named pdesys = PDESystem([0 ~ Laplacian([x, z])(u(x, y))], Equation[], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
        err = try
            lower_vector_ops(pdesys)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("z is not an independent variable of the PDESystem", err.msg)
    end

    @testset "a steady Poisson problem written with the Laplacian converges like the hand-written one" begin
        ustar(x_, y_) = sin(pi * x_) * exp(y_)
        function solve_poisson(eq, n)
            Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
            @named pdesys = PDESystem([eq], [u(Γ) ~ ustar(x, y) for Γ in sides(Ω)], [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
            prob = discretize(pdesys, FEMDiscretization(Ω; spaces = Dict(u => Lagrange{RefTriangle, 2}())))
            return solve(prob, NewtonRaphson(); abstol = 1.0e-12)
        end
        forcing = (1 - pi^2) * ustar(x, y)
        a = solve_poisson(Laplacian(X)(u(x, y)) ~ forcing, 8)
        b = solve_poisson(Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ forcing, 8)
        @test a[u(x, y)] ≈ b[u(x, y)] rtol = 1.0e-12
        pts = range(0.0123, 0.9877; length = 21)
        @test maximum(abs(a(px, py; dv = u(x, y)) - ustar(px, py)) for px in pts, py in pts) < 5.0e-3
    end
end
