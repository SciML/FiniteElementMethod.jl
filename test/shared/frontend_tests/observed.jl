# End-to-end tests of observed expressions, `sol[expr]` and `sol(t, xs...; dv = expr)` (spec §7.8,
# O9): PDESystem -> DAEProblem -> solve, then each expression is checked against the same
# expression computed from the field arrays and against the exact solution.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, piece_nodes, normal_flux
using Ferrite: Triangle, Vec

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..) v(..) z(..) w
@parameters k
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# u* = (1 + x + 2y)(1 + t) solves u_t = Δu + (1 + x + 2y) and is in the P1 space.
uexact(t_, x_, y_) = (1 + x_ + 2y_) * (1 + t_)
heat_eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x + 2y)
pieces(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
heat_bcs(Ω) = [[u(t, Γ) ~ (1 + x + 2y) * (1 + t) for Γ in pieces(Ω)]; u(0, x, y) ~ 1 + x + 2y]

observed_error(ex, why) = ArgumentError("cannot evaluate the expression $ex on this solution: $why")

@testset verbose = true "observed expressions" begin
    Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
    sol, _ = fem_solve([heat_eq], heat_bcs(Ω), Ω, (x, y), t, [u(t, x, y)]; ps = [k => 2.0])
    U = u(t, x, y)
    T, X, Y = sol.t, permutedims(sol[x]), permutedims(sol[y])
    nt, nv = length(T), length(X)

    @testset "heat: sol[expr] against the field arrays and the exact solution" begin
        @test size(sol[U^2 + k * x]) == (nt, nv)
        @test sol[U^2 + k * x] ≈ sol[U] .^ 2 .+ 2 .* X atol = 1.0e-12
        @test maximum(abs, sol[U^2 + k * x] .- (uexact.(T, X, Y) .^ 2 .+ 2 .* X)) <= 1.0e-6
        @test sol[2U] == 2 .* sol[U]
        @test sol[sin(t) * U] ≈ sin.(T) .* sol[U] atol = 1.0e-12
        # P1 reproduces the linear u*, so its projected gradient is (1 + t, 2(1 + t)).
        @test sol[Dx(U) * U] ≈ sol[Dx(U)] .* sol[U] atol = 1.0e-12
        @test maximum(abs, sol[Dx(U) + Dy(U)] .- 3 .* (1 .+ T)) <= 1.0e-6
    end

    @testset "coordinate, time and parameter expressions" begin
        @test sol[x^2 + y] == repeat(X .^ 2 .+ Y, nt)
        @test sol[t * x + k] ≈ T .* X .+ 2 atol = 1.0e-14
        @test sol[2k] == fill(4.0, nt, nv)
    end

    @testset "point evaluation sol(t, xs...; dv = expr)" begin
        @test sol(0.5, 0.3, 0.2; dv = U^2 + k * x) ≈ sol(0.5, 0.3, 0.2; dv = U)^2 + 0.6 atol = 1.0e-12
        @test sol(0.5, 0.3, 0.2; dv = U^2 + k * x) ≈ uexact(0.5, 0.3, 0.2)^2 + 0.6 atol = 1.0e-6
        xs, ys = range(0, 1, length = 5), range(0, 1, length = 4)
        vals = sol(T, xs, ys; dv = t * U + Dy(U))
        @test size(vals) == (nt, 5, 4)
        @test vals ≈ [
            T[n] * sol(T[n], xs[i], ys[j]; dv = U) + sol(T[n], xs[i], ys[j]; dv = Dy(U))
                for n in 1:nt, i in 1:5, j in 1:4
        ] atol = 1.0e-12
        @test size(sol(0.5, xs, 0.2; dv = 2U)) == (5,)
        # A coordinate-only expression is NaN outside the mesh, like a field value.
        @test isequal(sol(0.5, [Vec(0.3, 0.2), Vec(2.0, 0.0)]; dv = x + y), [0.5, NaN])
        @test_throws DomainError sol(0.5, 2.0, 0.0; dv = 2x, outside = :error)
        # Outside points are not evaluated, so an expression undefined there is NaN, not an error.
        @test isnan(sol(0.5, -1.0, 0.5; dv = sqrt(x)))
        @test isequal(sol(0.5, [0.25, -1.0], 0.5; dv = log(x) + U), [log(0.25) + sol(0.5, 0.25, 0.5; dv = U), NaN])
        # An integer-valued expression is returned as floating point so outside points hold NaN.
        @test sol(0.5, 0.3, 0.5; dv = 2k) === 4.0
        @test isequal(sol(0.5, [0.3, 2.0], 0.5; dv = 2k), [4.0, NaN])
        @test isnan(sol(0.5, 2.0, 0.5; dv = 2k))
        Γ = first(pieces(Ω))
        nodes = piece_nodes(sol, Γ)
        @test sol(0.5, Γ; dv = U + y) ≈ sol(0.5, Γ; dv = U) .+ sol[y][nodes] atol = 1.0e-14
        @test sol(T, Γ; dv = t * U) ≈ T .* sol[u(t, Γ)] atol = 1.0e-12
        err = errmsg(() -> sol(0.5, Γ; dv = Dx(U) + 1))
        @test occursin("traces of derivatives of u(t, x, y) are not supported", err)
    end

    @testset "error paths" begin
        dvs, ivs = "u(t, x, y)", "t, x, y"
        unknown(s) = "$s is not a dependent variable, an independent variable or a parameter; the dependent variables are $dvs and the independent variables $ivs"
        derivative(d) = "$d is not a first spatial derivative of a dependent variable, the only derivatives an expression accepts"
        Γ = first(pieces(Ω))
        cases = [
            U * z(t, x, y) => unknown(z(t, x, y)),
            2w => unknown(w),
            u(t, Γ)^2 => "it contains the trace $(u(t, Γ)); an expression is evaluated at every vertex, so write u(t, x, y) instead",
            Dx(Dx(U)) + 1 => derivative(Dx(Dx(U))),
            2Dt(U) => derivative(Dt(U)),
            u(t, 0, y) + 1 => "$(u(t, 0, y)) fixes arguments of u(t, x, y); write the dependent variable as u(t, x, y)",
            Dy(u(t, 0, y)) + 1 => "$(u(t, 0, y)) fixes arguments of u(t, x, y); write the dependent variable as u(t, x, y)",
        ]
        for (ex, why) in cases
            @test_throws observed_error(ex, why) sol[ex]
            @test_throws observed_error(ex, why) sol(0.5, 0.3, 0.2; dv = ex)
        end
        # A derivative key that is not an expression keeps the key error.
        @test occursin("first", errmsg(() -> sol[Dx(Dx(U))]))
    end

    @testset "a dependent variable at fixed arguments is rejected on every path" begin
        Γ = first(pieces(Ω))
        fixed(ex) = ArgumentError("$ex fixes arguments of u(t, x, y); write the dependent variable as u(t, x, y)")
        for ex in (u(t, 0, y), u(t, x, 0.5), u(0, x, y))
            @test_throws fixed(ex) sol[ex]
            @test_throws fixed(ex) sol(0.5, 0.3, 0.2; dv = ex)
            @test_throws fixed(ex) sol(0.5, Γ; dv = ex)
            @test_throws fixed(ex) sol[Dx(ex)]
            @test_throws fixed(ex) sol(0.5, 0.3, 0.2; dv = Dx(ex))
            @test_throws fixed(ex) normal_flux(sol, ex, Γ)
        end
        # A trace must keep time and replace the coordinates by one piece.
        @test_throws fixed(u(0, Γ)) sol[u(0, Γ)]
        @test sol[u(t, Γ)] == sol[U][:, piece_nodes(sol, Γ)]
        @test occursin("is not a key of this solution", errmsg(() -> sol[NormalDerivative(Γ)(u(0, Γ))]))
        trace = ArgumentError("dv = $(u(t, Γ)) is a trace; evaluate a trace as sol(t, Γ; dv = u(t, x, y))")
        @test_throws trace sol(0.5, 0.3, 0.2; dv = u(t, Γ))
        @test_throws trace sol(0.5, Γ; dv = u(t, Γ))
        @test occursin("is not a dependent variable of this solution", errmsg(() -> normal_flux(sol, u(t, Γ), Γ)))
    end

    @testset "two fields" begin
        # u_t = Δu + v with the algebraic row 0 = Δv; u* = (1 + x + 2y)(1 + t), v* = 1 + x + 2y.
        parabolic = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + v(t, x, y)
        algebraic = 0 ~ Dx(Dx(v(t, x, y))) + Dy(Dy(v(t, x, y)))
        bcs = [heat_bcs(Ω); [v(t, Γ) ~ 1 + x + 2y for Γ in pieces(Ω)]]
        sol2, _ = fem_solve(
            [parabolic, algebraic], bcs, Ω, (x, y), t, [u(t, x, y), v(t, x, y)]; pairing = Dict(algebraic => v)
        )
        V = v(t, x, y)
        @test sol2[U * V + V] ≈ sol2[U] .* sol2[V] .+ sol2[V] atol = 1.0e-12
        @test maximum(abs, sol2[U - V] .- sol2.t .* (1 .+ X .+ 2 .* Y)) <= 1.0e-6
        @test sol2(0.5, 0.3, 0.2; dv = U - V) ≈ 0.5 * 1.7 atol = 1.0e-6
        @test sol2(0.5, 0.3, 0.2; dv = U / V) ≈ sol2(0.5, 0.3, 0.2; dv = U) / sol2(0.5, 0.3, 0.2; dv = V) atol = 1.0e-12
        dvs2 = "u(t, x, y), v(t, x, y)"
        @test_throws observed_error(U * z(t, x, y), "z(t, x, y) is not a dependent variable, an independent variable or a parameter; the dependent variables are $dvs2 and the independent variables t, x, y") sol2[U * z(t, x, y)]
    end

    @testset "time-last fields" begin
        UL = u(x, y, t)
        eq = Dt(UL) ~ Dx(Dx(UL)) + Dy(Dy(UL)) + (1 + x + 2y)
        bcs = [[u(Γ, t) ~ (1 + x + 2y) * (1 + t) for Γ in pieces(Ω)]; u(x, y, 0) ~ 1 + x + 2y]
        soll, _ = fem_solve([eq], bcs, Ω, (x, y), t, [UL]; ps = [k => 2.0])
        Tl, Xc, Yc = permutedims(soll.t), soll[x], soll[y]
        exact = uexact.(Tl, Xc, Yc)
        # Arrays keep the orientation of sol[u(x, y, t)], nvertices × nt.
        @test size(soll[UL^2 + k * x]) == (nv, length(soll.t))
        @test soll[UL^2 + k * x] ≈ soll[UL] .^ 2 .+ 2 .* Xc atol = 1.0e-12
        @test maximum(abs, soll[UL^2 + k * x] .- (exact .^ 2 .+ 2 .* Xc)) <= 1.0e-6
        @test soll[t * UL] ≈ Tl .* soll[UL] atol = 1.0e-12
        @test soll[x + y] == repeat(Xc .+ Yc, 1, length(soll.t))
        @test soll[Dx(UL) + UL] ≈ soll[Dx(UL)] .+ soll[UL] atol = 1.0e-12
        # Calls take the time last.
        @test soll(0.3, 0.2, 0.5; dv = UL^2) ≈ uexact(0.5, 0.3, 0.2)^2 atol = 1.0e-6
        xs, ys, ts = range(0, 1, length = 5), range(0, 1, length = 4), [0.2, 0.5, 0.9]
        vals = soll(xs, ys, ts; dv = t * UL)
        @test size(vals) == (5, 4, 3)
        @test vals ≈ [ts[n] * soll(xs[i], ys[j], ts[n]; dv = UL) for i in 1:5, j in 1:4, n in 1:3] atol = 1.0e-12
        @test maximum(abs, vals .- [ts[n] * uexact(ts[n], xs[i], ys[j]) for i in 1:5, j in 1:4, n in 1:3]) <= 1.0e-6
        Γ = first(pieces(Ω))
        nodes = piece_nodes(soll, Γ)
        @test size(soll(Γ, ts; dv = UL + y)) == (length(nodes), 3)
        @test soll(Γ, ts; dv = UL + y) ≈ soll(Γ, ts; dv = UL) .+ Yc[nodes] atol = 1.0e-12
        @test soll[u(Γ, t)] == soll[UL][nodes, :]
        # Fixed arguments and malformed traces in time-last form.
        fixed(ex) = ArgumentError("$ex fixes arguments of u(x, y, t); write the dependent variable as u(x, y, t)")
        for ex in (u(0, y, t), u(x, y, 0))
            @test_throws fixed(ex) soll[ex]
            @test_throws fixed(ex) soll(0.3, 0.2, 0.5; dv = ex)
            @test_throws fixed(ex) soll[Dx(ex)]
            @test_throws fixed(ex) normal_flux(soll, ex, Γ)
            @test_throws observed_error(ex + 1, "$ex fixes arguments of u(x, y, t); write the dependent variable as u(x, y, t)") soll[ex + 1]
        end
        @test_throws fixed(u(Γ, 0)) soll[u(Γ, 0)]
        @test_throws fixed(u(t, Γ)) soll[u(t, Γ)]
        @test_throws ArgumentError("dv = $(u(Γ, t)) is a trace; evaluate a trace as sol(Γ, t; dv = u(x, y, t))") soll(0.3, 0.2, 0.5; dv = u(Γ, t))
    end

    @testset "fields with different time positions" begin
        parabolic = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + v(x, y, t)
        algebraic = 0 ~ Dx(Dx(v(x, y, t))) + Dy(Dy(v(x, y, t)))
        bcs = [heat_bcs(Ω); [v(Γ, t) ~ 1 + x + 2y for Γ in pieces(Ω)]]
        solm, _ = fem_solve(
            [parabolic, algebraic], bcs, Ω, (x, y), t, [u(t, x, y), v(x, y, t)]; pairing = Dict(algebraic => v)
        )
        VL = v(x, y, t)
        # One field per expression: its own orientation.
        @test solm[2U] == 2 .* solm[U]
        @test solm[VL * x] ≈ solm[VL] .* solm[x] atol = 1.0e-12
        @test solm(0.3, 0.2, 0.5; dv = 2VL) ≈ 2 * (1 + 0.3 + 0.4) atol = 1.0e-6
        # Mixed time positions have no orientation, documented as an error.
        mixed = "its dependent variables (all those of the system when it names none) put time in different argument positions; evaluate the time-first and the time-last parts separately"
        @test_throws observed_error(U + VL, mixed) solm[U + VL]
        @test_throws observed_error(U + VL, mixed) solm(0.5, 0.3, 0.2; dv = U + VL)
        @test_throws observed_error(2x, mixed) solm[2x]
    end

    @testset "evaluation is batched at 10^4 vertices" begin
        big = MeshDomain(rectangle_grid(Triangle, (99, 99)))
        bsol, _ = fem_solve([heat_eq], heat_bcs(big), big, (x, y), t, [u(t, x, y)]; ps = [k => 2.0], saveat = 0.5)
        @test size(bsol[U]) == (3, 10_000)
        E = U^2 + k * x * U + Dx(U)
        @test maximum(abs, bsol[E] .- (bsol[U] .^ 2 .+ 2 .* permutedims(bsol[x]) .* bsol[U] .+ bsol[Dx(U)])) <= 1.0e-10
        # Past the symbolic compile, which does not depend on the mesh, the cost is the projection
        # of Dx(U) and a few output-sized arrays; a substitution per vertex would allocate far more.
        bsol[E]
        bsol[Dx(U)]
        compile = @allocated FiniteElementMethod._compile_observed(bsol, E)
        @test @allocated(bsol[E]) < compile + @allocated(bsol[Dx(U)]) + 4 * sizeof(bsol[U])
        xs = range(0, 1, length = 100)
        F = U^2 + k * x * U
        bsol(bsol.t, xs, xs; dv = F)
        bsol(bsol.t, xs, xs; dv = U)
        compile = @allocated FiniteElementMethod._compile_observed(bsol, F)
        @test @allocated(bsol(bsol.t, xs, xs; dv = F)) < compile + 4 * @allocated(bsol(bsol.t, xs, xs; dv = U))
        @info "observed evaluation at 10^4 vertices" compile_bytes = compile vertex_bytes = @allocated(bsol[F]) seconds = @elapsed(bsol[F])
    end
end
