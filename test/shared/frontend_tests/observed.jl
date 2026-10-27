# End-to-end tests of observed expressions, `sol[expr]` and `sol(t, xs...; dv = expr)` (spec §7.8,
# O9): PDESystem -> DAEProblem -> solve, then each expression is checked against the same
# expression computed from the field arrays and against the exact solution.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, piece_nodes
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
        @test occursin("traces of derivatives are not supported", err)
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
        # A derivative of a dependent variable at fixed arguments is not a gradient key either.
        @test occursin("is not a key of this solution", errmsg(() -> sol[Dy(u(t, 0, y))]))
        @test occursin("is not a dependent variable or a first spatial derivative", errmsg(() -> sol(0.5, 0.3, 0.2; dv = Dy(u(t, 0, y)))))
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
