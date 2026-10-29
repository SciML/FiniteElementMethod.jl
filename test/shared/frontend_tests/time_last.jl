# Time-last dependent variables u(x, y, t): the front end and the solution wrapper follow the
# declared argument order, as MethodOfLines does, so arrays put the time axis last and calls take
# the time last. Each problem is checked against a manufactured solution and, where it has one,
# against the same problem written time-first.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, piece_nodes, normal_flux
using Ferrite: Triangle, Vec

include(joinpath(@__DIR__, "..", "frontend.jl"))
include(joinpath(@__DIR__, "..", "lshape.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..) v(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const tol = 1.0e-8

pieces(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

"Largest nodal error of the time-last `dv` against `exact(t, x...)`; `sol[dv]` is `nvertices × nt`."
function nodal_error_last(sol, dv, exact)
    vals = sol[dv]
    xs = [sol[Num(x)] for x in sol.disc_data.x̄]
    return maximum(abs(vals[i, k] - exact(sol.t[k], getindex.(xs, i)...)) for i in axes(vals, 1), k in eachindex(sol.t))
end

# u* = (1 + x + 2y)(1 + t) with time-dependent Dirichlet and Neumann data; `U(a, b)` is
# u(t, a, b) or u(a, b, t), and `U(Γ)` the trace.
function heat_system(U, Ω, ic)
    Γl, Γr, Γb, Γt = pieces(Ω)
    eq = Dt(U(x, y)) ~ Dx(Dx(U(x, y))) + Dy(Dy(U(x, y))) + (1 + x + 2y)
    bcs = [
        U(Γl) ~ (1 + 2y) * (1 + t),
        NormalDerivative(Γr)(U(Γr)) ~ 1 + t,
        U(Γb) ~ (1 + x) * (1 + t),
        NormalDerivative(Γt)(U(Γt)) ~ 2 * (1 + t),
        ic,
    ]
    return [eq], bcs
end

@testset verbose = true "time last" begin
    @testset "the same heat problem written time-first and time-last" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        first_(a...) = u(t, a...)
        last_(a...) = u(a..., t)
        sol_f, _ = fem_solve(heat_system(first_, Ω, u(0, x, y) ~ 1 + x + 2y)..., Ω, (x, y), t, [u(t, x, y)]; saveat = 0.1)
        sol_l, _ = fem_solve(heat_system(last_, Ω, u(x, y, 0) ~ 1 + x + 2y)..., Ω, (x, y), t, [u(x, y, t)]; saveat = 0.1)
        uf, ul = u(t, x, y), u(x, y, t)
        nt, nv = length(sol_f.t), length(sol_f[x])
        @test sol_l.t == sol_f.t
        @test nodal_error(sol_f, uf, (t_, x_, y_) -> (1 + x_ + 2y_) * (1 + t_)) <= tol
        @test nodal_error_last(sol_l, ul, (t_, x_, y_) -> (1 + x_ + 2y_) * (1 + t_)) <= tol

        # Arrays: the time axis is last.
        @test size(sol_l[ul]) == (nv, nt)
        @test sol_l[ul] ≈ permutedims(sol_f[uf]) atol = 1.0e-12
        @test sol_l[ul, 3, :] == sol_l[ul][3, :]
        nΓ = length(piece_nodes(sol_l, Γl))
        @test size(sol_l[u(Γl, t)]) == (nΓ, nt)
        @test sol_l[u(Γl, t)] ≈ permutedims(sol_f[u(t, Γl)]) atol = 1.0e-12
        @test sol_l[Dx(ul)] ≈ permutedims(sol_f[Dx(uf)]) atol = 1.0e-10
        @test size(sol_l[Dx(ul)]) == (nv, nt)
        Dn = NormalDerivative(Γr)
        @test sol_l[Dn(u(Γr, t))] ≈ permutedims(sol_f[Dn(u(t, Γr))]) atol = 1.0e-10
        @test normal_flux(sol_l, ul, Γr) == sol_l[Dn(u(Γr, t))]
        @test size(normal_flux(sol_l, ul, Γr), 2) == nt

        # Calls: the time argument is last.
        @test sol_l(0.3, 0.4, 0.55; dv = ul) ≈ sol_f(0.55, 0.3, 0.4; dv = uf) atol = 1.0e-12
        @test sol_l(0.3, 0.4, 0.55; dv = ul) ≈ (1 + 0.3 + 0.8) * 1.55 atol = 1.0e-6
        @test sol_l(0.3, 0.4, 0.55) ≈ [sol_l(0.3, 0.4, 0.55; dv = ul)]
        xs, ys, ts = range(0, 1, length = 5), range(0, 1, length = 4), [0.2, 0.5, 0.9]
        @test size(sol_l(xs, ys, ts; dv = ul)) == (5, 4, 3)
        @test sol_l(xs, ys, ts; dv = ul) ≈ permutedims(sol_f(ts, xs, ys; dv = uf), (2, 3, 1)) atol = 1.0e-12
        @test sol_l(xs, 0.3, ts; dv = ul) ≈ permutedims(sol_f(ts, xs, 0.3; dv = uf)) atol = 1.0e-12
        @test size(sol_l(xs, ys, 0.5; dv = ul)) == (5, 4)
        @test sol_l(Γl, 0.5; dv = ul) ≈ sol_f(0.5, Γl; dv = uf) atol = 1.0e-12
        @test size(sol_l(Γl, ts; dv = ul)) == (nΓ, 3)
        @test sol_l(Γl, ts; dv = ul) ≈ permutedims(sol_f(ts, Γl; dv = uf)) atol = 1.0e-12
        pts = [Vec(0.2, 0.3), Vec(0.7, 0.1)]
        @test sol_l(pts, ts; dv = ul) ≈ permutedims(sol_f(ts, pts; dv = uf)) atol = 1.0e-12
        @test sol_l(0.3, 0.4, 0.55; dv = Dx(ul)) ≈ sol_f(0.55, 0.3, 0.4; dv = Dx(uf)) atol = 1.0e-12
        @test_throws ArgumentError("the last argument must be a time or a vector of times, as u(x, y, t) declares; got $(repr(Γl))") sol_l(
            0.5, Γl; dv = ul
        )
        @test_throws ArgumentError("the first argument must be a time or a vector of times, as u(t, x, y) declares; got $(repr(Γl))") sol_f(
            Γl, 0.5; dv = uf
        )
        @test_throws ArgumentError("the last argument must be a time or a vector of times, as u(x, y, t) declares; got nothing") sol_l(; dv = ul)
    end

    @testset "manufactured solution with time-dependent source, Dirichlet and Neumann data" begin
        # u* = (1 + x + 2y)(1 + t²) needs the source 2t(1 + x + 2y), reassembled at every t.
        u_exact(t_, x_, y_) = (1 + x_ + 2y_) * (1 + t_^2)
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = pieces(Ω)
        U = u(x, y, t)
        eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + 2t * (1 + x + 2y)
        bcs = [
            u(Γl, t) ~ (1 + 2y) * (1 + t^2),
            u(Γb, t) ~ (1 + x) * (1 + t^2),
            NormalDerivative(Γr)(u(Γr, t)) ~ 1 + t^2,
            NormalDerivative(Γt)(u(Γt, t)) ~ 2 * (1 + t^2),
            u(x, y, 0) ~ 1 + x + 2y,
        ]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [U])
        @test SciMLBase.successful_retcode(sol)
        @test nodal_error_last(sol, U, u_exact) <= 1.0e-6
        @test sol[u(Γl, t)] ≈ [u_exact(τ, 0.0, sol[y][i]) for i in piece_nodes(sol, Γl), τ in sol.t] atol = 1.0e-10
    end

    @testset "Gmsh L-shape" begin
        u_exact(t_, x_, y_) = (1 + x_) * (1 + t_)
        Ω = MeshDomain(lshape_grid(; h = 0.5))
        Γi, Γw, Γo = boundarypiece.(Ω, (:inlet, :wall, :outlet))
        U = u(x, y, t)
        eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + (1 + x)
        bcs = [u(Γi, t) ~ 1 + t, u(Γw, t) ~ (1 + x) * (1 + t), NormalDerivative(Γo)(u(Γo, t)) ~ 1 + t, u(x, y, 0) ~ 1 + x]
        sol, _ = fem_solve([eq], bcs, Ω, (x, y), t, [U])
        @test nodal_error_last(sol, U, u_exact) <= tol
        @test size(sol[U]) == (length(sol[x]), length(sol.t))
        @test sol[u(Γi, t)] ≈ repeat(permutedims(1 .+ sol.t), length(piece_nodes(sol, Γi))) atol = tol
        @test isnan(sol(1.5, 1.5, 0.5; dv = U))
        @test sol(0.5, 1.5, [0.0, 1.0]; dv = U) ≈ u_exact.([0.0, 1.0], 0.5, 1.5) atol = tol
    end

    @testset "fields with different time positions" begin
        # u_t = Δu + v with the algebraic row 0 = Δv; u(t, x, y) is time-first, v(x, y, t) time-last.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        parabolic = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + v(x, y, t)
        algebraic = 0 ~ Dx(Dx(v(x, y, t))) + Dy(Dy(v(x, y, t)))
        bcs = [
            [u(t, Γ) ~ (1 + x + 2y) * (1 + t) for Γ in pieces(Ω)];
            [v(Γ, t) ~ 1 + x + 2y for Γ in pieces(Ω)];
            u(0, x, y) ~ 1 + x + 2y
        ]
        sol, _ = fem_solve(
            [parabolic, algebraic], bcs, Ω, (x, y), t, [u(t, x, y), v(x, y, t)]; pairing = Dict(algebraic => v)
        )
        @test nodal_error(sol, u(t, x, y), (t_, x_, y_) -> (1 + x_ + 2y_) * (1 + t_)) <= tol
        @test nodal_error_last(sol, v(x, y, t), (t_, x_, y_) -> 1 + x_ + 2y_) <= tol
        nt, nv = length(sol.t), length(sol[x])
        @test size(sol[u(t, x, y)]) == (nt, nv) && size(sol[v(x, y, t)]) == (nv, nt)
        @test sol(0.3, 0.4, 0.5; dv = v(x, y, t)) ≈ 1 + 0.3 + 0.8 atol = tol
        @test sol(0.5, 0.3, 0.4; dv = u(t, x, y)) ≈ (1 + 0.3 + 0.8) * 1.5 atol = tol
        @test_throws ArgumentError(
            "the dependent variables $(sol.dvs) put time in different argument positions; pass dv to choose one"
        ) sol(0.5, 0.3, 0.4)
    end
end
