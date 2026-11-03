# `remake(prob; p)` on a problem from `discretize`: the operators are assembled at the PDESystem's parameter
# values, so solving a problem with other values raises an ArgumentError naming the parameter, for the steady
# `NonlinearProblem` as for the transient `DAEProblem`; keeping `p` (or remaking `u0`) still solves.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Triangle
using SciMLBase: SciMLBase, remake
using NonlinearSolveFirstOrder: NewtonRaphson

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t

@parameters D
@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

square() = MeshDomain(rectangle_grid(Triangle, (3, 3)))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

function steady_problem()
    Ω = square()
    eq = D * (Dx(Dx(u(x, y))) + Dy(Dy(u(x, y)))) ~ 6 * D
    bcs = [u(Γ) ~ 1 + x^2 + 2y^2 for Γ in sides(Ω)]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω], [x, y], [u(x, y)], [D => 2.0])
    return discretize(pdesys, FEMDiscretization(Ω))
end

function transient_problem()
    Ω = square()
    eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
    bcs = [[u(t, Γ) ~ 0 for Γ in sides(Ω)]; u(0, x, y) ~ x * (1 - x) * y * (1 - y)]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [u(t, x, y)], [D => 2.0])
    return discretize(pdesys, FEMDiscretization(Ω, t))
end

@testset verbose = true "remake(prob; p) is rejected, not ignored" for (kind, make) in
    (("steady", steady_problem), ("transient", transient_problem))
    prob = make()
    alg = kind == "steady" ? NewtonRaphson() : EXACT_ROWS
    @test SciMLBase.successful_retcode(solve(prob, alg))
    @testset "the same p still solves" begin
        @test SciMLBase.successful_retcode(solve(remake(prob; p = [D => 2.0]), alg))
        @test SciMLBase.successful_retcode(solve(remake(prob; u0 = copy(prob.u0)), alg))
    end
    @testset "a numeric p vector" begin
        err = try
            solve(remake(prob; p = [3.0]), alg)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("parameter D", sprint(showerror, err))
        @test occursin("discretize", sprint(showerror, err))
    end
    @testset "a symbolic p pair" begin
        err = try
            solve(remake(prob; p = [D => 3.0]), alg)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("parameter D", sprint(showerror, err))
    end
end
