# `emit_dirichlet(analysis; expression = Val(true))` returns the generated `Expr`s, which build the function the
# default call compiles.
import FiniteElementMethod
using FiniteElementMethod: emit_dirichlet
using FiniteElementMethod.RuntimeGeneratedFunctions: RuntimeGeneratedFunction
using Ferrite: Vec
using Symbolics
using Test

include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse, BC

@variables t x y k
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

@testset "expression form of the Dirichlet data round-trips" begin
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U))
    # Constant, space-dependent and time-dependent prescribed values.
    bcs = [BC("left", U ~ 1), BC("right", U ~ 0.1 * y + x), BC("top", U ~ sin(t) * x)]
    an = analyse([eq], bcs; depvars = [U], t, sivs = [x, y], params = [k])
    compiled, generated = emit_dirichlet(an), emit_dirichlet(an; expression = Val(true))
    @test length(generated) == length(compiled) == 3
    P = [2.0]
    for ((piece, field, g), (epiece, efield, ex)) in zip(compiled, generated)
        @test (epiece, efield) == (piece, field)
        @test ex isa Expr
        evaluated = RuntimeGeneratedFunction(FiniteElementMethod, FiniteElementMethod, ex)
        for _ in 1:5
            X, T = Vec(rand(), rand()), rand()
            @test isequal(evaluated(X, T, P), g(X, T, P))
        end
    end
end
