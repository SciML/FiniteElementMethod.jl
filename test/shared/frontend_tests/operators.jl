# The data API for callers that build their own system around the operators: `operators`,
# `facet_dofs`, `dof_coordinates` and `cell_dofs`.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, facet_dofs, dof_coordinates, cell_dofs, prescribed_dofs
using Ferrite: Quadrilateral, Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using SparseArrays: nnz

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))

function heat_system(Ω, bcs; eq = heat)
    @named pdesys = PDESystem([eq], [bcs; u(0, x, y) ~ x], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    return pdesys
end

@testset verbose = true "operators" begin
    @testset "Neumann heat on a rectangle, P1 quadrilaterals" begin
        Ω = MeshDomain(rectangle_grid(Quadrilateral, (4, 3); hi = (2.0, 1.0)))
        ops = @test_logs (:warn,) match_mode = :any FiniteElementMethod.operators(heat_system(Ω, Equation[]), FEMDiscretization(Ω, t))
        n = 5 * 4
        @test size(ops.M) == size(ops.K) == (n, n)
        @test ops.fields == [:u => 1:n]
        @test sum(ops.M) ≈ 2.0                      # ∫ 1 dΩ
        @test maximum(abs, ops.K * ones(n)) < 1.0e-12   # constants are in the kernel
        @test ops.M ≈ ops.M' && ops.K ≈ ops.K'
        @test all(ops.differential_vars)
        @test iszero(ops.F) && isempty(ops.prescribed)
        X = dof_coordinates(ops.metadata)
        @test ops.u0 ≈ X[1, :]
    end

    @testset "Dirichlet rows are left unconstrained" begin
        Ω = MeshDomain(rectangle_grid(Quadrilateral, (4, 3)))
        left, right = boundarypiece.(Ω, (:left, :right))
        pdesys = heat_system(Ω, [u(t, left) ~ 2.0, u(t, right) ~ 0.0])
        disc = FEMDiscretization(Ω, t)
        ops = FiniteElementMethod.operators(pdesys, disc)
        md = ops.metadata
        @test sort(ops.prescribed) == sort([facet_dofs(md, left, u); facet_dofs(md, right, u)])
        @test maximum(abs, ops.K * ones(size(ops.K, 1))) < 1.0e-12
        @test ops.u0[facet_dofs(md, left, u)] ≈ fill(2.0, 4)
        # Off the Dirichlet rows the operators are those of the DAEProblem.
        prob = DAEProblem(pdesys, disc)
        free = setdiff(1:size(ops.K, 1), ops.prescribed)
        res = zeros(length(prob.u0))
        du = randn(length(prob.u0))
        prob.f(res, du, prob.u0, prob.p, 0.0)
        @test res[free] ≈ (ops.M * du + ops.K * prob.u0 - ops.F)[free]
    end

    @testset "facet_dofs, dof_coordinates and cell_dofs" begin
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 3)))
        left, top = boundarypiece.(Ω, (:left, :top))
        for (order, nleft, ntop) in ((1, 4, 5), (2, 7, 9))
            ops = @test_logs (:warn,) match_mode = :any FiniteElementMethod.operators(
                heat_system(Ω, Equation[]), FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, order}()))
            )
            md = ops.metadata
            X = dof_coordinates(md)
            @test size(X) == (2, size(ops.K, 1))
            dl = facet_dofs(md, left, u)
            @test length(dl) == nleft && issorted(dl)
            @test all(iszero, X[1, dl])
            @test facet_dofs(md, "top", :u) == facet_dofs(md, top, u)
            @test length(facet_dofs(md, top, u(t, x, y))) == ntop
            @test all(≈(1.0), X[2, facet_dofs(md, :top, u)])
            C = cell_dofs(md, u)
            @test size(C) == (order == 1 ? 3 : 6, 4 * 3 * 2)
            @test sort!(unique(vec(C))) == 1:size(ops.K, 1)
        end
    end

    @testset "rejections" begin
        Ω = MeshDomain(rectangle_grid(Quadrilateral, (2, 2)))
        left = boundarypiece(Ω, :left)
        msg(f) = errmsg(f)
        nonlinear = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) - u(t, x, y)^3
        @test occursin("nonlinear", msg(() -> FiniteElementMethod.operators(heat_system(Ω, [u(t, left) ~ 0]; eq = nonlinear), FEMDiscretization(Ω, t))))
        @test occursin("constant in time", msg(() -> FiniteElementMethod.operators(heat_system(Ω, [u(t, left) ~ sin(t)]), FEMDiscretization(Ω, t))))
        sourced = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + t
        @test occursin("time-independent", msg(() -> FiniteElementMethod.operators(heat_system(Ω, [u(t, left) ~ 0]; eq = sourced), FEMDiscretization(Ω, t))))
        ops = FiniteElementMethod.operators(heat_system(Ω, [u(t, left) ~ 0]), FEMDiscretization(Ω, t))
        @test occursin("not a facet set", msg(() -> facet_dofs(ops.metadata, "nope", u)))
        @test occursin("not a field", msg(() -> facet_dofs(ops.metadata, left, :v)))
    end
end
