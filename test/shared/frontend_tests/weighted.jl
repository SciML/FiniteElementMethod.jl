# `Weighted` is exported: the r-weighted weak form of the axisymmetric heat equation u_t = (1/r)(r u_r)_r + u_zz
# in (r, z), with the weight r of the volume element r dr dz, converges at the P2 rate.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, expand_derivatives, substitute

@testset verbose = true "Weighted" begin
    @testset "exported" begin
        @test isdefined(FiniteElementMethod, :Weighted)
        @test Base.isexported(FiniteElementMethod, :Weighted)
        @test Weighted(2) isa FiniteElementMethod.PDEBase.StrongForm.IBPPolicy
    end

    @testset "axisymmetric heat equation, ibp = Dict(eq => Weighted(r))" begin
        @variables r z u(..)
        Dt, Dr, Dz = Differential(t), Differential(r), Differential(z)
        # The solution is flat at the axis r = 0, where the weak form carries the natural condition.
        exact = exp(-t) * exp(-r^2 - z^2)
        source = expand_derivatives(Dt(exact) - (Dr(r * Dr(exact)) / r + Dz(Dz(exact))))
        errors = map((4, 8, 16)) do n
            grid = rectangle_grid(Triangle, (n, n); lo = (0.0, -1.0), hi = (1.0, 1.0))
            Ω = MeshDomain(grid)
            axis, outer, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
            U = u(t, r, z)
            eq = Dt(U) ~ Dr(r * Dr(U)) / r + Dz(Dz(U)) + source
            bcs = [
                [u(t, Γ) ~ exact for Γ in (outer, bottom, top)]
                u(0, r, z) ~ substitute(exact, t => 0)
            ]
            sol, _ = fem_solve(
                [eq], bcs, Ω, (r, z), t, [U]; tspan = (0.0, 0.5),
                spaces = Dict(u => Lagrange{RefTriangle, 2}()), ibp = Dict(eq => Weighted(r)), natural = (axis,),
                solve_kwargs = (; abstol = 1.0e-11, reltol = 1.0e-11),
            )
            @test SciMLBase.successful_retcode(sol)
            reference = [exp(-sol.t[end]) * exp(-ri^2 - zi^2) for (ri, zi) in zip(sol[r], sol[z])]
            maximum(abs, sol[U][end, :] .- reference) / maximum(abs, reference)
        end
        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
        @info "Weighted(r) axisymmetric heat" errors rates
        @test errors[end] < 2.0e-4
        @test all(>(2.7), rates)
    end
end
