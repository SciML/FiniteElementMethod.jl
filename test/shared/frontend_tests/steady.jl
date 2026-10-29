# End-to-end tests of steady problems: a PDESystem with no time variable -> NonlinearProblem ->
# solve -> PDENoTimeSolution, checked against manufactured solutions.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid, field_range
using Ferrite: Ferrite, Triangle
using NonlinearSolveFirstOrder: NewtonRaphson

include(joinpath(@__DIR__, "..", "frontend.jl"))
include(joinpath(@__DIR__, "..", "lshape.jl"))

@variables x y u(..) v(..)
const Dx, Dy = Differential(x), Differential(y)

sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))

function steady_solve(eqs, bcs, Ω, dvs; disc_kwargs...)
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω], [x, y], dvs, [])
    prob = discretize(pdesys, FEMDiscretization(Ω; disc_kwargs...))
    return solve(prob, NewtonRaphson(); abstol = 1.0e-12), prob
end

"Largest error of `dv` against `exact(x, y)` at a lattice of points off the mesh nodes."
function point_error(sol, dv, exact)
    pts = range(0.0123, 0.9877; length = 41)
    return maximum(abs(sol(px, py; dv) - exact(px, py)) for px in pts, py in pts)
end

ustar(x_, y_) = sin(pi * x_) * exp(y_)

# Δu = Δu* on the unit square, u = u* on every side.
function poisson_error(n, order)
    Ω = MeshDomain(rectangle_grid(Triangle, (n, n)))
    eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ (1 - pi^2) * ustar(x, y)
    sol, _ = steady_solve(
        [eq], [u(Γ) ~ ustar(x, y) for Γ in sides(Ω)], Ω, [u(x, y)];
        spaces = Dict(u => Lagrange{RefTriangle, order}())
    )
    @test SciMLBase.successful_retcode(sol)
    @test sol.stats.nsteps == 1
    return point_error(sol, u(x, y), ustar)
end

@testset verbose = true "steady" begin
    @testset "Poisson on a rectangle grid converges at rate $(order + 1) for P$order" for order in (1, 2)
        e1, e2 = poisson_error.((8, 16), order)
        @test log2(e1 / e2) > order + 0.8
    end

    @testset "PDESystem without parameters" begin
        # Leaving out `ps` gives the system `NullParameters()`, which means no parameters.
        # u* = 1 + x² + 2y² + xy has Δu* = 6 and is exact in P2.
        u_exact(x_, y_) = 1 + x_^2 + 2y_^2 + x_ * y_
        Ω = MeshDomain(rectangle_grid(Triangle, (3, 3)))
        bcs = [u(Γ) ~ u_exact(x, y) for Γ in sides(Ω)]
        @named pdesys = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 6], bcs, [(x, y) ∈ Ω], [x, y], [u(x, y)])
        @test pdesys.ps isa SciMLBase.NullParameters
        disc = FEMDiscretization(Ω; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
        sol = solve(discretize(pdesys, disc), NewtonRaphson(); abstol = 1.0e-12)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol[u(x, y)] - u_exact.(sol[x], sol[y])) <= 1.0e-10
    end

    @testset "L-shape with Dirichlet, Neumann and Robin pieces" begin
        # u* = 1 + x + 2y + xy + x², Δu* = 2, is exact in P2 on straight triangles. ∇u* = (1 + y + 2x, 2 + x).
        u_exact(x_, y_) = 1 + x_ + 2y_ + x_ * y_ + x_^2
        Ω = MeshDomain(lshape_grid(; h = 0.5))
        Γi, Γw, Γo = boundarypiece.(Ω, (:inlet, :wall, :outlet))
        # The wall is y = 0 (outward normal -y), the edges y = 1 for x > 1 and y = 2 (normal +y), and
        # x = 1 for 1 < y < 2 (normal +x); facet quadrature points lie strictly inside the edges.
        wall_flux = ifelse(y < 0.5, -(2 + x), ifelse(x > 1 + 1.0e-8, 2 + x, ifelse(y > 2 - 1.0e-8, 2 + x, 1 + 2x + y)))
        bcs = [
            u(Γi) ~ 1 + 2y,
            NormalDerivative(Γw)(u(Γw)) ~ wall_flux,
            NormalDerivative(Γo)(u(Γo)) + u(Γo) ~ (5 + y) + (7 + 4y),
        ]
        sol, _ = steady_solve(
            [Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 2], bcs, Ω, [u(x, y)];
            spaces = Dict(u => Lagrange{RefTriangle, 2}())
        )
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol[u(x, y)] - u_exact.(sol[x], sol[y])) <= 1.0e-10
        @test abs(sol(0.5, 1.5; dv = u(x, y)) - u_exact(0.5, 1.5)) <= 1.0e-10
    end

    @testset "two fields, paired $(how)" for how in ("by declared order", "explicitly")
        # Δu + v = 5 + x and Δv = 0 with u* = x² + y², v* = 1 + x, both exact in P2.
        Ω = MeshDomain(rectangle_grid(Triangle, (3, 3)))
        u_eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) + v(x, y) ~ 5 + x
        v_eq = Dx(Dx(v(x, y))) + Dy(Dy(v(x, y))) ~ 0
        bcs = [[u(Γ) ~ x^2 + y^2 for Γ in sides(Ω)]; [v(Γ) ~ 1 + x for Γ in sides(Ω)]]
        P2 = Lagrange{RefTriangle, 2}()
        kw = how == "explicitly" ? (; pairing = Dict(v_eq => v)) : (;)
        eqs = how == "explicitly" ? [v_eq, u_eq] : [u_eq, v_eq]
        sol, prob = steady_solve(eqs, bcs, Ω, [u(x, y), v(x, y)]; spaces = Dict(u => P2, v => P2), kw...)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs, sol[u(x, y)] - (sol[x] .^ 2 + sol[y] .^ 2)) <= 1.0e-10
        @test maximum(abs, sol[v(x, y)] - (1 .+ sol[x])) <= 1.0e-10
        @test length(sol(0.4, 0.7)) == 2
        @test [f.name for f in prob.problem_type.fields] == [:u, :v]
    end

    @testset "steady solution interface" begin
        # u* = 1 + x + 2y is exact in P1: harmonic, with u = u* on the left and right and
        # ∂u/∂n = ±2 on the bottom and top.
        Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
        Γl, Γr, Γb, Γt = sides(Ω)
        bcs = [u(Γl) ~ 1 + 2y, u(Γr) ~ 2 + 2y, NormalDerivative(Γb)(u(Γb)) ~ -2, NormalDerivative(Γt)(u(Γt)) ~ 2]
        @named pdesys = PDESystem([Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0], bcs, [(x, y) ∈ Ω], [x, y], [u(x, y)], [])
        prob = discretize(pdesys, FEMDiscretization(Ω; abstol = 1.0e-12))
        @test prob isa SciMLBase.NonlinearProblem
        @test prob.kwargs[:abstol] == 1.0e-12
        @test SciMLBase.NonlinearProblem(pdesys, FEMDiscretization(Ω)).u0 == prob.u0
        # The initial guess is zero except on the Dirichlet dofs, which hold the data.
        space = prob.problem_type.space
        pd = space.ch.prescribed_dofs
        @test all(iszero, prob.u0[setdiff(1:length(prob.u0), pd)])
        coords = FiniteElementMethod._dof_coordinates(space)
        @test prob.u0[pd] ≈ [1 + c[1] + 2c[2] for c in coords[pd]]
        @test sprint(show, prob.problem_type) == "FEMMetadata(fields = [u], ndofs = 25, hasTime = false)"

        sol = solve(prob, NewtonRaphson())
        U = u(x, y)
        @test sol isa SciMLBase.PDENoTimeSolution
        @test sol.disc_data isa FEMMetadata{Val(false)}
        nv = Ferrite.getnnodes(Ω.mesh)
        @test sol[U] isa Vector{Float64} && length(sol[U]) == nv
        @test sol[U] ≈ 1 .+ sol[x] .+ 2 .* sol[y] atol = 1.0e-12
        @test sol(0.3, 0.6; dv = U) ≈ 2.5 atol = 1.0e-12
        @test sol([0.3, 2.0], 0.6; dv = U)[1] ≈ 2.5 atol = 1.0e-12
        @test isnan(sol([0.3, 2.0], 0.6; dv = U)[2])
        @test sol[u(Γr)] ≈ 2 .+ 2 .* sol[y][piece_nodes(sol, Γr)] atol = 1.0e-12
        @test sol(Γl; dv = U) ≈ 1 .+ 2 .* sol[y][piece_nodes(sol, Γl)] atol = 1.0e-12
        @test sol[Dx(U)] ≈ ones(nv) atol = 1.0e-10
        @test sol(0.3, 0.6; dv = Dy(U)) ≈ 2 atol = 1.0e-10
        @test normal_flux(sol, U, Γt) ≈ fill(2.0, 4) atol = 1.0e-10
        @test normal_flux(sol, U, Γl) ≈ fill(-1.0, 4) atol = 1.0e-10
        mktempdir() do dir
            path = write_vtk(sol, joinpath(dir, "poisson"))
            @test path == joinpath(dir, "poisson.vtu")
            @test isfile(path) && occursin("<VTKFile", read(path, String))
        end
    end
end

@testset "steady dirichlet = :eliminate" begin
    Ω = MeshDomain(rectangle_grid(Triangle, (4, 4)))
    eq = Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ 0
    bcs = [u(Γ) ~ 1 + x + 2y for Γ in sides(Ω)]
    sol_a, prob_a = steady_solve([eq], bcs, Ω, [u(x, y)])
    sol_e, prob_e = steady_solve([eq], bcs, Ω, [u(x, y)]; dirichlet = :eliminate)
    @test length(prob_e.u0) < length(prob_a.u0)
    @test SciMLBase.successful_retcode(sol_e)
    @test sol_e[u(x, y)] ≈ sol_a[u(x, y)] atol = 1.0e-10
    @test sol_e[u(x, y)] ≈ 1 .+ sol_e[x] .+ 2 .* sol_e[y] atol = 1.0e-10
end
