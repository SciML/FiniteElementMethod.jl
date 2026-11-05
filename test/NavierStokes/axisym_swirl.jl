# The axisymmetric swirl example (docs/src/axisymmetric_swirl.md): the r-weighted weak form on a scalar problem
# with a known exact solution, then the full Navier–Stokes system on graded Taylor–Hood P2/P1 triangles from Gmsh
# against the manufactured collapsing core, with the convergence rates of the Taylor–Hood pair.
using FiniteElementMethod, Test
using Ferrite: Triangle, Vec
include(joinpath(@__DIR__, "..", "..", "examples", "axisym_swirl", "axisym_swirl.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "axisym_swirl", "checksum.jl"))

rate(coarse, fine) = log2(coarse / fine)

@testset "Weighted(r): axisymmetric heat equation with a -u/r² term" begin
    @variables u(..)
    exact_heat = exp(-t) * r * exp(-r^2 - z^2)
    source = expand_derivatives(Dt(exact_heat) - (laplacian(exact_heat) - exact_heat / r^2))
    exact = Dict(:u => build_function(exact_heat, t, r, z; expression = Val(false)))
    errors = map((4, 8, 16)) do n
        Ω = MeshDomain(Ferrite.generate_grid(Triangle, (n, n), Vec(0.0, -1.0), Vec(2.0, 1.0)))
        sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
        U = u(t, r, z)
        eq = Dt(U) ~ laplacian(U) - U / r^2 + source
        bcs = [[u(t, Γ) ~ exact_heat for Γ in sides]; u(0, r, z) ~ substitute(exact_heat, t => 0)]
        @named heat = PDESystem([eq], bcs, [(r, z) ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, r, z], [U], [])
        disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()), ibp = Dict(eq => Weighted(r)))
        sol = solve(discretize(heat, disc), DFBDF(linsolve = UMFPACKFactorization()); abstol = 1.0e-11, reltol = 1.0e-11)
        @test SciMLBase.successful_retcode(sol)
        err, nrm = weighted_errors(sol, 0.5, exact)
        err[:u] / nrm[:u]
    end
    @info "Weighted(r) heat" errors rate.(errors[1:(end - 1)], errors[2:end])
    @test all(>(2.7), rate.(errors[1:(end - 1)], errors[2:end]))
end

@testset "axisymmetric swirl: collapsing core" begin
    # The manufactured field is divergence free and finite on the axis.
    divergence = build_function(FORCE[4], t, r, z; expression = Val(false))
    for point in ((0.3, 0.2, -0.4), (0.9, 0.05, 0.1), (0.99, 0.01, 0.02))
        @test abs(divergence(point...)) <= 1.0e-10
    end
    tau_end, report = 0.1, (0.5, 0.9)
    runs = [axisym_solve(; c, tau_end, report) for c in (0.2, 0.1)]
    @info "axisymmetric swirl" [(r.ndofs, r.solve_s, r.errors) for r in runs]
    for run in runs
        @test SciMLBase.successful_retcode(run.sol)
        @test run.sol.t[end] ≈ 1 - tau_end
    end
    for (k, tt) in enumerate(report)
        coarse, fine = runs[1].errors[k], runs[2].errors[k]
        @test rate(coarse[2], fine[2]) > 2.7    # velocity: Taylor–Hood order 3
        @test rate(coarse[3], fine[3]) > 1.7    # pressure: Taylor–Hood order 2
    end
end

@testset "axisymmetric swirl images are current" begin
    assets = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")
    # figures.jl needs CairoMakie, which is not a test dependency: it must at least parse.
    @test Meta.parseall(read(joinpath(@__DIR__, "..", "..", "examples", "axisym_swirl", "figures.jl"), String)) isa Expr
    for file in ("axisym_mesh.png", "axisym_core.png", "axisym_swirl.fnv1a")
        @test isfile(joinpath(assets, file))
    end
    if figure_sources_hash() != strip(read(joinpath(assets, "axisym_swirl.fnv1a"), String))
        error(
            "the sources of the axisymmetric swirl figures changed; regenerate docs/src/assets/axisym_mesh.png, " *
                "axisym_core.png and axisym_swirl.fnv1a with `julia --project=docs examples/axisym_swirl/figures.jl`"
        )
    end
end
