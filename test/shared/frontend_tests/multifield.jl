# End-to-end multi-field tests through the PDESystem front end (spec §6.2, §6.3, §7.8, §11.1 M4): fields of
# unequal order against a manufactured solution with both fields' rates, Taylor–Hood Stokes, and a field on a
# region (SubDofHandler) that reads NaN outside it.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle, Quadrilateral, Vec, getfacetset, addfacetset!, addcellset!, getcellset

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics, expand_derivatives

@variables x y u(..) v(..) w(..) u1(..) u2(..) p(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
Δ(f) = Dx(Dx(f)) + Dy(Dy(f))
sides(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top))
const SHAPE = Dict(Triangle => RefTriangle, Quadrilateral => RefQuadrilateral)

# Root mean square error at a 40×40 point lattice at the final time, a stand-in for the L2 error.
function rms_error(sol, dv, exact)
    pts = [Vec(a, b) for a in range(0.01, 0.99, 40) for b in range(0.01, 0.99, 40)]
    tf = sol.t[end]
    vals = sol(tf, pts; dv)
    return sqrt(sum(abs2, vals .- [exact(tf, q[1], q[2]) for q in pts]) / length(pts))
end
# The unit square split at x = 1/2: cell set "left_half", and bottom and top split into _left and _right pieces.
const HALF_PIECES = (:left, :right, :bottom_left, :bottom_right, :top_left, :top_right)
function halves_grid(n)
    grid = rectangle_grid(Triangle, (n, n))
    addcellset!(grid, "left_half", X -> X[1] <= 0.5 + 1.0e-12)
    region = getcellset(grid, "left_half")
    for side in ("bottom", "top")
        set = getfacetset(grid, side)
        delete!(grid.facetsets, side)
        addfacetset!(grid, side * "_left", Set(fi for fi in set if fi[1] in region))
        addfacetset!(grid, side * "_right", Set(fi for fi in set if !(fi[1] in region)))
    end
    return grid
end

# The unit square with one bottom facet, at the corner (0, 0), split off "bottom" into the piece "corner".
function corner_grid(ct, n)
    grid = rectangle_grid(ct, (n, n))
    bottom = getfacetset(grid, "bottom")
    facet_nodes(fi) = Ferrite.facets(grid.cells[fi[1]])[fi[2]]
    corner = Set(fi for fi in bottom if all(nd -> grid.nodes[nd].x[1] < 1 / n + 1.0e-12, facet_nodes(fi)))
    delete!(grid.facetsets, "bottom")
    addfacetset!(grid, "bottom", setdiff(bottom, corner))
    addfacetset!(grid, "corner", corner)
    return grid
end

rates(errs) = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

@testset verbose = true "multi-field front end" begin
    @testset "unequal orders, $(ct)s: rates p + 1 for both fields" for ct in (Triangle, Quadrilateral)
        # u in P2/Q2, w in P1/Q1. w's equation reads u but not the reverse, so u keeps its own rate.
        U, W = u(t, x, y), w(t, x, y)
        u_exact(t_, x_, y_) = (1 + t_) * sin(pi * x_) * sin(pi * y_) + x_
        w_exact(t_, x_, y_) = (1 + t_) * cos(x_ + y_)
        f1 = sin(pi * x) * sin(pi * y) * (1 + 2pi^2 * (1 + t))
        f2 = cos(x + y) * (3 + 2t) - ((1 + t) * sin(pi * x) * sin(pi * y) + x)
        eqs = [Dt(U) ~ Δ(U) + f1, Dt(W) ~ Δ(W) + U + f2]
        shape = SHAPE[ct]
        errs = map((4, 8, 16)) do n
            Ω = MeshDomain(rectangle_grid(ct, (n, n)))
            bcs = [
                [u(t, Γ) ~ (1 + t) * sin(pi * x) * sin(pi * y) + x for Γ in sides(Ω)];
                [w(t, Γ) ~ (1 + t) * cos(x + y) for Γ in sides(Ω)];
                u(0, x, y) ~ sin(pi * x) * sin(pi * y) + x; w(0, x, y) ~ cos(x + y)
            ]
            sol, prob = fem_solve(
                eqs, bcs, Ω, (x, y), t, [U, W]; tspan = (0.0, 0.25),
                spaces = Dict(u => Lagrange{shape, 2}(), w => Lagrange{shape, 1}())
            )
            @test SciMLBase.successful_retcode(sol)
            space = sol.disc_data.space
            nu, nw = length.(FiniteElementMethod.field_range.(Ref(space), (:u, :w)))
            @test nw == (n + 1)^2 && nu > nw
            @test length(prob.u0) == nu + nw
            # Vertex values are uniform across orders: one column per mesh vertex for both fields.
            @test size(sol[U], 2) == size(sol[W], 2) == (n + 1)^2
            (rms_error(sol, U, u_exact), rms_error(sol, W, w_exact))
        end
        @test all(>(2.7), rates(first.(errs)))
        @test all(>(1.8), rates(last.(errs)))
    end

    @testset "Taylor–Hood Stokes, $(ct)s: velocity rate 3, pressure rate 2" for ct in (Triangle, Quadrilateral)
        # Stream function ψ = x²(1 − x)²y²(1 − y)², u = (ψ_y, −ψ_x), which vanishes on the boundary, and p = x³ − y³;
        # the source makes (u, p) steady. Marched from u = 0 to the steady state (the slowest Stokes mode decays like
        # e^{−52t}). The continuity rows are algebraic and the pressure is absent from them (index 2): the pressure
        # starts from its exact values, is pinned on the one-facet piece "corner", and is left out of the error
        # control through a large abstol, without which DFBDF fails at t ≈ 1e-10 on the 16 × 16 meshes.
        U1, U2, P = u1(t, x, y), u2(t, x, y), p(t, x, y)
        ψ = x^2 * (1 - x)^2 * y^2 * (1 - y)^2
        u1s, u2s, ps = expand_derivatives(Dy(ψ)), expand_derivatives(-Dx(ψ)), x^3 - y^3
        f1, f2 = expand_derivatives(-Δ(u1s) + Dx(ps)), expand_derivatives(-Δ(u2s) + Dy(ps))
        continuity = 0 ~ Dx(U1) + Dy(U2)
        eqs = [Dt(U1) ~ Δ(U1) - Dx(P) + f1, Dt(U2) ~ Δ(U2) - Dy(P) + f2, continuity]
        exact = map(e -> Symbolics.build_function(e, x, y; expression = Val(false)), (u1s, u2s, ps))
        shape = SHAPE[ct]
        errs = map((4, 8, 16)) do n
            Ω = MeshDomain(corner_grid(ct, n))
            Γs = boundarypiece.(Ω, (:left, :right, :bottom, :top, :corner))
            bcs = [
                [u1(t, Γ) ~ 0 for Γ in Γs]; [u2(t, Γ) ~ 0 for Γ in Γs]; p(t, Γs[end]) ~ x^3 - y^3;
                u1(0, x, y) ~ 0; u2(0, x, y) ~ 0; p(0, x, y) ~ x^3 - y^3
            ]
            @named stokes = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.5)], [t, x, y], [U1, U2, P], [])
            P2, P1 = Lagrange{shape, 2}(), Lagrange{shape, 1}()
            prob = discretize(stokes, FEMDiscretization(Ω, t; spaces = Dict(u1 => P2, u2 => P2, p => P1), pairing = Dict(continuity => p)))
            @test count(.!prob.differential_vars) > length(FiniteElementMethod.field_range(prob.problem_type.space, :p))
            abstol = [d ? 1.0e-8 : 1.0e8 for d in prob.differential_vars]
            sol = solve(prob, EXACT_ROWS; abstol, reltol = 1.0e-8)
            @test SciMLBase.successful_retcode(sol)
            [rms_error(sol, dv, (_, x_, y_) -> e(x_, y_)) for (dv, e) in zip((U1, U2, P), exact)]
        end
        @test all(>(2.7), rates(getindex.(errs, 1))) && all(>(2.7), rates(getindex.(errs, 2)))
        @test all(>(1.7), rates(getindex.(errs, 3)))
    end

    @testset "a field on a region reads NaN outside it" begin
        # u everywhere, w on the cells left of x = 1/2 only. w* = (1 + 2y)(1 + t) has zero flux through the
        # interface x = 1/2, where w gets the natural condition; both exact solutions are in P1.
        U, W = u(t, x, y), w(t, x, y)
        u_exact(t_, x_, y_) = (1 + x_ + 2y_) * (1 + t_)
        w_exact(t_, x_, y_) = (1 + 2y_) * (1 + t_)
        eqs = [Dt(U) ~ Δ(U) + (1 + x + 2y), Dt(W) ~ Δ(W) + U - (1 + x + 2y) * (1 + t) + (1 + 2y)]
        Ω = MeshDomain(halves_grid(4))
        Γ = Dict(n => boundarypiece(Ω, n) for n in HALF_PIECES)
        bcs = [
            [u(t, Γ[n]) ~ (1 + x + 2y) * (1 + t) for n in HALF_PIECES];
            [w(t, Γ[n]) ~ (1 + 2y) * (1 + t) for n in (:left, :bottom_left, :top_left)];
            u(0, x, y) ~ 1 + x + 2y; w(0, x, y) ~ 1 + 2y
        ]
        sol, prob = fem_solve(eqs, bcs, Ω, (x, y), t, [U, W]; regions = Dict(w => Region(Ω, :left_half)))
        @test SciMLBase.successful_retcode(sol)
        space = sol.disc_data.space
        @test length(space.dh.subdofhandlers) == 2
        @test length(FiniteElementMethod.field_range(space, :w)) == 15   # the 3 × 5 vertices with x ≤ 1/2
        @test nodal_error(sol, U, u_exact) <= 1.0e-8
        xs, ys = sol[x], sol[y]
        inside = findall(<=(0.5 + 1.0e-12), xs)
        outside = findall(>(0.5 + 1.0e-12), xs)
        wv = sol[W]
        @test all(isnan, wv[:, outside])
        @test maximum(abs(wv[k, i] - w_exact(sol.t[k], xs[i], ys[i])) for k in eachindex(sol.t), i in inside) <= 1.0e-8
        vals = @test_logs (:warn, r"could not be evaluated") sol(0.5, [0.25, 0.75], 0.5; dv = W)
        @test vals[1] ≈ w_exact(0.5, 0.25, 0.5) atol = 1.0e-8
        @test isnan(vals[2])
        grad = @test_logs sol(0.5, [0.25, 0.75], 0.5; dv = Dy(W))
        @test grad[1] ≈ 3.0 atol = 1.0e-8
        @test isnan(grad[2])
        dyw = sol[Dy(W)][end, :]
        @test all(isnan, dyw[outside]) && all(v -> isapprox(v, 2 * (1 + sol.t[end]); atol = 1.0e-8), dyw[inside])
        @test all(v -> abs(v) <= 1.0e-8, normal_flux(sol, W, Γ[:left])[end, :])
        @test all(isnan, normal_flux(sol, W, Γ[:bottom_right]))
        @test isfile(write_vtk(sol, joinpath(mktempdir(), "region")))
    end

    @testset "error paths" begin
        Ω = MeshDomain(halves_grid(2))
        Γ = Dict(n => boundarypiece(Ω, n) for n in HALF_PIECES)
        U, W = u(t, x, y), w(t, x, y)
        eqs = [Dt(U) ~ Δ(U), Dt(W) ~ Δ(W)]
        ics = [u(0, x, y) ~ 0, w(0, x, y) ~ 0]
        lower(bcs; kw...) = discretize(
            PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U, W], []; name = :sys),
            FEMDiscretization(Ω, t; kw...)
        )
        ubcs = [u(t, Γ[n]) ~ 0 for n in HALF_PIECES]
        @test_throws ArgumentError(
            "the condition on w on Γ_bottom_right has facets outside the region :left_half of w; " *
                "a field on a region takes conditions only on pieces whose facets belong to the region's cells"
        ) lower([ubcs; w(t, Γ[:bottom_right]) ~ 0; ics]; regions = Dict(w => Region(Ω, :left_half)))
        @test_throws ArgumentError(
            "the condition on w on Γ_right has facets outside the region :left_half of w; " *
                "a field on a region takes conditions only on pieces whose facets belong to the region's cells"
        ) lower([ubcs; NormalDerivative(Γ[:right])(w(t, Γ[:right])) ~ 1; ics]; regions = Dict(w => Region(Ω, :left_half)))
        @test_throws ArgumentError(
            "`regions` has keys that are not dependent variables of the system: v; the dependent variables are u(t, x, y), w(t, x, y)"
        ) lower([ubcs; ics]; regions = Dict(v => Region(Ω, :left_half)))
        other = MeshDomain(halves_grid(2))
        @test_throws ArgumentError(
            "the region of w must be a Region of the FEMDiscretization's MeshDomain, as Region(Ω, name); got $(repr(Region(other, :left_half)))"
        ) FEMDiscretization(Ω, t; regions = Dict(w => Region(other, :left_half)))
        @test_throws ArgumentError(
            "the region of w must be a Region of the FEMDiscretization's MeshDomain, as Region(Ω, name); got :left_half"
        ) FEMDiscretization(Ω, t; regions = Dict(w => :left_half))
        @test_throws ArgumentError(
            "the interpolation $(Lagrange{RefQuadrilateral, 2}()) is not on the reference shape $(RefTriangle) of the mesh cells"
        ) lower([ubcs; ics]; spaces = Dict(w => Lagrange{RefQuadrilateral, 2}()))
    end
end
