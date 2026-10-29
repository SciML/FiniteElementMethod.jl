# End to end on a curved domain (spec §6.2, §10.3 for P2, M4): a second-order Gmsh mesh of an
# annulus through the PDESystem front end. The space, the assembly (facet normals included) and
# the solution interface map cells with the mesh's own geometric interpolation, so P2 on
# quadratic cells keeps the L2 rate 3, while the same vertices with straight-sided cells cap it
# at 2: the Dirichlet data live on the true circle and the Robin data use its true normal.
using Test
using LinearAlgebra: norm, dot
using FiniteElementMethod: FiniteElementMethod, piece_nodes, normal_flux
using Ferrite: Ferrite, RefTriangle

include(joinpath(@__DIR__, "..", "frontend.jl"))
include(joinpath(@__DIR__, "..", "annulus.jl"))

using ModelingToolkit: t_nounits as t

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const r_in, r_out = 0.5, 1.0

# u* = (1 + t)φ is linear in t, which BDF integrates exactly, so the error is the spatial one.
φ = sin(2x) * cos(y) + x^2 * y
u_star = (1 + t) * φ
heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) +
    expand_derivatives(Dt(u_star) - Dx(Dx(u_star)) - Dy(Dy(u_star)))
ρ = sqrt(x^2 + y^2)
# The outer Dirichlet data are u* at the radial projection onto |x| = r_out: equal to u* only on
# the true circle. The inner Robin condition uses the true outward normal −(x, y)/ρ.
outer_data = substitute(u_star, Dict(x => r_out * x / ρ, y => r_out * y / ρ))
inner_data = expand_derivatives(-(x * Dx(u_star) + y * Dy(u_star)) / ρ) + u_star
const φ_f = Symbolics.build_function(φ, x, y; expression = Val(false))
u_exact(t_, x_, y_) = (1 + t_) * φ_f(x_, y_)
const ∇φ_f = [Symbolics.build_function(g, x, y; expression = Val(false)) for g in Symbolics.gradient(φ, [x, y])]
# Coordinates of the vertices of facet `fi`.
facet_ends(grid, fi) = [Ferrite.get_node_coordinate(grid, n) for n in Ferrite.facets(Ferrite.getcells(grid, fi[1]))[fi[2]]]

const P2 = Lagrange{RefTriangle, 2}()

function annulus_solve(h; order)
    Ω = MeshDomain(annulus_grid(h; order, r = r_in, R = r_out))
    Γi, Γo = boundarypiece.(Ω, (:inner, :outer))
    bcs = [
        u(t, Γo) ~ outer_data,
        NormalDerivative(Γi)(u(t, Γi)) + u(t, Γi) ~ inner_data,
        u(0, x, y) ~ φ,
    ]
    sol, _ = fem_solve([heat], bcs, Ω, (x, y), t, [u(t, x, y)]; spaces = Dict(u => P2))
    return (; sol, Ω, Γi, Γo, grid = Ω.mesh)
end

# L2 error at the final time over the discrete domain, with the geometry of the space.
function l2_error(sol)
    space = sol.disc_data.space
    ip_geo = Ferrite.geometric_interpolation(Ferrite.getcelltype(space.grid))
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{RefTriangle}(8), P2, ip_geo)
    U, tk = sol.original_sol.u[end], sol.t[end]
    err = 0.0
    for cell in Ferrite.CellIterator(space.dh)
        Ferrite.reinit!(cv, cell)
        xs, u_cell = Ferrite.getcoordinates(cell), U[Ferrite.celldofs(cell)]
        for q in 1:Ferrite.getnquadpoints(cv)
            X = Ferrite.spatial_coordinate(cv, q, xs)
            err += (Ferrite.function_value(cv, q, u_cell) - u_exact(tk, X...))^2 * Ferrite.getdetJdV(cv, q)
        end
    end
    return sqrt(err)
end

# Least-squares slope of log(error) against log(h).
function slope(hs, errs)
    X, Y = log.(hs), log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

const SIZES = (0.2, 0.1, 0.05, 0.025)

@testset verbose = true "curved Gmsh domain" begin
    runs = Dict(order => [annulus_solve(h; order) for h in SIZES] for order in (2, 1))

    @testset "P2 converges at rate 3 on quadratic geometry and 2 on straight-sided cells" begin
        @test all(run -> SciMLBase.successful_retcode(run.sol), vcat(values(runs)...))
        @test all(run -> Ferrite.getcelltype(run.grid) === Ferrite.QuadraticTriangle, runs[2])
        @test all(run -> Ferrite.getcelltype(run.grid) === Ferrite.Triangle, runs[1])
        # Gmsh meshes are not nested; h is the mean cell size.
        hs = [sqrt(π * (r_out^2 - r_in^2) / Ferrite.getncells(run.grid)) for run in runs[2]]
        curved, straight = (map(run -> l2_error(run.sol), runs[order]) for order in (2, 1))
        rate(errs) = [log(errs[k] / errs[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(errs) - 1)]
        show3(v) = Tuple(round.(v; sigdigits = 3))
        @info "annulus, P2 through the front end" h = show3(hs) curved = show3(curved) rates_curved = show3(rate(curved)) slope(hs, curved) straight = show3(straight) rates_straight = show3(rate(straight)) slope(hs, straight)
        @test slope(hs, curved) >= 2.8
        @test all(>=(2.6), rate(curved))
        @test slope(hs, straight) <= 2.3
        @test all(<=(2.4), rate(straight))
        @test curved[end] < straight[end] / 20
    end

    curved, straight = runs[2][2], runs[1][2]

    @testset "space and facet values use the mesh's geometry" begin
        # Perimeter of the inner circle and the deviation of the facet normals from −X/|X|.
        function geometry(run)
            space = run.sol.disc_data.space
            fv = only(space.facetvalues)
            perimeter, normal = 0.0, 0.0
            for fi in Ferrite.getfacetset(space.grid, "inner")
                xs = Ferrite.getcoordinates(space.grid, fi[1])
                Ferrite.reinit!(fv, xs, fi[2])
                for q in 1:Ferrite.getnquadpoints(fv)
                    X = Ferrite.spatial_coordinate(fv, q, xs)
                    perimeter += Ferrite.getdetJdV(fv, q)
                    normal = max(normal, norm(Ferrite.getnormal(fv, q) + X / norm(X)))
                end
            end
            return (; perimeter = abs(perimeter - 2π * r_in), normal)
        end
        c, s = geometry(curved), geometry(straight)
        @info "geometry errors at h = $(SIZES[2])" curved = c straight = s
        @test c.perimeter < s.perimeter / 100
        @test c.normal < s.normal / 100
    end

    @testset "solution interface on the curved mesh" begin
        (; sol, Γi, Γo, grid) = curved
        U = u(t, x, y)
        # Traces: every node of the outer piece, mid-side nodes included, is on the circle.
        nodes = piece_nodes(sol, Γo)
        X, Y = sol[x][nodes], sol[y][nodes]
        @test length(nodes) == 2 * length(Ferrite.getfacetset(grid, "outer"))
        @test maximum(abs, hypot.(X, Y) .- r_out) <= 1.0e-12
        trace = sol[u(t, Γo)]
        @test trace ≈ sol(sol.t, Γo; dv = U) rtol = 1.0e-12
        @test maximum(abs, trace .- [u_exact(tk, X[i], Y[i]) for tk in sol.t, i in eachindex(nodes)]) <= 1.0e-8
        @test nodal_error(sol, U, u_exact) <= 1.0e-3

        # Point evaluation inside, and between the vertices of an outer arc, where the chord of a
        # straight-sided cell leaves the domain.
        pts = [(0.7, 0.1), (-0.3, 0.6), (0.0, -0.9)]
        @test all(p -> abs(sol(1.0, p...; dv = U) - u_exact(1.0, p...)) <= 1.0e-3, pts)
        fi = first(sort!(collect(Ferrite.getfacetset(grid, "outer")); by = fi -> (fi[1], fi[2])))
        a, b = (atan(v[2], v[1]) for v in facet_ends(grid, fi))
        θ = a + 0.25 * rem2pi(b - a, RoundNearest)
        p = (r_out - 1.0e-9) .* (cos(θ), sin(θ))
        @test abs(sol(1.0, p...; dv = U) - u_exact(1.0, p...)) <= 1.0e-3
        @test isnan(straight.sol(1.0, p...; dv = U))

        # The facet-mean normal flux on the curved Robin piece converges to that of u*.
        @test sol[NormalDerivative(Γi)(u(t, Γi))] == normal_flux(sol, U, Γi)
        function flux_error(run)
            (; sol, Γi, grid) = run
            fv = only(sol.disc_data.space.facetvalues)
            facets = sort!(collect(Ferrite.getfacetset(grid, "inner")); by = fi -> (fi[1], fi[2]))
            exact = map(facets) do fi
                xs = Ferrite.getcoordinates(grid, fi[1])
                Ferrite.reinit!(fv, xs, fi[2])
                qs = 1:Ferrite.getnquadpoints(fv)
                sum(qs) do q
                    X = Ferrite.spatial_coordinate(fv, q, xs)
                    (1 + sol.t[end]) * dot([g(X...) for g in ∇φ_f], Ferrite.getnormal(fv, q)) * Ferrite.getdetJdV(fv, q)
                end / sum(q -> Ferrite.getdetJdV(fv, q), qs)
            end
            return maximum(abs, sol[NormalDerivative(Γi)(u(t, Γi))][end, :] - exact)
        end
        errs = [flux_error(runs[2][k]) for k in 2:3]
        @info "inner normal flux error at h = $(SIZES[2:3])" errs
        @test errs[2] < errs[1] / 3
        @test errs[2] <= 5.0e-3
    end
end
