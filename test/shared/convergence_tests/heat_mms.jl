# Spec §10.3 (P1 rows, owned by M2): heat equation with a manufactured solution that is neither
# harmonic nor polynomial in space, with Dirichlet, Neumann and Robin pieces, on P1 triangles and
# quadrilaterals of the unit square and on the Gmsh L-shape. DFBDF at tight tolerances over a short
# time, so the spatial error dominates. Expected L2 and nodal rates 2; the test fails below 1.9.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using Ferrite: Ferrite, Triangle, Quadrilateral

include(joinpath(@__DIR__, "..", "frontend.jl"))
include(joinpath(@__DIR__, "..", "convergence.jl"))
include(joinpath(@__DIR__, "..", "lshape.jl"))

using ModelingToolkit: t_nounits as t
using Symbolics: Symbolics

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

ustar(t_, x_, y_) = exp(-t_) * sin(π * x_) * cos(π * y_ / 2) + x_^2 * y_
const U = ustar(t, x, y)
const Ux, Uy = expand_derivatives(Dx(U)), expand_derivatives(Dy(U))
# f = u*_t - Δu*
const SOURCE = expand_derivatives(Dt(U) - Dx(Dx(U)) - Dy(Dy(U)))
const HEAT = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + SOURCE
const IC = u(0, x, y) ~ ustar(0, x, y)
const TSPAN = (0.0, 0.1)
const MIN_RATE = 1.9

# Unit square: Neumann on the left, Robin on the right, Dirichlet on the bottom and top, so the two
# natural pieces share no corner (see the last testset).
function square_bcs(Ω)
    Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    return [
        NormalDerivative(Γl)(u(t, Γl)) ~ -Ux,
        NormalDerivative(Γr)(u(t, Γr)) + 2u(t, Γr) ~ Ux + 2U,
        u(t, Γb) ~ U, u(t, Γt) ~ U,
        IC,
    ]
end

# Neumann on the right and Robin on the top: the natural pieces meet at the corner (1, 1).
function corner_bcs(Ω)
    Γl, Γr, Γb, Γt = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    return [
        u(t, Γl) ~ U, u(t, Γb) ~ U,
        NormalDerivative(Γr)(u(t, Γr)) ~ Ux,
        NormalDerivative(Γt)(u(t, Γt)) + 2u(t, Γt) ~ Uy + 2U,
        IC,
    ]
end

# L-shape: Dirichlet on the inlet (x = 0) and the walls, Robin on the outlet (x = 2).
function lshape_bcs(Ω)
    Γi, Γw, Γo = boundarypiece.(Ω, (:inlet, :wall, :outlet))
    return [u(t, Γi) ~ U, u(t, Γw) ~ U, NormalDerivative(Γo)(u(t, Γo)) + 3u(t, Γo) ~ Ux + 3U, IC]
end

function convergence(grids, bcs)
    errs = map(grids) do grid
        Ω = MeshDomain(grid)
        sol, _ = fem_solve([HEAT], bcs(Ω), Ω, (x, y), t, [u(t, x, y)]; tspan = TSPAN, saveat = [TSPAN[2]])
        @test SciMLBase.successful_retcode(sol)
        # h ∝ 1 / sqrt(cells) on a quasi-uniform mesh of fixed area
        (; h = 1 / sqrt(Ferrite.getncells(grid)), nodal = nodal_error(sol, u(t, x, y), ustar), l2 = l2_error(sol, :u, ustar))
    end
    hs = [e.h for e in errs]
    nodal, l2 = [e.nodal for e in errs], [e.l2 for e in errs]
    return (; hs, nodal, l2, nodal_rates = rates(nodal, hs), l2_rates = rates(l2, hs))
end

fmt(v) = join(string.(round.(v; sigdigits = 3)), ", ")
report(name, r) = @info "Convergence: $name" h = fmt(r.hs) nodal = fmt(r.nodal) l2 = fmt(r.l2) nodal_rates = fmt(r.nodal_rates) l2_rates = fmt(r.l2_rates)

@testset verbose = true "P1 heat MMS convergence" begin
    @testset "$label" for (label, grids, bcs) in [
            ("P1 triangles, unit square", [rectangle_grid(Triangle, (n, n)) for n in (8, 16, 32, 64)], square_bcs),
            ("P1 quadrilaterals, unit square", [rectangle_grid(Quadrilateral, (n, n)) for n in (8, 16, 32)], square_bcs),
            ("P1 triangles, Gmsh L-shape", [lshape_grid(; h) for h in (0.2, 0.1, 0.05, 0.025)], lshape_bcs),
        ]
        r = convergence(grids, bcs)
        report(label, r)
        @test issorted(r.l2; rev = true)
        @test r.l2_rates[end] >= MIN_RATE
        @test r.nodal_rates[end] >= MIN_RATE
    end

    @testset "P1 triangles, natural pieces meeting at a corner" begin
        # The P1 maximum-norm error is O(h² log(1/h)) and the log factor is sharp: here it shows at
        # the corner (1, 1) of the right Neumann and top Robin pieces, where the nodal error is
        # largest and its rates are 1.64, 1.70, 1.74, matching h² log(1/h) (1.58, 1.68, 1.74). The
        # quadrilateral grid keeps nodal superconvergence there. L2 is unaffected.
        r = convergence([rectangle_grid(Triangle, (n, n)) for n in (8, 16, 32, 64)], corner_bcs)
        report("P1 triangles, natural corner", r)
        @test r.l2_rates[end] >= MIN_RATE
        log_rate = log(4 * log(32) / log(64)) / log(2)
        @test r.nodal_rates[end] >= log_rate - 0.05
    end
end
