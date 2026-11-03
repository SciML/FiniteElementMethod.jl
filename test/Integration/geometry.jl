# End to end, P2 on quadratic geometry (spec §6.2, §10.3 for P2, M4): the driver receives values built with the
# grid's geometric interpolation, so quadratic cells are mapped isoparametrically. On a curved domain this keeps
# the P2 L2 rate at 3, while the same vertices with straight-sided cells lose it to the polygonal boundary.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y k
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

@testset "straight-sided quadratic cells reproduce the affine operators" begin
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    c = chain([eq], [BC("right", Dx(U) ~ 1.5, RIGHT)]; depvars = [U], t, sivs = [x, y], params = [k])
    for (affine, quadratic) in ((Triangle, QuadraticTriangle), (Quadrilateral, QuadraticQuadrilateral))
        a = grid_setup(generate_grid(affine, (6, 6)), 2, ("right",))
        q = grid_setup(generate_grid(quadratic, (6, 6)), 2, ("right",))
        @test celldofs(q.dh, 7) == celldofs(a.dh, 7)
        Aa, Aq = assemble_linear(a.ctx, c.kernels, [2.0]), assemble_linear(q.ctx, c.kernels, [2.0])
        @test relerr(Aq.K, Aa.K) ≤ 1.0e-12
        @test relerr(Aq.M, Aa.M) ≤ 1.0e-12
        @test relerr(Aq.F, Aa.F) ≤ 1.0e-12
    end
end

@testset "P2 on a curved domain: quadratic geometry keeps the L2 rate" begin
    # −Δu = f on the quarter annulus 1 ≤ r ≤ 2, u* on x = 0, ∂ₙu* on y = 0 (outward normal −y), and on the arcs the
    # data the boundary carries: u* at the radial projection of (x, y) onto r = 1 or r = 2. At a node on the arc
    # this is u*; at a node on a chord of a straight-sided cell it is not, which is the geometric error.
    ustar = sin(pi * x / 2) * cos(pi * y / 3) + x^2 * y
    f = expand_derivatives(-(Dx(Dx(ustar)) + Dy(Dy(ustar))))
    on_arc(R) = substitute(ustar, Dict(x => R * x / sqrt(x^2 + y^2), y => R * y / sqrt(x^2 + y^2)))
    bcs = [
        BC("inner", U ~ on_arc(1)), BC("outer", U ~ on_arc(2)), BC("left", U ~ ustar),
        BC("bottom", -Dy(U) ~ -expand_derivatives(Dy(ustar)), [0.0, -1.0]),
    ]
    c = chain([0 ~ Dx(Dx(U)) + Dy(Dy(U)) + f], bcs; depvars = [U], t, sivs = [x, y], domain = [(0.0, 2.0), (0.0, 2.0)])
    @test any(term -> term.piece == "bottom", c.wf.facet)
    exact = numeric(ustar, [x, y])
    sizes = (2, 4, 8, 16)
    errors(quadratic) = map(sizes) do n
        setup = grid_setup(quarter_annulus(n; quadratic), 2, ("bottom",))
        l2_error(setup, linear_solve(setup, c.kernels, constraints(setup, c.dirichlet, P0), P0), exact)
    end
    curved, straight = errors(true), errors(false)
    @info "quarter annulus, P2" curved rates_curved = rates(curved) straight rates_straight = rates(straight)
    @test rates(curved)[end] ≥ 2.8
    @test curved[end] < straight[end] / 5
    @test rates(straight)[end] < 2.4
end
