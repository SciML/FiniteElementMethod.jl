# End to end, manufactured solutions solved by a linear solve: variable-coefficient diffusion in P1 and P2 on
# triangles, quadrilaterals and lines (L2 rate p + 1), and the non-divergence a(x,y)Δu row of spec §10.1
# (nodal rates ≥ 1.9 on the last two refinements; red cell measured 1.96, 2.01, 2.01).
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y
@variables u(..) u1(..)
Dx, Dy = Differential(x), Differential(y)
const P0 = [0.0]

@testset "variable-coefficient diffusion, $(celltype) P$(order)" for (celltype, order, nels) in (
        (Triangle, 1, (8, 16, 32)), (Quadrilateral, 1, (8, 16, 32)),
        (Triangle, 2, (4, 8, 16)), (Quadrilateral, 2, (4, 8, 16)),
        (Line, 1, (8, 16, 32)), (Line, 2, (8, 16, 32)),
    )
    if celltype === Line
        U, sivs = u1(t, x), [x]
        ustar, k = sin(pi * x) + x^2, 1 + x^2
        f = expand_derivatives(-Dx(k * Dx(ustar)))
        eq = 0 ~ Dx(k * Dx(U)) + f
        bcs = [BC("left", U ~ ustar), BC("right", Dx(U) ~ expand_derivatives(Dx(ustar)), [1.0])]
    else
        U, sivs = u(t, x, y), [x, y]
        ustar, k = sin(pi * x) * cos(pi * y) + x^2 * y, 1 + x^2 * y^2
        f = expand_derivatives(-(Dx(k * Dx(ustar)) + Dy(k * Dy(ustar))))
        eq = 0 ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + f
        bcs = [
            BC("left", U ~ ustar), BC("bottom", U ~ ustar), BC("top", U ~ ustar),
            BC("right", Dx(U) ~ expand_derivatives(Dx(ustar)), [1.0, 0.0]),
        ]
    end
    c = chain([eq], bcs; depvars = [U], t, sivs)
    exact = numeric(ustar, sivs)
    errs = map(nels) do nel
        setup = fe_setup(celltype, order, nel)
        l2_error(setup, linear_solve(setup, c.kernels, constraints(setup, c.dirichlet, P0), P0), exact)
    end
    @test minimum(rates(errs)) ≥ order + 1 - 0.15
end

@testset "non-divergence a(x,y)Δu, non-harmonic manufactured solution" begin
    U = u(t, x, y)
    ustar = sin(pi * x) * cos(pi * y) + x^2 * y
    axy = 1 + x^2 + y
    f = expand_derivatives(-axy * (Dx(Dx(ustar)) + Dy(Dy(ustar))))
    bcs = [
        BC("left", U ~ ustar), BC("bottom", U ~ ustar), BC("top", U ~ ustar),
        BC("right", Dx(U) ~ expand_derivatives(Dx(ustar)), [1.0, 0.0]),
    ]
    c = chain([0 ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) + f], bcs; depvars = [U], t, sivs = [x, y])
    exact = numeric(ustar, [x, y])
    errs = map((8, 16, 32, 64)) do nel
        setup = fe_setup(Triangle, 1, nel)
        nodal_error(setup, linear_solve(setup, c.kernels, constraints(setup, c.dirichlet, P0), P0), exact)
    end
    @test all(≥(1.9), rates(errs)[(end - 1):end])
end
