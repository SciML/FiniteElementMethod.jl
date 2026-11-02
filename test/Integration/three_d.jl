# End to end in 3D, PDESystem -> DAEProblem -> solve: the heat equation on the unit cube with the manufactured
# solution u* = exp(-t) sin(πx) sin(πy) sin(πz), homogeneous Dirichlet on all six faces, on hexahedra and
# tetrahedra in P1 and P2. The L2 error at the final time must converge at rate p + 1; the time integrator runs at
# tight tolerances so that its error stays far below the spatial error.
using FiniteElementMethod, Test
using FiniteElementMethod: rectangle_grid
using Ferrite: Ferrite, Hexahedron, Tetrahedron, RefHexahedron, RefTetrahedron, CellValues, CellIterator, QuadratureRule
using ModelingToolkit: t_nounits as t

include(joinpath(@__DIR__, "..", "shared", "frontend.jl"))

rates(errs) = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]

# L2 error of the dof vector `u` of `space` against `exact(X)`, with quadrature of order 2p + 2 (Ferrite's Keast
# rules on tetrahedra stop at order 5).
function cube_l2_error(space, u, shape, order, exact)
    cv = CellValues(QuadratureRule{shape}(min(2order + 2, 5)), Ferrite.Lagrange{shape, order}())
    err = 0.0
    for cell in CellIterator(space.dh)
        Ferrite.reinit!(cv, cell)
        u_cell = u[Ferrite.celldofs(cell)]
        xs = Ferrite.getcoordinates(cell)
        for q in 1:Ferrite.getnquadpoints(cv)
            err += (Ferrite.function_value(cv, q, u_cell) - exact(Ferrite.spatial_coordinate(cv, q, xs)))^2 * Ferrite.getdetJdV(cv, q)
        end
    end
    return sqrt(err)
end

# Solves the heat problem on an `nel`³ mesh and returns the L2 error at t = 1.
function cube_heat_error(celltype, shape, order, nel)
    @variables x y z
    @variables u(..)
    Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)
    U = u(t, x, y, z)
    s = sin(pi * x) * sin(pi * y) * sin(pi * z)
    Ω = MeshDomain(rectangle_grid(celltype, (nel, nel, nel)))
    eq = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + Dz(Dz(U)) + (3pi^2 - 1) * exp(-t) * s
    faces = boundarypiece.(Ω, (:x1_lo, :x1_hi, :x2_lo, :x2_hi, :x3_lo, :x3_hi))
    bcs = [[u(t, Γ) ~ 0 for Γ in faces]; u(0, x, y, z) ~ s]
    sol, _ = fem_solve(
        [eq], bcs, Ω, (x, y, z), t, [U];
        spaces = Dict(u => Ferrite.Lagrange{shape, order}()), saveat = [1.0],
        solve_kwargs = (; abstol = 1.0e-11, reltol = 1.0e-11)
    )
    @test SciMLBase.successful_retcode(sol)
    exact(X) = exp(-1.0) * sin(pi * X[1]) * sin(pi * X[2]) * sin(pi * X[3])
    return cube_l2_error(sol.disc_data.space, sol.original_sol.u[end], shape, order, exact)
end

@testset "3D heat on the unit cube, $(celltype) P$(order)" for (celltype, shape, order, sizes) in (
        (Hexahedron, RefHexahedron, 1, (4, 8, 16)), (Hexahedron, RefHexahedron, 2, (2, 4, 8)),
        (Tetrahedron, RefTetrahedron, 1, (4, 8, 16)), (Tetrahedron, RefTetrahedron, 2, (2, 4, 8)),
    )
    errs = [cube_heat_error(celltype, shape, order, nel) for nel in sizes]
    @info "3D heat $(celltype) P$(order): L2 errors $(errs), rates $(rates(errs))"
    @test minimum(rates(errs)) ≥ order + 1 - 0.3
end
