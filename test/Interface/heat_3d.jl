# The 3D heat equation through the full interface on a hexahedron mesh and on a tetrahedron mesh, in
# P1 and P2 elements, compared with exact solutions. Meshes are kept to a few cells per side because
# 3D degrees of freedom grow fast; the cases still check every step of the workflow in three dimensions.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Tetrahedron, Hexahedron, Vec
using ModelingToolkit: t_nounits as t
include(joinpath(@__DIR__, "..", "shared", "interface_helpers.jl"))

@variables x y z u(..)
Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)
heat = Dt(u(t, x, y, z)) ~ Dx(Dx(u(t, x, y, z))) + Dy(Dy(u(t, x, y, z))) + Dz(Dz(u(t, x, y, z)))

unit_cube(celltype, n) = MeshDomain(generate_grid(celltype, (n, n, n), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0)))
faces(Ω) = boundarypiece.(Ω, (:left, :right, :bottom, :top, :front, :back))

# Solves `heat` on Ω with the given element space and boundary and initial conditions to `tend`.
function solve_heat_3d(Ω, space, bcs, tend)
    domains = [(x, y, z) ∈ Ω, t ∈ Interval(0.0, tend)]
    @named pdesys = PDESystem([heat], bcs, domains, [t, x, y, z], [u(t, x, y, z)], [])
    prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict(u => space)))
    return solve(prob, DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
end

# L2 error at `tend` of the decaying eigenmode sin(pi x) sin(pi y) sin(pi z) with zero Dirichlet data.
function eigenmode_error(celltype, space, n, tend)
    Ω = unit_cube(celltype, n)
    bcs = [[u(t, Γ) ~ 0 for Γ in faces(Ω)]; u(0, x, y, z) ~ sin(pi * x) * sin(pi * y) * sin(pi * z)]
    sol = solve_heat_3d(Ω, space, bcs, tend)
    exact(x_, y_, z_) = exp(-3pi^2 * tend) * sin(pi * x_) * sin(pi * y_) * sin(pi * z_)
    points = sample_points(11)
    return l2_error(sol, u(t, x, y, z), exact, tend, points, points, points)
end

# PDE:    u_t = u_xx + u_yy + u_zz
# Domain: the unit cube, t in (0, 0.02); 4 cells per side for P1, 2 for P2
# BCs:    u = 0 on all six faces (constant Dirichlet); IC u(0, x, y, z) = sin(pi x) sin(pi y) sin(pi z)
# Exact:  u = exp(-3 pi^2 t) sin(pi x) sin(pi y) sin(pi z)
# Shows:  the decaying eigenmode is reproduced by P1 and P2 elements on hexahedra and on tetrahedra.
@testset "the 3D heat equation with zero Dirichlet data decays as its eigenmode on $name meshes" for (name, celltype, ref, order, n, tol) in (
        ("P1 hexahedron", Hexahedron, RefHexahedron, 1, 4, 5.0e-2), ("P2 hexahedron", Hexahedron, RefHexahedron, 2, 2, 1.2e-2),
        ("P1 tetrahedron", Tetrahedron, RefTetrahedron, 1, 4, 8.0e-2), ("P2 tetrahedron", Tetrahedron, RefTetrahedron, 2, 2, 4.0e-2),
    )
    @test eigenmode_error(celltype, Lagrange{ref, order}(), n, 0.02) < tol
end

# PDE:    u_t = u_xx + u_yy + u_zz
# Domain: the unit cube, t in (0, 1), 2 cells per side
# BCs:    u = x^2 + y^2 + z^2 + 6t on all six faces (time-dependent Dirichlet); IC u(0, x, y, z) = x^2 + y^2 + z^2
# Exact:  u = x^2 + y^2 + z^2 + 6t, which lies in the P2 space, so the error is the solver tolerance
# Shows:  time-dependent Dirichlet data is imposed at every time in 3D, on both cell shapes.
@testset "time-dependent Dirichlet data is reproduced exactly by P2 elements on $name meshes" for (name, celltype, ref) in (
        ("hexahedron", Hexahedron, RefHexahedron), ("tetrahedron", Tetrahedron, RefTetrahedron),
    )
    Ω = unit_cube(celltype, 2)
    bcs = [[u(t, Γ) ~ x^2 + y^2 + z^2 + 6t for Γ in faces(Ω)]; u(0, x, y, z) ~ x^2 + y^2 + z^2]
    sol = solve_heat_3d(Ω, Lagrange{ref, 2}(), bcs, 1.0)
    points = sample_points(7)
    @test l2_error(sol, u(t, x, y, z), (x_, y_, z_) -> x_^2 + y_^2 + z_^2 + 6, 1.0, points, points, points) < 1.0e-7
end

# PDE:    u_t = u_xx + u_yy + u_zz
# Domain: the unit cube, t in (0, 0.02), meshes of 4, 8, 16 (P1 hexahedra) or 2, 3, 4 (P2 tetrahedra) cells per side
# BCs:    u = 0 on all six faces (constant Dirichlet); IC u(0, x, y, z) = sin(pi x) sin(pi y) sin(pi z)
# Exact:  u = exp(-3 pi^2 t) sin(pi x) sin(pi y) sin(pi z)
# Shows:  the optimal L2 rates order + 1 hold in 3D; the coarsest mesh is pre-asymptotic, so the finest pair is checked.
@testset "$name elements converge at rate $(order + 1) for the 3D heat equation" for (name, celltype, ref, order, ns) in (
        ("P1 hexahedron", Hexahedron, RefHexahedron, 1, (4, 8, 16)), ("P2 tetrahedron", Tetrahedron, RefTetrahedron, 2, (2, 3, 4)),
    )
    errors = [eigenmode_error(celltype, Lagrange{ref, order}(), n, 0.02) for n in ns]
    @test issorted(errors; rev = true)
    @test last(observed_rate(errors, 1 ./ ns)) > order + 0.85
end
