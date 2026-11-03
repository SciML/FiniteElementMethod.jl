# The data API for callers that build their own system around the discretization: `facet_dofs`, `dof_coordinates` and
# `cell_dofs` read the index data of a problem's `FEMMetadata` (its `problem_type`), and `FiniteElementMethod.operators`
# returns the assembled mass and stiffness matrices and the load vector as plain data.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using Ferrite: generate_grid, Triangle, Vec
using LinearAlgebra: norm
using ModelingToolkit: t_nounits as t

@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

# The PDESystem u_t = u_xx + u_yy + 4 on a 4 x 3 mesh of the unit square with Dirichlet data `g` on all four sides.
function heat_system(g)
    Ω = MeshDomain(generate_grid(Triangle, (4, 3), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + 4], [[u(t, Γ) ~ g for Γ in sides]; u(0, x, y) ~ 0],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
    )
    return pdesys, Ω, sides
end

# PDE:    u_t = u_xx + u_yy + 4
# Domain: the unit square, 4 x 3 triangle pairs, P1 and P2
# BCs:    u = 0 on all sides
# Shows:  `dof_coordinates` places each dof, `facet_dofs` lists the dofs of a boundary piece (by piece, name or field
#         expression) and `cell_dofs` gives the dofs of every cell, one column per cell.
@testset "facet_dofs, dof_coordinates and cell_dofs describe where the dofs of a field sit" for (order, nleft, ntop, nnodes) in ((1, 4, 5, 20), (2, 7, 9, 63))
    pdesys, Ω, (left, _, _, top) = heat_system(0)
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, order}()))
    md = discretize(pdesys, disc).problem_type
    X = dof_coordinates(md)
    @test size(X) == (2, nnodes)
    dofs = facet_dofs(md, left, u)
    @test length(dofs) == nleft && issorted(dofs) && all(iszero, X[1, dofs])
    @test facet_dofs(md, "top", :u) == facet_dofs(md, top, u) == facet_dofs(md, top, u(t, x, y))
    @test length(facet_dofs(md, top, u)) == ntop && all(≈(1.0), X[2, facet_dofs(md, top, u)])
    cells = cell_dofs(md, u)
    @test size(cells) == (order == 1 ? 3 : 6, 4 * 3 * 2)
    @test sort!(unique(vec(cells))) == 1:nnodes
end

# PDE:    u_t = u_xx + u_yy + 4, solved at its steady state -(u_xx + u_yy) = 4 by hand from `operators`
# Domain: the unit square, 4 x 3 triangle pairs, P2
# BCs:    u = x (1 - x) + y (1 - y) on all sides
# Exact:  u = x (1 - x) + y (1 - y), in the P2 space
# Shows:  `operators` returns M, K and F with the Dirichlet rows left unconstrained; eliminating the prescribed dofs
#         and solving K u = F reproduces the exact solution at every dof.
@testset "operators gives the matrices and load to assemble a steady solve by hand" begin
    exact(x_, y_) = x_ * (1 - x_) + y_ * (1 - y_)
    pdesys, Ω, _ = heat_system(x * (1 - x) + y * (1 - y))
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => Lagrange{RefTriangle, 2}()))
    ops = FiniteElementMethod.operators(pdesys, disc)
    n = size(ops.K, 1)
    @test size(ops.M) == (n, n) && ops.fields == [:u => 1:n]
    @test norm(ops.K * ones(n)) < 1.0e-12 # constants are in the kernel of the stiffness matrix
    fixed, free = ops.prescribed, setdiff(1:n, ops.prescribed)
    U = zeros(n)
    U[fixed] = ops.dirichlet_values
    U[free] = ops.K[free, free] \ (ops.F[free] - ops.K[free, fixed] * ops.dirichlet_values)
    X = dof_coordinates(ops.metadata)
    @test U ≈ [exact(X[1, i], X[2, i]) for i in 1:n] atol = 1.0e-12
end
