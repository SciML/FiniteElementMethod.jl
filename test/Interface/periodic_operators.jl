# `operators` on a PDESystem with periodic conditions returns the periodic identification `P` (one column per class master
# and per untied dof) and the tied dofs, so that `P' M P`, `P' K P` and `P' F` are the system `discretize` condenses.
# Each case reduces the heat equation with `P`, advances it exactly with a matrix exponential and compares with `discretize`.
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF, Test
using FiniteElementMethod: operators
using Ferrite: generate_grid, Triangle, Quadrilateral, Hexahedron, Vec
using LinearAlgebra: eigvals, Symmetric, norm
using SparseArrays: sparse
using ModelingToolkit: t_nounits as t

@variables x y z u(..)
Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)

# The reduced solution at `tend`: the dofs of `P' M P dc/dt = -P' K P c + P' F` restricted to the columns `free` (all but
# the prescribed ones, whose value is zero here), from the projection of the initial dof vector. The affine system is
# advanced exactly through the exponential of the matrix [-A b; 0 0].
function reduced_solution(ops, tend)
    (; M, K, F, P, u0) = ops
    c0 = (P' * P) \ (P' * u0)
    free = setdiff(1:size(P, 2), ops.prescribed_columns)
    Mr = Matrix(P' * M * P)[free, free]
    A = Mr \ Matrix(P' * K * P)[free, free]
    b = Mr \ (P' * F)[free]
    nf = length(free)
    E = exp(tend * [-A b; zeros(1, nf + 1)]) * [c0[free]; 1.0]
    c = zeros(size(P, 2))
    c[free] = E[1:nf]
    return P * c
end

function discretize_solution(pdesys, disc, tend)
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-13, reltol = 1.0e-13)
    return sol.original_sol.u[end]
end

# Columns of `P` whose dof is prescribed (a master prefers a prescribed dof of its class).
with_columns(ops) = merge(ops, (; prescribed_columns = [argmax(Vector(ops.P[d, :])) for d in ops.prescribed]))

reldiff(a, b) = norm(a - b) / norm(b)

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), 16 x 16 triangle pairs, P1
# BCs:    periodic left-right and bottom-top; IC sin(2 pi (x + y))
# Shows:  `operators` accepts the periodic pairs and its reduced system reproduces the solution of `discretize`.
@testset "operators reduces a fully periodic box to the system discretize solves" begin
    Ω = MeshDomain(generate_grid(Triangle, (16, 16), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ sin(2π * (x + y))],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    disc = FEMDiscretization(Ω, t)
    ops = with_columns(operators(pdesys, disc))
    n = size(ops.M, 1)
    @test size(ops.P) == (n, n - length(ops.tied))
    @test length(ops.tied) == 17^2 - 16^2
    @test issorted(ops.tied)
    @test all(sum(ops.P; dims = 2) .== 1)
    @test size(ops.P, 2) == 16^2
    diff = reldiff(reduced_solution(ops, 0.05), discretize_solution(pdesys, disc, 0.05))
    println("periodic box relative difference: ", diff)
    @test diff < 1.0e-10
end

# PDE:    u_t = u_xx + u_yy + u_zz
# Domain: the unit cube, 4 x 4 x 4 hexahedra, Q1, t in (0, 0.05)
# BCs:    periodic in x, y and z; IC sin(2 pi x) cos(2 pi y) + sin(2 pi z)
# Shows:  the same agreement in 3D, where edges and corners belong to several pairs.
@testset "operators reduces a fully periodic cube to the system discretize solves" begin
    Ω = MeshDomain(generate_grid(Hexahedron, (4, 4, 4), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0)))
    left, right, front, back, bottom, top = boundarypiece.(Ω, (:left, :right, :front, :back, :bottom, :top))
    U = u(t, x, y, z)
    @named pdesys = PDESystem(
        [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + Dz(Dz(U))],
        [u(t, left) ~ u(t, right), u(t, front) ~ u(t, back), u(t, bottom) ~ u(t, top), u(0, x, y, z) ~ sin(2π * x) * cos(2π * y) + sin(2π * z)],
        [(x, y, z) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y, z], [U], []
    )
    disc = FEMDiscretization(Ω, t)
    ops = with_columns(operators(pdesys, disc))
    @test size(ops.P, 2) == 4^3
    diff = reldiff(reduced_solution(ops, 0.05), discretize_solution(pdesys, disc, 0.05))
    println("periodic cube relative difference: ", diff)
    @test diff < 1.0e-10
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), 8 x 8 triangle pairs
# BCs:    periodic left and right; u = 0 on bottom and top; IC sin(2 pi x) sin(pi y)
# Shows:  where a periodic class holds a prescribed dof, that dof is the master, and the reduced system with the prescribed
#         columns removed matches `discretize`.
@testset "operators with periodic x and Dirichlet y keeps the prescribed dof as master" begin
    Ω = MeshDomain(generate_grid(Triangle, (8, 8), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ sin(2π * x) * sin(π * y)],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    disc = FEMDiscretization(Ω, t)
    ops = with_columns(operators(pdesys, disc))
    diff = reldiff(reduced_solution(ops, 0.05), discretize_solution(pdesys, disc, 0.05))
    println("periodic x, Dirichlet y relative difference: ", diff)
    @test diff < 1.0e-10
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, t in (0, 0.05), 8 x 8 triangle pairs, P1
# BCs:    periodic left and right; u = 0 on bottom and top
# Shows:  the left and right nodes on a Dirichlet edge are one periodic class, so `P` has one column for the pair and maps
#         every reduced vector to equal values at equivalent dofs, although neither is tied by the solver.
@testset "P has one column per periodic class when the class holds several prescribed dofs" begin
    n = 8
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ 0, u(t, top) ~ 0, u(0, x, y) ~ sin(2π * x) * sin(π * y)],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    ops = operators(pdesys, FEMDiscretization(Ω, t))
    xy = dof_coordinates(ops.metadata)
    key(d) = (round(mod(xy[1, d], 1.0); digits = 8), round(xy[2, d]; digits = 8))
    classes = unique(key.(axes(xy, 2)))
    @test size(ops.P, 2) == length(classes) == n * (n + 1)
    @test length(ops.tied) == size(xy, 2) - length(classes)
    @test all(count(!iszero, ops.P[d, :]) == 1 && sum(ops.P[d, :]) == 1 for d in axes(ops.P, 1))
    values = ops.P * randn(size(ops.P, 2))
    @test all(values[d] == values[first(findall(e -> key(e) == key(d), axes(xy, 2)))] for d in axes(xy, 2))
end

# PDE:    u_t = u_xx + u_yy
# Domain: the unit square, 8 x 8 triangle pairs, P1
# BCs:    periodic left and right; u = 5e-9 x on bottom and u = 0 on top, whose corner values differ within the agreement tolerance
# Shows:  the returned `u0` and `dirichlet_values` are constant on each periodic class, so `P * c == u0` for some `c`.
@testset "operators returns u0 and Dirichlet values that P reproduces when corner data agree within tolerance" begin
    Ω = MeshDomain(generate_grid(Triangle, (8, 8), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ 5.0e-9 * x, u(t, top) ~ 0, u(0, x, y) ~ sin(2π * x) * sin(π * y)],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    ops = operators(pdesys, FEMDiscretization(Ω, t))
    master_dofs = [first(findall(==(1.0), ops.P[:, j])) for j in axes(ops.P, 2)]
    @test ops.P * ops.u0[master_dofs] == ops.u0
    dirichlet = zeros(size(ops.P, 1))
    dirichlet[ops.prescribed] = ops.dirichlet_values
    @test (ops.P * dirichlet[master_dofs])[ops.prescribed] == ops.dirichlet_values
    xy = dof_coordinates(ops.metadata)
    corners = [d for d in ops.prescribed if xy[2, d] == 0 && xy[1, d] in (0.0, 1.0)]
    @test length(corners) == 2
    spread = abs(dirichlet[corners[1]] - dirichlet[corners[2]]) + abs(ops.u0[corners[1]] - ops.u0[corners[2]])
    println("bottom corner values u0: ", ops.u0[corners], ", Dirichlet: ", dirichlet[corners], ", spread: ", spread)
    @test spread == 0
end

# PDE:    u_t = u_xx + u_yy + 1
# Domain: the unit square, t in (0, 0.05), 8 x 8 triangle pairs, P1
# BCs:    periodic in x and y; IC u = 0
# Shows:  `P' * F` carries the source: the reduced system gives u = t, as `discretize` does.
@testset "operators reduces a constant source on a periodic box through P'F" begin
    Ω = MeshDomain(generate_grid(Triangle, (8, 8), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + 1],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ 0],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    disc = FEMDiscretization(Ω, t)
    ops = with_columns(operators(pdesys, disc))
    reduced = reduced_solution(ops, 0.05)
    exact_error = norm(reduced .- 0.05, Inf)
    diff = reldiff(reduced, discretize_solution(pdesys, disc, 0.05))
    println("periodic source u(t) = t error: ", exact_error, ", relative difference from discretize: ", diff)
    @test exact_error < 1.0e-10
    @test diff < 1.0e-10
end

# PDE:    u_t = u_xx + u_yy, mass and stiffness only
# Domain: the unit square, 8 x 8 triangle pairs, P1, periodic in x and y
# Shows:  the generalized eigenvalues of (P'KP, P'MP) are the discrete Fourier symbols of the uniform triangle grid:
#         K has the five-point stencil (4, -1, -1, -1, -1) and M = h^2/12 (6; 1 on the four axis neighbours; 1 along the cut diagonal).
@testset "periodic operators on a structured triangle grid have the discrete Fourier eigenvalues" begin
    n = 8
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))],
        [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ 0],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], []
    )
    ops = operators(pdesys, FEMDiscretization(Ω, t))
    λ = sort(eigvals(Symmetric(Matrix(ops.P' * ops.K * ops.P)), Symmetric(Matrix(ops.P' * ops.M * ops.P))))
    h = 1 / n
    θ = [2π * k / n for k in 0:(n - 1)]
    symbols = sort(
        vec(
            [
                (4 - 2cos(a) - 2cos(b)) / (h^2 / 12 * (6 + 2cos(a) + 2cos(b) + 2cos(a - b))) for a in θ, b in θ
            ]
        )
    )
    err = maximum(abs.(λ - symbols) ./ max.(1, abs.(symbols)))
    println("periodic triangle grid eigenvalue error: ", err)
    @test err < 1.0e-12
end

# PDE:    u_t = u_xx + u_yy on the unit square
# Shows:  without periodic pairs `P` is the identity and no dof is tied.
@testset "operators without periodic pairs returns the identity P" begin
    Ω = MeshDomain(generate_grid(Triangle, (4, 3), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    sides = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    @named pdesys = PDESystem(
        [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))], [[u(t, Γ) ~ 0 for Γ in sides]; u(0, x, y) ~ 0],
        [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], []
    )
    ops = operators(pdesys, FEMDiscretization(Ω, t))
    n = size(ops.M, 1)
    @test ops.P == sparse(1:n, 1:n, 1.0, n, n)
    @test isempty(ops.tied)
end
