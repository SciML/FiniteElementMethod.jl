using FiniteElementMethod, Ferrite, SciMLBase, Symbolics, Test
using OrdinaryDiffEqBDF: DFBDF
using ModelingToolkit: PDESystem, @named, t_nounits
using FiniteElementMethod: FEMMetadata, piece_nodes, normal_flux, write_vtk

include(joinpath(@__DIR__, "..", "shared", "heat.jl"))
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

const P1 = Lagrange{RefTriangle, 1}()
const P2 = Lagrange{RefTriangle, 2}()

direct_eval(grid, dh, U, field, px, py) = evaluate_at_points(PointEvalHandler(grid, [Vec(px, py)]), dh, U, field)[1]

# The saved states are read lazily and one at a time: the elements are `_full` of each saved vector, in order, the
# solution's own vectors are untouched, and the consumers give what a stack of per-state reads gives.
function states_checks(sol)
    md = sol.disc_data
    saved = deepcopy(sol.original_sol.u)
    expected = [FiniteElementMethod._full(md, U) for U in saved]
    @test collect(FiniteElementMethod._states(sol)) == expected
    @test sol.original_sol.u == saved
    first(FiniteElementMethod._states(sol))
    @test @allocated(first(FiniteElementMethod._states(sol))) < 4 * sizeof(first(saved))
    f = first(md.fields)
    projector = FiniteElementMethod._gradient_projector(md, f)
    reference = FiniteElementMethod._stack(sol, f, [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in expected])
    @test sol[Differential(Num(first(md.x̄)))(Num(f.dv))] == reference
    return nothing
end

# Wrapper checks shared by the square, the L-shape and the two-field case.
function wrapper_checks(grid, sol, syms)
    (; t, x, y, u, Γ, dh) = syms
    @test sol isa SciMLBase.PDETimeSeriesSolution
    @test sol.disc_data isa FEMMetadata
    @test sol.original_sol isa SciMLBase.DAESolution
    @test SciMLBase.successful_retcode(sol)
    nt = length(sol.t)
    for k in (1, nt ÷ 2, nt), (px, py) in ((0.2, 0.3), (0.45, 0.8), (0.9, 0.1))
        @test abs(sol(sol.t[k], px, py; dv = u(t, x, y)) - direct_eval(grid, dh, sol.original_sol.u[k], :u, px, py)) <= 1.0e-14
    end
    @testset "wrapper trace equals the raw inlet dofs" begin
        inlet_dofs = intersect(findall(.!sol.prob.differential_vars), dof_range_global(dh, :u))
        inlet_raw = permutedims(reduce(hcat, [sort(U[inlet_dofs]) for U in sol.original_sol.u]))
        @test sort(sol[u(t, Γ)]; dims = 2) == inlet_raw
        @test maximum(abs, sol[u(t, Γ)] .- sin.(sol.t)) <= 1.0e-3
    end
    @test isnan(sol(sol.t[end], 0.75, 0.75; dv = u(t, x, y))) == !inside(grid, 0.75, 0.75)
    @test_throws DomainError sol(sol.t[end] + 1, 0.2, 0.3; dv = u(t, x, y))
    @test_throws DomainError sol(sol.t[end] + 1, Γ; dv = u(t, x, y))
    mktempdir() do dir
        pvd = write_vtk(sol, joinpath(dir, "heat"))
        @test isfile(pvd)
        @test count(endswith(".vtu"), readdir(dir)) == nt
        @test count("<DataSet", read(pvd, String)) == nt
    end
    states_checks(sol)
    return nothing
end

_stack_rows(rows) = permutedims(reduce(hcat, rows))

inside(grid, px, py) = PointEvalHandler(grid, [Vec(px, py)]; warn = false).cells[1] !== nothing

@testset "Unit square, P1" begin
    grid = with_inlet(square_grid(8))
    prob, syms = heat_problem(grid, (; u = P1))
    sol = solve(prob, DFBDF())
    wrapper_checks(grid, sol, syms)
    (; t, x, y, u, unknowns, Γ, dh) = syms
    U = u(t, x, y)
    nt, nv = length(sol.t), getnnodes(grid)

    @test sol[t] == sol.t
    @test sol[x] == [get_node_coordinate(n)[1] for n in getnodes(grid)]
    @test sol[y] == [get_node_coordinate(n)[2] for n in getnodes(grid)]
    @test size(sol[U]) == (nt, nv)
    @test sol[U, 3, :] == sol[U][3, :]
    @test_throws ArgumentError sol[U, end, :]
    @test sol.original_sol[only(unknowns)] == sol.original_sol.u
    @test sort(sol[U][end, :]) == sort(sol.original_sol.u[end])
    k, i = nt ÷ 2, 17
    @test sol(sol.t[k], sol[x][i], sol[y][i]; dv = U) ≈ sol[U][k, i] atol = 1.0e-14

    xs, ys = range(0, 1, length = 5), range(0, 1, length = 4)
    grid_vals = sol(0.5, xs, ys; dv = U)
    @test size(grid_vals) == (5, 4)
    @test grid_vals[2, 3] == sol(0.5, xs[2], ys[3]; dv = U)
    @test size(sol(sol.t, xs, 0.3; dv = U)) == (nt, 5)
    @test sol(sol.t, xs, 0.3; dv = U)[k, 2] == sol(sol.t[k], xs[2], 0.3; dv = U)
    @test sol(0.5, 0.2, 0.3) == [sol(0.5, 0.2, 0.3; dv = U)]
    pts = [Vec(0.2, 0.3), Vec(0.7, 0.1), Vec(2.0, 0.0)]
    vals = sol(0.5, pts; dv = U)
    @test vals[1:2] == [sol(0.5, 0.2, 0.3; dv = U), sol(0.5, 0.7, 0.1; dv = U)]
    @test isnan(vals[3])
    sol(0.5, pts; dv = U)
    ph = sol.interp.cache[].ph
    sol(0.6, pts; dv = U)
    @test sol.interp.cache[].ph === ph

    @test isnan(sol(1.0, 2.0, 2.0; dv = U))
    @test isequal(sol(1.0, [0.5, 1.5], 0.5; dv = U)[2], NaN)
    @test_throws DomainError sol(1.0, 2.0, 2.0; dv = U, outside = :error)
    @test_throws DomainError sol(-0.5, 0.2, 0.3; dv = U)

    nodes = piece_nodes(sol, Γ)
    @test length(nodes) == 9
    @test all(iszero, sol[x][nodes])
    @test size(sol[u(t, Γ)]) == (nt, 9)
    @test sol[u(t, Γ)] == sol[U][:, nodes]
    @test sol(0.5, Γ; dv = U) ≈ sin(0.5) * ones(9) atol = 1.0e-3
    @test sol(sol.t[k], Γ; dv = U) ≈ sol[u(t, Γ)][k, :] atol = 1.0e-14
    @test_throws ArgumentError sol[u(t, TestPiece("nowhere"))]

    Dx, Dy = Differential(x), Differential(y)
    h = 1.0e-6
    gx = sol(1.0, 0.21, 0.33; dv = Dx(U))
    fd = (sol(1.0, 0.21 + h, 0.33; dv = U) - sol(1.0, 0.21 - h, 0.33; dv = U)) / 2h
    @test gx ≈ fd rtol = 1.0e-6
    @test size(sol[Dx(U)]) == (nt, nv)
    field = only(sol.disc_data.fields)
    projector = FiniteElementMethod._gradient_projector(sol.disc_data, field)
    @test sol[Dx(U)] == _stack_rows([FiniteElementMethod._nodal_gradient(sol.disc_data, field, 1, V, projector) for V in sol.original_sol.u])
    sol[Dx(U)]
    build_bytes = @allocated FiniteElementMethod._gradient_projector(sol.disc_data, field)
    @test @allocated(sol[Dx(U)]) < 3 * build_bytes + nt * @allocated(FiniteElementMethod._nodal_gradient(sol.disc_data, field, 1, sol.original_sol.u[1], projector))
    @test size(sol[Dy(U)]) == (nt, nv)
    err = try
        sol[Dy(Dy(U))]
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("first", sprint(showerror, err))
    @test_throws ArgumentError sol[Differential(t)(U)]
    @test_throws ArgumentError sol(1.0, 0.2, 0.3; dv = Dy(Dy(U)))

    flux = normal_flux(sol, U, Γ)
    @test size(flux) == (nt, 8)
    @test all(isfinite, flux)
    @test_throws ArgumentError normal_flux(sol, x, Γ)

    @test sprint(show, MIME"text/plain"(), sol) isa String

    tight = solve(prob, DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test maximum(abs, tight[u(t, Γ)] .- sin.(tight.t)) <= 1.0e-9

    mktempdir() do dir
        pvd = write_vtk(sol, joinpath(dir, "heat&<flow>'\""))
        xml = read(pvd, String)
        @test occursin("file=\"heat&amp;&lt;flow&gt;&apos;&quot;_0001.vtu\"", xml)
        @test !occursin(r"&(?!amp;|lt;|gt;|apos;|quot;)", xml)
        @test count(endswith(".vtu"), readdir(dir)) == nt
    end
end

@testset "L-shape, P1" begin
    grid = lshape_grid(8)
    prob, syms = heat_problem(grid, (; u = P1))
    sol = solve(prob, DFBDF())
    @test !inside(grid, 0.75, 0.75)
    wrapper_checks(grid, sol, syms)
    (; t, x, y, u) = syms
    @test length(sol[x]) == getnnodes(grid)
    @test all(isfinite, sol[u(t, x, y)])
    @test isnan(sol(1.0, 0.75, 0.75; dv = u(t, x, y)))
    @test_throws DomainError sol(1.0, 0.75, 0.75; dv = u(t, x, y), outside = :error)
end

@testset "Two fields, P1 and P2" begin
    grid = with_inlet(square_grid(6))
    prob, syms = heat_problem(grid, (; u = P1, w = P2))
    sol = solve(prob, DFBDF())
    wrapper_checks(grid, sol, syms)
    (; t, x, y, u, w, unknowns, Γ, dh) = syms
    nt, nv = length(sol.t), getnnodes(grid)
    @test length(unknowns[2]) > nv
    @test size(sol[w(t, x, y)]) == (nt, nv)
    k = nt ÷ 2
    for i in (1, 10, nv)
        @test sol(sol.t[k], sol[x][i], sol[y][i]; dv = w(t, x, y)) ≈ sol[w(t, x, y)][k, i] atol = 1.0e-12
    end
    for px in (0.1, 0.55), py in (0.3, 0.95)
        @test abs(sol(sol.t[k], px, py; dv = w(t, x, y)) - direct_eval(grid, dh, sol.original_sol.u[k], :w, px, py)) <= 1.0e-14
    end
    @test maximum(abs, sol[w(t, Γ)] .- sin.(sol.t)) <= 1.0e-3
    @test length(sol(0.5, 0.2, 0.3)) == 2
    @test sol.original_sol[unknowns[2]] == [U[dof_range_global(dh, :w)] for U in sol.original_sol.u]
end

@testset "Steady solution" begin
    grid = square_grid(4)
    with_inlet(grid)
    sol, syms = steady_solution(grid, (; u = P1))
    (; x, y, u, Γ) = syms
    U = u(x, y)
    @test sol isa SciMLBase.PDENoTimeSolution
    @test sol.disc_data isa FEMMetadata{Val(false)}
    @test sol[U] ≈ 1 .- sol[x] atol = 1.0e-12
    @test sol(0.3, 0.6; dv = U) ≈ 0.7 atol = 1.0e-12
    @test size(sol(range(0, 1, length = 3), 0.5; dv = U)) == (3,)
    @test sol[u(Γ)] ≈ ones(length(piece_nodes(sol, Γ))) atol = 1.0e-12
    @test sol(Γ; dv = U) ≈ ones(length(piece_nodes(sol, Γ))) atol = 1.0e-12
    @test normal_flux(sol, U, Γ) ≈ ones(4) atol = 1.0e-10
    @test sol[Differential(x)(U)] ≈ -ones(getnnodes(grid)) atol = 1.0e-10
    @test sol(0.3, 0.6; dv = Differential(x)(U)) ≈ -1 atol = 1.0e-12
    @test isnan(sol(1.5, 0.5; dv = U))
    @test sol[U^2 + x] ≈ sol[U] .^ 2 .+ sol[x] atol = 1.0e-12
    @test sol[2U + Differential(x)(U)] ≈ 1 .- 2 .* sol[x] atol = 1.0e-10
    @test sol(0.3, 0.6; dv = U * y) ≈ 0.7 * 0.6 atol = 1.0e-12
    @test sol(Γ; dv = U + y) ≈ 1 .+ sol[y][piece_nodes(sol, Γ)] atol = 1.0e-12
    fixed = Differential(y)(u(0, y)) + 1
    msg = ArgumentError("cannot evaluate the expression $fixed on this solution: $(u(0, y)) fixes arguments of u(x, y); write the dependent variable as u(x, y)")
    @test_throws msg sol[fixed]
    @test_throws msg sol(0.3, 0.6; dv = fixed)
    bare = ArgumentError("$(u(0, y)) fixes arguments of u(x, y); write the dependent variable as u(x, y)")
    for ex in (u(0, y), Differential(y)(u(0, y)))
        @test_throws bare sol[ex]
        @test_throws bare sol(0.3, 0.6; dv = ex)
    end
    @test_throws bare normal_flux(sol, u(0, y), Γ)
end

@testset "Points missed by Ferrite's nearest-node search" begin
    nodes = Node.([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (5.1, 5.1), (5.2, 5.1), (5.1, 5.2)])
    cells = Triangle.([(1, 2, 3), (2, 4, 3), (2, 5, 4), (4, 5, 6), (3, 4, 6)])
    grid = Grid(cells, nodes)
    addfacetset!(grid, "inlet", x -> x[1] ≈ 0.0)
    @test PointEvalHandler(grid, [Vec(4.9, 4.9)]; warn = false).cells[1] === nothing
    dh = DofHandler(grid)
    add!(dh, :u, P1)
    close!(dh)
    U = zeros(ndofs(dh))
    apply_analytical!(U, dh, :u, x -> x[1] + 2x[2])
    sol, syms = steady_from_dofs(dh, (; u = P1), U)
    (; x, y, u) = syms
    @test sol(4.9, 4.9; dv = u(x, y)) ≈ 4.9 + 2 * 4.9
    @test sol(4.9, 4.9; dv = u(x, y), outside = :error) ≈ 4.9 + 2 * 4.9
    @test isnan(sol(11.0, 11.0; dv = u(x, y)))
    @test isnan(sol(5.2, 5.2; dv = u(x, y)))
    @test_throws DomainError sol(5.2, 5.2; dv = u(x, y), outside = :error)
end

@testset "Projected derivatives reproduce polynomials" begin
    for (ip, g, dgx, dgy) in (
            (P1, x -> 3x[1] - 2x[2], x -> 3.0, x -> -2.0),
            (P2, x -> x[1]^2 + x[1] * x[2], x -> 2x[1] + x[2], x -> x[1]),
        )
        for n in (1, 3)
            grid = square_grid(n)
            dh = DofHandler(grid)
            add!(dh, :u, ip)
            close!(dh)
            U = zeros(ndofs(dh))
            apply_analytical!(U, dh, :u, g)
            sol, syms = steady_from_dofs(dh, (; u = ip), U)
            (; x, y, u) = syms
            pts = Vec.(sol[x], sol[y])
            @test sol[Differential(x)(u(x, y))] ≈ dgx.(pts) atol = 1.0e-12
            @test sol[Differential(y)(u(x, y))] ≈ dgy.(pts) atol = 1.0e-12
            @test sol(0.3, 0.6; dv = Differential(x)(u(x, y))) ≈ dgx(Vec(0.3, 0.6)) atol = 1.0e-12
        end
    end
end

@testset "Backward time span" begin
    grid = square_grid(2)
    dh = DofHandler(grid)
    add!(dh, :u, P1)
    close!(dh)
    U = zeros(ndofs(dh))
    apply_analytical!(U, dh, :u, x -> x[1] + x[2])
    prob, syms = constant_problem(dh, (; u = P1), U, (1.0, 0.0))
    sol = solve(prob, DFBDF())
    (; t, x, y, u) = syms
    @test first(sol.t) == 1.0 && last(sol.t) == 0.0
    @test sol(0.5, 0.2, 0.3; dv = u(t, x, y)) ≈ 0.5 atol = 1.0e-12
    @test sol(1.0, 0.2, 0.3; dv = u(t, x, y)) ≈ 0.5 atol = 1.0e-12
    @test sol(0.0, 0.2, 0.3; dv = u(t, x, y)) ≈ 0.5 atol = 1.0e-12
    @test_throws DomainError sol(1.5, 0.2, 0.3; dv = u(t, x, y))
    @test_throws DomainError sol(-0.5, 0.2, 0.3; dv = u(t, x, y))
end

@testset "Quadratic geometry" begin
    grid = with_inlet(generate_grid(QuadraticTriangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    for (g, dgx, dgy) in (
            (x -> x[1] + 2x[2], x -> 1.0, x -> 2.0),
            (x -> x[1]^2 + x[1] * x[2], x -> 2x[1] + x[2], x -> x[1]),
        )
        dh = DofHandler(grid)
        add!(dh, :u, P2)
        close!(dh)
        U = zeros(ndofs(dh))
        apply_analytical!(U, dh, :u, g)
        sol, syms = steady_from_dofs(dh, (; u = P2), U)
        (; x, y, u, Γ) = syms
        p = Vec(0.3, 0.6)
        @test sol(p[1], p[2]; dv = u(x, y)) ≈ g(p) atol = 1.0e-12
        @test sol(p[1], p[2]; dv = Differential(x)(u(x, y))) ≈ dgx(p) atol = 1.0e-12
        @test sol(p[1], p[2]; dv = Differential(y)(u(x, y))) ≈ dgy(p) atol = 1.0e-12
        nodes = Vec.(sol[x], sol[y])
        @test sol[u(x, y)] ≈ g.(nodes) atol = 1.0e-12
        @test sol[Differential(x)(u(x, y))] ≈ dgx.(nodes) atol = 1.0e-12
        @test sol[Differential(y)(u(x, y))] ≈ dgy.(nodes) atol = 1.0e-12
        # Facet means of -∂u/∂x over the inlet's two facets, y ∈ [0, 1/2] and [1/2, 1].
        inlet_flux = [-dgx(Vec(0.0, 0.25)), -dgx(Vec(0.0, 0.75))]
        @test normal_flux(sol, u(x, y), Γ) ≈ inlet_flux atol = 1.0e-12
        nodes_on_inlet = piece_nodes(sol, Γ)
        @test nodes_on_inlet == findall(n -> get_node_coordinate(n)[1] == 0.0, getnodes(grid))
        @test length(nodes_on_inlet) == 5
        @test sol[u(Γ)] == sol[u(x, y)][nodes_on_inlet]
        @test sol[u(Γ)] ≈ g.(Vec.(sol[x][nodes_on_inlet], sol[y][nodes_on_inlet])) atol = 1.0e-12
        @test sol(Γ; dv = u(x, y)) == sol[u(Γ)]
    end
end

@testset "Quadratic geometry, transient trace" begin
    grid = with_inlet(generate_grid(QuadraticTriangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    prob, syms = heat_problem(grid, (; u = P2))
    sol = solve(prob, DFBDF())
    (; t, x, y, u, Γ) = syms
    nodes_on_inlet = piece_nodes(sol, Γ)
    @test sort(nodes_on_inlet) == findall(n -> get_node_coordinate(n)[1] == 0.0, getnodes(grid))
    @test all(iszero, sol[x][nodes_on_inlet])
    @test size(sol[u(t, Γ)]) == (length(sol.t), 5)
    @test sol[u(t, Γ)] == sol[u(t, x, y)][:, nodes_on_inlet]
    k = length(sol.t) ÷ 2
    @test sol(sol.t[k], Γ; dv = u(t, x, y)) ≈ sol[u(t, Γ)][k, :] atol = 1.0e-14
    @test maximum(abs, sol[u(t, Γ)] .- sin.(sol.t)) <= 1.0e-3
end

@testset "Complex coefficients" begin
    grid = with_inlet(square_grid(2))
    dh = DofHandler(grid)
    add!(dh, :u, P1)
    close!(dh)
    c = 1.0 + 2.0im
    U = fill(c, ndofs(dh))
    nv = getnnodes(grid)

    steady, syms = steady_from_dofs(dh, (; u = P1), U)
    (; x, y, u, Γ) = syms
    @test steady[u(x, y)] isa Vector{ComplexF64}
    @test steady[u(x, y)] ≈ fill(c, nv)
    @test steady(0.3, 0.6; dv = u(x, y)) ≈ c
    vals = steady([0.3, 2.0], 0.6; dv = u(x, y))
    @test eltype(vals) == ComplexF64 && vals[1] ≈ c && isnan(vals[2])
    @test steady[u(Γ)] ≈ fill(c, length(piece_nodes(steady, Γ)))
    @test abs(steady(0.3, 0.6; dv = Differential(x)(u(x, y)))) <= 1.0e-12
    @test maximum(abs, steady[Differential(x)(u(x, y))]) <= 1.0e-12
    @test maximum(abs, normal_flux(steady, u(x, y), Γ)) <= 1.0e-12

    prob, tsyms = constant_problem(dh, (; u = P1), U, (0.0, 1.0))
    raw = SciMLBase.build_solution(prob, DFBDF(), [0.0, 1.0], [U, 2U]; retcode = SciMLBase.ReturnCode.Success)
    sol = SciMLBase.wrap_sol(raw)
    (; t, x, y, u) = tsyms
    @test sol isa SciMLBase.PDETimeSeriesSolution
    @test sol[u(t, x, y)] isa Matrix{ComplexF64}
    @test sol[u(t, x, y)] ≈ [fill(c, 1, nv); fill(2c, 1, nv)]
    @test sol(1.0, 0.3, 0.6; dv = u(t, x, y)) ≈ 2c
    vals = sol([0.0, 1.0], [0.3, 2.0], 0.6; dv = u(t, x, y))
    @test eltype(vals) == ComplexF64 && vals[:, 1] ≈ [c, 2c] && all(isnan, vals[:, 2])
end

@testset "Complex gradients" begin
    grid = with_inlet(square_grid(3))
    a, b = 1.0 + 2.0im, 3.0 - 1.0im
    for (ip, g, dgx, dgy) in (
            (P1, p -> a * p[1] + b * p[2], p -> a, p -> b),
            (P2, p -> a * p[1]^2 + b * p[1] * p[2], p -> 2a * p[1] + b * p[2], p -> b * p[1]),
        )
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
        U = zeros(ComplexF64, ndofs(dh))
        apply_analytical!(U, dh, :u, g)
        sol, syms = steady_from_dofs(dh, (; u = ip), U)
        (; x, y, u, Γ) = syms
        nodes = Vec.(sol[x], sol[y])
        @test sol[u(x, y)] ≈ g.(nodes) atol = 1.0e-12
        @test sol[Differential(x)(u(x, y))] ≈ dgx.(nodes) atol = 1.0e-12
        @test sol[Differential(y)(u(x, y))] ≈ dgy.(nodes) atol = 1.0e-12
        p = Vec(0.3, 0.6)
        @test sol(p[1], p[2]; dv = Differential(x)(u(x, y))) ≈ dgx(p) atol = 1.0e-12
        @test sol(p[1], p[2]; dv = Differential(y)(u(x, y))) ≈ dgy(p) atol = 1.0e-12
        # Inlet facets span y ∈ [0, 1/3], [1/3, 2/3], [2/3, 1]; -∂u/∂x is linear in y, so the facet mean is its midpoint value.
        @test normal_flux(sol, u(x, y), Γ) ≈ [-dgx(Vec(0.0, ym)) for ym in (1 / 6, 1 / 2, 5 / 6)] atol = 1.0e-12
    end
end

# A mirror dof of a periodic pair carries the solve's residual for `u_mirror - u_master = 0`; the
# field the solution reports lies on the periodic manifold, so mirrors read as their master's value.
@testset "Periodic mirror dofs report their master's value" begin
    grid = square_grid(4)
    space = FiniteElementMethod.FEMSpace(grid, [:u => P1]; periodic = [(:u, :left, :right)])
    md0, syms = heat_metadata(space.dh, (; u = P1); steady = true)
    md = FEMMetadata(space, md0.time, md0.x̄, md0.fields, md0.pieces, nothing)
    (; x, y, u) = syms
    U = collect(Float64, 1:ndofs(space.dh))
    prob = NonlinearProblem{true}(NonlinearFunction{true}((res, u, p) -> nothing), U, SciMLBase.NullParameters(), md)
    raw = SciMLBase.build_solution(prob, nothing, U, zero(U); retcode = SciMLBase.ReturnCode.Success)
    sol = SciMLBase.wrap_sol(raw)

    masters = copy(U)
    foreach(((m, s),) -> masters[m] = masters[s], space.ties)
    @test !isempty(space.ties)
    @test sol[u(x, y)] == Ferrite.evaluate_at_grid_nodes(space.dh, masters, :u)
    for y0 in (0.3, 0.6, 0.8)
        @test sol(1.0, y0; dv = u(x, y)) ≈ sol(0.0, y0; dv = u(x, y)) atol = 1.0e-14
    end

end

# Two saved states of the field (1 + t) g with known offsets on the mirror dofs, linear in between.
@testset "Periodic seam of a saved time series" begin
    grid = square_grid(4)
    space = FiniteElementMethod.FEMSpace(grid, [:u => P1]; periodic = [(:u, :left, :right)])
    md0, syms = heat_metadata(space.dh, (; u = P1))
    md = FEMMetadata(space, md0.time, md0.x̄, md0.fields, md0.pieces, nothing)
    (; t, x, y, u) = syms
    g = p -> 2 + 3p[2] + abs(2p[1] - 1)
    G = zeros(ndofs(space.dh))
    apply_analytical!(G, space.dh, :u, g)
    mirrors = first.(space.ties)
    states = [copy(G), 2G]
    states[1][mirrors] .+= 5
    states[2][mirrors] .+= 15
    saved = deepcopy(states)
    ys = (0.3, 0.6, 0.8)
    for (U, jump) in zip(states, (5, 15)), y0 in ys
        @test direct_eval(grid, space.dh, U, :u, 1.0, y0) - direct_eval(grid, space.dh, U, :u, 0.0, y0) ≈ jump atol = 1.0e-12
    end

    prob = DAEProblem(DAEFunction{true}((res, du, v, p, t) -> nothing), zero(G), states[1], (0.0, 1.0), SciMLBase.NullParameters(), md)
    raw = SciMLBase.build_solution(prob, DFBDF(), [0.0, 1.0], states; retcode = SciMLBase.ReturnCode.Success)
    sol = SciMLBase.wrap_sol(raw)
    nodes = Vec.(sol[x], sol[y])
    @test sol[u(t, x, y)] ≈ [1, 2] .* permutedims(g.(nodes)) atol = 1.0e-12
    for time in (0.0, 0.375, 1.0), x0 in (0.0, 1.0), y0 in ys
        @test sol(time, x0, y0; dv = u(t, x, y)) ≈ (1 + time) * g(Vec(x0, y0)) atol = 1.0e-12
    end
    @test sol.original_sol.u == saved
    states_checks(sol)
end

@testset "Periodic seam of a DFBDF solve" begin
    t = t_nounits
    @variables x y u(..) v(..)
    Δ(w) = Differential(x)(Differential(x)(w)) + Differential(y)(Differential(y)(w))
    Ω = MeshDomain(square_grid(6))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U, V = u(t, x, y), v(t, x, y)
    eqs = [Differential(t)(U) ~ Δ(U), Differential(t)(V) ~ Δ(V) + U]
    bcs = [
        u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), v(t, left) ~ v(t, right),
        u(0, x, y) ~ sin(2pi * x) * cos(2pi * y), v(0, x, y) ~ 1,
    ]
    tend = 0.002
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ DomainSets.Interval(0.0, tend)], [t, x, y], [U, V], [])
    disc = FEMDiscretization(Ω, t; spaces = Dict(u => P2, v => P1), natural = [bottom, top])
    sol = solve(discretize(pdesys, disc), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    edge = collect(range(0.0, 1.0; length = 101))
    @test maximum(abs, sol(tend, [0.0], edge; dv = U) .- sol(tend, [1.0], edge; dv = U)) <= 1.0e-12
    states_checks(sol)
end

@testset "A custom space's `ties` property is not read" begin
    dh = DofHandler(square_grid(2))
    add!(dh, :u, P1)
    close!(dh)
    g = p -> 2 + 3p[2]
    U = zeros(ndofs(dh))
    apply_analytical!(U, dh, :u, g)
    md0, syms = heat_metadata(dh, (; u = P1); steady = true)
    md = FEMMetadata((; dh, ties = nothing), md0.time, md0.x̄, md0.fields, md0.pieces, nothing)
    (; x, y, u) = syms
    prob = NonlinearProblem{true}(NonlinearFunction{true}((res, u, p) -> nothing), U, SciMLBase.NullParameters(), md)
    raw = SciMLBase.build_solution(prob, nothing, U, zero(U); retcode = SciMLBase.ReturnCode.Success)
    sol = SciMLBase.wrap_sol(raw)
    @test sol[u(x, y)] ≈ g.(Vec.(sol[x], sol[y])) atol = 1.0e-14
    @test sol(0.3, 0.6; dv = u(x, y)) ≈ g(Vec(0.3, 0.6)) atol = 1.0e-14
end

@testset "Periodic solution with read-only coefficients" begin
    space = FiniteElementMethod.FEMSpace(square_grid(4), [:u => P1]; periodic = [(:u, :left, :right)])
    md0, syms = heat_metadata(space.dh, (; u = P1); steady = true)
    md = FEMMetadata(space, md0.time, md0.x̄, md0.fields, md0.pieces, nothing)
    (; x, y, u) = syms
    U = range(2.0, 2.0; length = ndofs(space.dh))
    prob = NonlinearProblem{true}(NonlinearFunction{true}((res, u, p) -> nothing), U, SciMLBase.NullParameters(), md)
    raw = SciMLBase.build_solution(prob, nothing, U, zeros(length(U)); retcode = SciMLBase.ReturnCode.Success)
    sol = SciMLBase.wrap_sol(raw)
    @test sol[u(x, y)] == fill(2.0, getnnodes(space.grid))
    for x0 in (0.0, 1.0), y0 in (0.3, 0.6)
        @test sol(x0, y0; dv = u(x, y)) ≈ 2.0 atol = 1.0e-14
    end
    @test sol.original_sol.u isa StepRangeLen
end

# A fixed low facet quadrature order was exact only for straight facets up to P4.
@testset "Facet-mean flux at high order and on curved facets" begin
    # u = x y⁴ is in P5; the inlet normal is -x, so the mean of -∂u/∂x = -y⁴ over [a, b] is -(b⁵ - a⁵) / (5 (b - a)).
    grid = with_inlet(square_grid(3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 5}())
    close!(dh)
    U = zeros(ndofs(dh))
    apply_analytical!(U, dh, :u, p -> p[1] * p[2]^4)
    sol, syms = steady_from_dofs(dh, (; u = Lagrange{RefTriangle, 5}()), U)
    exact = [-((k / 3)^5 - ((k - 1) / 3)^5) / (5 / 3) for k in 1:3]
    @test normal_flux(sol, syms.u(syms.x, syms.y), syms.Γ) ≈ exact rtol = 1.0e-12

    # Quadratic cells with P2 and P4 fields: the facet-mean flux on the outer arc against a high-order reference.
    grid = quarter_annulus(3; quadratic = true)
    for p in (2, 4)
        ip = Lagrange{RefTriangle, p}()
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
        U = zeros(ndofs(dh))
        apply_analytical!(U, dh, :u, X -> X[1]^2 + X[2]^2)
        sol, syms = steady_from_dofs(dh, (; u = ip), U)
        sol.disc_data.pieces[syms.Γ] = "outer"
        fv = FacetValues(FacetQuadratureRule{RefTriangle}(14), ip, geometric_interpolation(QuadraticTriangle))
        reference = map(sort!(collect(getfacetset(grid, "outer")); by = fi -> (fi[1], fi[2]))) do (c, f)
            Ferrite.reinit!(fv, getcoordinates(grid, c), f)
            u_cell = U[celldofs(dh, c)]
            q = 1:getnquadpoints(fv)
            sum(q) do i
                (function_gradient(fv, i, u_cell) ⋅ getnormal(fv, i)) * getdetJdV(fv, i)
            end / sum(i -> getdetJdV(fv, i), q)
        end
        @test normal_flux(sol, syms.u(syms.x, syms.y), syms.Γ) ≈ reference rtol = 1.0e-10
    end
end

function gradient_case(shape, order; periodic = false, second_field = false, region = false, nel = 6)
    t = t_nounits
    @variables x y u(..) v(..)
    Δ(w) = Differential(x)(Differential(x)(w)) + Differential(y)(Differential(y)(w))
    grid = generate_grid(shape, (nel, nel), Vec(0.0, 0.0), Vec(1.0, 1.0))
    region && addcellset!(grid, "left_half", p -> p[1] <= 0.5 + 1.0e-12)
    Ω = MeshDomain(grid)
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    U, V = u(t, x, y), v(t, x, y)
    eqs = [Differential(t)(U) ~ Δ(U)]
    bcs = periodic ? [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top)] : [u(t, Γ) ~ 0 for Γ in (left, right, bottom, top)]
    bcs = [bcs; u(0, x, y) ~ sin(2pi * x) * cos(pi * y)]
    dvs, spaces = [U], Dict{Any, Any}(u => Lagrange{shape === Triangle ? RefTriangle : RefQuadrilateral, order}())
    if second_field || region
        push!(eqs, Differential(t)(V) ~ Δ(V) + U)
        push!(dvs, V)
        append!(bcs, [v(t, Γ) ~ 0 for Γ in (region ? (left,) : (left, right, bottom, top))])
        push!(bcs, v(0, x, y) ~ 1)
        spaces[v] = Lagrange{shape === Triangle ? RefTriangle : RefQuadrilateral, 1}()
    end
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ DomainSets.Interval(0.0, 1.0)], [t, x, y], dvs, [])
    disc = FEMDiscretization(Ω, t; spaces, regions = region ? Dict(v => Region(Ω, :left_half)) : Dict())
    prob = discretize(pdesys, disc)
    n = length(prob.u0)
    ts = collect(range(0.0, 1.0; length = 16))
    states = [prob.u0 .+ 0.3 * sin.(k .* (1:n)) for k in 1:16]
    raw = SciMLBase.build_solution(prob, DFBDF(), ts, states; retcode = SciMLBase.ReturnCode.Success)
    return SciMLBase.wrap_sol(raw), (; t, x, y)
end

# A heat equation with natural boundaries on `grid`, so that every dof is a state, and a series of the given states.
function free_case(grid, ip, states_of)
    t = t_nounits
    @variables x y u(..)
    Ω = MeshDomain(grid)
    U = u(t, x, y)
    eqs = [Differential(t)(U) ~ Differential(x)(Differential(x)(U)) + Differential(y)(Differential(y)(U))]
    @named pdesys = PDESystem(eqs, [u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ DomainSets.Interval(0.0, 1.0)], [t, x, y], [U], [])
    prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces = Dict{Any, Any}(u => ip)))
    series(states) = SciMLBase.wrap_sol(SciMLBase.build_solution(prob, DFBDF(), collect(range(0.0, 1.0; length = length(states))), states; retcode = SciMLBase.ReturnCode.Success))
    dh = FiniteElementMethod._dh(series([prob.u0, prob.u0]).disc_data)
    @test length(prob.u0) == ndofs(dh)
    return series(states_of(dh)), (; x, y, U)
end

@testset "Projected gradients of a time series equal the per-state Ferrite.project route" begin
    cases = (
        ("P1 triangles", Triangle, 1, (;)), ("P2 triangles", Triangle, 2, (;)),
        ("Q1 quadrilaterals", Quadrilateral, 1, (;)), ("Q2 quadrilaterals", Quadrilateral, 2, (;)),
        ("periodic P1", Triangle, 1, (; periodic = true)), ("periodic P2 quadrilaterals", Quadrilateral, 2, (; periodic = true)),
        ("two fields (P2, P1)", Triangle, 2, (; second_field = true)),
    )
    @testset "$name" for (name, shape, order, options) in cases
        sol, (; t, x, y) = gradient_case(shape, order; options...)
        md = sol.disc_data
        states = [FiniteElementMethod._full(md, U) for U in sol.original_sol.u]
        for f in md.fields, (d, coordinate) in enumerate((x, y))
            projector = FiniteElementMethod._gradient_projector(md, f)
            rows = [FiniteElementMethod._nodal_gradient(md, f, d, U, projector) for U in states]
            @test FiniteElementMethod._nodal_gradients(md, f, d, states) == rows
            @test sol[Differential(coordinate)(Num(f.dv))] == FiniteElementMethod._stack(sol, f, rows)
        end
    end

    sol, (; t, x, y) = gradient_case(Triangle, 2)
    md = sol.disc_data
    f = first(md.fields)
    projector = FiniteElementMethod._gradient_projector(md, f)
    states = [FiniteElementMethod._full(md, U) for U in sol.original_sol.u]
    reference = [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in states]
    @test FiniteElementMethod._nodal_gradients(md, f, 1, states; max_bytes = 0) == reference
    @test FiniteElementMethod._nodal_gradients(md, f, 1, states[1:1]) == reference[1:1]
    complex_states = [U .+ im .* reverse(U) for U in states]
    @test FiniteElementMethod._nodal_gradients(md, f, 1, complex_states) == [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in complex_states]
    few = states[1:(FiniteElementMethod.GRADIENT_TABLE_MIN_STATES - 1)]
    @test FiniteElementMethod._nodal_gradients(md, f, 1, few) == reference[eachindex(few)]
    # Below the minimum no tables are built: on Q2 quadrilaterals they would add 4% (0.5 MiB) for two states.
    quads, _ = gradient_case(Quadrilateral, 2; nel = 30)
    quad_md, quad_f = quads.disc_data, first(quads.disc_data.fields)
    pair = [FiniteElementMethod._full(quad_md, U) for U in quads.original_sol.u[1:2]]
    quad_per_state(states) = FiniteElementMethod._nodal_gradients(quad_md, quad_f, 1, states; max_bytes = 0)
    FiniteElementMethod._nodal_gradients(quad_md, quad_f, 1, pair), quad_per_state(pair)
    @test @allocated(FiniteElementMethod._nodal_gradients(quad_md, quad_f, 1, pair)) <= 1.01 * @allocated(quad_per_state(pair))
    mixed = AbstractVector[iseven(k) ? complex_states[k] : states[k] for k in eachindex(states)]
    @test FiniteElementMethod._nodal_gradients(md, f, 1, mixed) == [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in mixed]
    reads = Ref(0)
    lazy(vs) = (
        begin
            reads[] += 1; copy(U)
        end for U in vs
    )
    for series in (states, complex_states, mixed)
        reads[] = 0
        @test FiniteElementMethod._nodal_gradients(md, f, 1, lazy(series)) == [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in series]
        @test reads[] == length(series)
    end
    @test_throws BoundsError FiniteElementMethod._nodal_gradients(md, f, 1, [states[1:2]; [states[3][1:(end - 1)]]; states[4:end]])

    sol, (; t, x, y) = gradient_case(Triangle, 2; region = true)
    md = sol.disc_data
    @test length(md.space.dh.subdofhandlers) > 1
    for f in md.fields
        projector = FiniteElementMethod._gradient_projector(md, f)
        states = [FiniteElementMethod._full(md, U) for U in sol.original_sol.u]
        reference = FiniteElementMethod._stack(sol, f, [FiniteElementMethod._nodal_gradient(md, f, 1, U, projector) for U in states])
        @test isequal(sol[Differential(x)(Num(f.dv))], reference)
    end
end

@testset "Projected gradients of states near floatmax stay finite" begin
    # Two unit-width Q1 cells on [0, 2] × [0, 6]: ∂u/∂x is 0.72 on the left cell and 0.85 - 1.15 y / 6 on the right.
    # Both are linear in y, so the projection is the 1D projection along x of the bottom and top traces.
    grid = generate_grid(Quadrilateral, (2, 1), Vec(0.0, 0.0), Vec(2.0, 6.0))
    nodal(p) = p[1] == 0 ? -0.72 : p[1] == 1 ? 0.0 : (p[2] == 0 ? 0.85 : -0.3)
    sol, (; x, U) = free_case(
        grid, Lagrange{RefQuadrilateral, 1}(), dh -> begin
            V = zeros(ndofs(dh))
            apply_analytical!(V, dh, :u, nodal)
            [[zero(V)]; [1.0e308 .* V for _ in 1:15]]
        end
    )
    G = sol[Differential(x)(U)]
    @test all(isfinite, G)
    @test all(iszero, G[1, :])
    along_x(left, right) = ([2 1 0; 1 4 1; 0 1 2] / 6) \ [left / 2, (left + right) / 2, right / 2]
    bottom, top = along_x(0.72, 0.85), along_x(0.72, -0.3)
    exact = [(p = get_node_coordinate(n); (p[2] == 0 ? bottom : top)[Int(p[1]) + 1]) for n in getnodes(grid)]
    # The 3 × 3 solve and the 4-point cell sums each round a few times: 1e-14 is about 45 ulps.
    @test G[end, :] ./ 1.0e308 ≈ exact rtol = 1.0e-14
end

@testset "Projected gradients recover each row of an affine series" begin
    # Projecting a constant gradient is exact in exact arithmetic, so each row is held to its own slope. Its rounding, from
    # the quadrature sums and the Cholesky solve of a well-conditioned mass matrix, scales with that slope: at most
    # 1.2e-14 relative on these meshes, under rtol 1e-12 by about 80 times.
    slopes = [0.0; [isodd(k) ? 1.0 : 5.0e-13 for k in 1:15]]
    @testset "$name" for (name, shape, ip) in (("P1 triangles", Triangle, P1), ("P2 triangles", Triangle, P2), ("Q1 quadrilaterals", Quadrilateral, Lagrange{RefQuadrilateral, 1}()))
        grid = generate_grid(shape, (6, 6), Vec(0.0, 0.0), Vec(1.0, 1.0))
        sol, (; x, y, U) = free_case(
            grid, ip, dh -> map(slopes) do a
                V = zeros(ndofs(dh))
                apply_analytical!(V, dh, :u, p -> a * p[1] - 3 * a * p[2])
                V
            end
        )
        for (D, scale) in ((Differential(x), 1), (Differential(y), -3))
            G = sol[D(U)]
            @test all(iszero, G[1, :])
            for k in 2:length(slopes)
                @test all(isapprox.(G[k, :], scale * slopes[k]; rtol = 1.0e-12))
            end
        end
    end
end
