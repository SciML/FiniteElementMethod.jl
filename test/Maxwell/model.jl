# The frequency-polynomial operator, PEC traces, material assignment and point evaluation of the Maxwell model.
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Test
include(joinpath(@__DIR__, "..", "shared", "maxwell.jl"))

# Two-material cube: the lower half lossy, the rest vacuum (unnamed).
function two_material_model(n; kwargs...)
    grid = unit_cube(n)
    addcellset!(grid, "lossy", x -> x[3] <= 0.5)
    lossy = MaxwellMaterial(; eps_r = 4.0 - 0.3im, mu_r = 1.5, sigma = 0.02)
    return MaxwellModel(grid; materials = Dict("lossy" => lossy), pec = CUBE_FACES, kwargs...), lossy
end

# Direct complex assembly of the SI operator at ω, independent of the stored blocks.
function direct_operator(model, ω)
    (; eps0, mu0, length_scale) = model
    L = length_scale
    A = allocate_matrix(SparseMatrixCSC{ComplexF64, Int}, model.dh)
    asm = start_assemble(A)
    cv = model.cellvalues
    n = getnbasefunctions(cv)
    Ae = zeros(ComplexF64, n, n)
    for cc in CellIterator(model.dh)
        reinit!(cv, cc)
        (; eps_r, mu_r, sigma) = model.materials[cellid(cc)]
        fill!(Ae, 0)
        for q in 1:getnquadpoints(cv), i in 1:n, j in 1:n
            dΩ = getdetJdV(cv, q)
            Ni, Nj = shape_value(cv, q, i), shape_value(cv, q, j)
            Ci, Cj = shape_curl(cv, q, i), shape_curl(cv, q, j)
            # Mesh lengths are length_scale metres: curl scales by 1/L and volumes by L^3, divided through by L.
            Ae[i, j] += (Ci ⋅ Cj / mu_r + L^2 * (im * ω * mu0 * sigma - ω^2 * mu0 * eps0 * eps_r) * (Ni ⋅ Nj)) * dΩ
        end
        assemble!(asm, celldofs(cc), Ae)
    end
    return A
end

@testset "A(ω) by nzval combination equals direct assembly" begin
    model, _ = two_material_model(3; length_scale = 1.0e-2)
    for ω in (2π * 1.0e9, 2π * 7.3e9)
        A = maxwell_operator(model, ω)
        Ad = direct_operator(model, ω)
        @test A.colptr == Ad.colptr && A.rowval == Ad.rowval
        @test nonzeros(A) ≈ nonzeros(Ad) rtol = 1.0e-12
        @test transpose(A) == A
        # In place on the same pattern.
        B = maxwell_operator(model, 1.0)
        nz = nonzeros(B)
        maxwell_operator!(B, model, ω)
        @test nonzeros(B) === nz && B == A
    end
    # Integer frequencies are promoted before squaring: (10^10)^2 overflows Int64.
    @test maxwell_operator(model, 10_000_000_000) == maxwell_operator(model, 1.0e10)
    @test FiniteElementMethod.maxwell_rhs(model, 10_000_000_000, x -> [1.0, 0.0, 0.0]) ==
        FiniteElementMethod.maxwell_rhs(model, 1.0e10, x -> [1.0, 0.0, 0.0])
    @test solve_driven(model, 10_000_000_000, x -> [1.0, 0.0, 0.0]).u ==
        solve_driven(model, 1.0e10, x -> [1.0, 0.0, 0.0]).u
    # A further term with a general coefficient function, as boundary blocks are added.
    extra = copy(model.blocks.sigma)
    nonzeros(extra) .= rand(nnz(extra))
    c(ω) = im * sqrt(ω) - 2.0
    A0 = maxwell_operator(model, 5.0)
    FiniteElementMethod.add_term!(model, extra, c)
    @test maxwell_operator(model, 5.0) ≈ A0 + c(5.0) * extra
    @test_throws "sparsity pattern" FiniteElementMethod.add_term!(model, sprand(size(extra)..., 0.01), c)
end

@testset "materials" begin
    model, lossy = two_material_model(2)
    for cc in CellIterator(model.dh)
        z = sum(x -> x[3], getcoordinates(cc)) / 4
        @test model.materials[cellid(cc)] == (z <= 0.5 ? lossy : MaxwellMaterial())
    end
    grid = unit_cube(2)
    addcellset!(grid, "a", x -> x[1] <= 0.5)
    addcellset!(grid, "b", x -> x[2] <= 0.5)
    @test_throws "in more than one material cell set" MaxwellModel(
        grid; materials = Dict("a" => MaxwellMaterial(), "b" => MaxwellMaterial(; eps_r = 2))
    )
    @test_throws "no cell set \"nope\"" MaxwellModel(grid; materials = Dict("nope" => MaxwellMaterial()))
    @test_throws "no facet set \"nope\"" MaxwellModel(grid; pec = ["nope"])
    @test_throws "Tetrahedron" MaxwellModel(generate_grid(Hexahedron, (2, 2, 2)))
end

@testset "PEC: tangential E vanishes on PEC facets" begin
    model, _ = two_material_model(4; eps0 = 1.0, mu0 = 1.0)
    # A current that does not itself vanish at the walls.
    sol = solve_driven(model, 3.0, x -> [1.0, x[1], x[2] * x[3]])
    @test maximum(abs, sol.u) > 1.0e-3
    ip = Nedelec{RefTetrahedron, 1}()
    fv = FacetValues(FacetQuadratureRule{RefTetrahedron}(3), ip)
    worst = 0.0
    for name in CUBE_FACES, fc in FacetIterator(model.dh, getfacetset(model.grid, name))
        reinit!(fv, fc)
        u_cell = sol.u[celldofs(fc)]
        for q in 1:getnquadpoints(fv)
            n = getnormal(fv, q)
            worst = max(worst, norm(collect(n × function_value(fv, q, u_cell))))
        end
    end
    @test worst < 1.0e-12 * maximum(abs, sol.u)
end

@testset "point evaluation" begin
    grid = unit_cube(8)
    model = MaxwellModel(grid; pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
    f(x) = curlcurl_exact(E_smooth, x) .- E_smooth(x)
    sol = solve_driven(model, 1.0, x -> im .* f(x))
    points = [Vec(0.31, 0.47, 0.62), Vec(0.5, 0.25, 0.75), Vec(2.0, 0.0, 0.0)]
    vals = field_values(sol, points)
    for k in 1:2
        @test norm(collect(vals.E[k]) - E_smooth(points[k])) < 0.1 * norm(E_smooth(points[k]))
        @test norm(collect(vals.curlE[k]) - curl_exact(E_smooth, points[k])) < 0.3 * norm(curl_exact(E_smooth, points[k]))
    end
    @test all(isnan, vals.E[3]) && all(isnan, vals.curlE[3])
    # SI curl with a length scale: the same dofs on a mesh in centimetres have 100 times the curl.
    cm = MaxwellModel(unit_cube(8); pec = CUBE_FACES, length_scale = 1.0e-2)
    vcm = field_values(MaxwellSolution(cm, 1.0, sol.u), points[1:1])
    @test collect(vcm.E[1]) ≈ collect(vals.E[1])
    @test collect(vcm.curlE[1]) ≈ 100 .* collect(vals.curlE[1])
end

@testset "LinearProblem path" begin
    model, _ = two_material_model(3; eps0 = 1.0, mu0 = 1.0)
    J(x) = [0.0, 0.0, 1.0]
    prob = driven_problem(model, 2.0, J)
    @test prob isa FiniteElementMethod.SciMLBase.LinearProblem
    @test prob.A \ prob.b ≈ solve_driven(model, 2.0, J).u
    # Per-cell-set constant currents equal the function form. With 4 cells per side, the cells of "src" are
    # exactly those with every quadrature point at x < 0.5.
    model2, _ = two_material_model(4; eps0 = 1.0, mu0 = 1.0)
    addcellset!(model2.grid, "src", x -> x[1] <= 0.5)
    Jfun(x) = x[1] < 0.5 ? [0.0, 1.0, 0.0] : [0.0, 0.0, 0.0]
    b = FiniteElementMethod.maxwell_rhs(model2, 2.0, Dict("src" => [0.0, 1.0, 0.0]))
    @test norm(b) > 0
    @test b ≈ FiniteElementMethod.maxwell_rhs(model2, 2.0, Jfun)
    @test_throws "no cell set \"nope\"" FiniteElementMethod.maxwell_rhs(model2, 2.0, Dict("nope" => [1, 0, 0]))
    # The SI load: -iωμ0 L² ∫ J·v.
    m = MaxwellModel(unit_cube(2); length_scale = 0.5)
    @test FiniteElementMethod.maxwell_rhs(m, 3.0, J) ≈
        -im * 3.0 * FiniteElementMethod.MU0 * 0.25 .* FiniteElementMethod.maxwell_rhs(MaxwellModel(unit_cube(2); mu0 = 1.0), 1.0, x -> im .* J(x))
end

@testset "point evaluation finds points the nearest-node search misses" begin
    nodes = Node.([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.35, 0.35, 0.35), (0.36, 0.35, 0.35), (0.35, 0.36, 0.35), (0.35, 0.35, 0.36)])
    grid = Grid(Tetrahedron.([(1, 2, 3, 4), (5, 6, 7, 8)]), nodes)
    model = MaxwellModel(grid; eps0 = 1.0, mu0 = 1.0)
    E0 = ComplexF64[1 + 0.2im, -2 + 0.3im, 3 - 0.4im]
    sol = solve_driven(model, 2.0, x -> -2im .* E0)
    vals = field_values(sol, [Vec(0.3, 0.3, 0.3), Vec(5.0, 5.0, 5.0)])
    @test collect(vals.E[1]) ≈ E0 atol = 1.0e-12
    @test all(isnan, vals.E[2]) && all(isnan, vals.curlE[2])
end

@testset "PEC facet edge dofs match Ferrite's tetrahedron table" begin
    @test FiniteElementMethod._TET_FACET_EDGE_DOFS == Ferrite.dirichlet_facetdof_indices(FiniteElementMethod._NEDELEC)
end
