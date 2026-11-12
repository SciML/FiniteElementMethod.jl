# Eigenmodes of a PEC rectangular cavity against the analytic TE/TM frequencies.
using FiniteElementMethod, Ferrite, KrylovKit, LinearAlgebra, Test
include(joinpath(@__DIR__, "..", "shared", "maxwell.jl"))

const BOX = (1.0, 0.5, 0.75)

# f_mnp = (c/2) sqrt((m/a)^2 + (n/b)^2 + (p/d)^2): TE needs p >= 1 and at most one of m, n zero; TM needs m, n >= 1.
function cavity_frequencies(a, b, d; c = 1.0, count = 15)
    f = Float64[]
    for m in 0:6, n in 0:6, p in 0:6
        fmnp = c / 2 * sqrt((m / a)^2 + (n / b)^2 + (p / d)^2)
        p >= 1 && (m == 0) + (n == 0) <= 1 && push!(f, fmnp)
        m >= 1 && n >= 1 && push!(f, fmnp)
    end
    return sort!(f)[1:count]
end

# k cells per quarter unit along x, so the mesh size is h = 1 / (4k) in every direction.
box_grid(k; scale = 1.0) = generate_grid(Tetrahedron, (4k, 2k, 3k), Vec(0.0, 0.0, 0.0), Vec((scale .* BOX)...))
box_model(k; kwargs...) = MaxwellModel(box_grid(k); pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0, kwargs...)

const EXACT = cavity_frequencies(BOX...)

@testset "analytic cavity list" begin
    @test round.(EXACT; digits = 4) == [
        0.8333, 1.118, 1.2019, 1.2019, 1.3017, 1.3017, 1.4142, 1.424, 1.5635, 1.5635, 1.6415, 1.6667, 1.6667,
        1.7401, 1.7401,
    ]
end

@testset "PEC box: 15 modes with multiplicities, O(h^2) convergence, no spurious modes" begin
    ks = (3, 4, 6)
    computed = map(ks) do k
        # A target below the first mode: the k = 0 gradient null space is closer to the shift than any physical
        # mode, so this fails unless it is deflated.
        modes = maxwell_eigenmodes(box_model(k); nev = length(EXACT), target_frequency = 0.4)
        @test length(modes.frequency) == length(EXACT)
        @test issorted(modes.frequency)
        @test all(isinf, modes.Q)
        modes.frequency
    end
    errs = [abs.(f .- EXACT) ./ EXACT for f in computed]
    @test maximum(errs[end]) < 0.01
    hs = [1 / (4k) for k in ks]
    @test minimum(rates(first.(errs), hs)) >= 1.7
    @test minimum(rates(sum.(errs) ./ length(EXACT), hs)) >= 1.7
    # Nothing between the filtered null space and the lowest physical mode, at every refinement.
    @test all(f -> f[1] > 0.95 * EXACT[1], computed)
    for (k, f, e) in zip(ks, computed, errs)
        println("k = $k: ", join(string.(round.(f; digits = 5)), " "), "  max rel err ", maximum(e))
    end
end

@testset "a target among the modes returns those nearest to it" begin
    modes = maxwell_eigenmodes(box_model(3); nev = 4, target_frequency = 1.3)
    @test length(modes.frequency) == 4
    # The four exact modes nearest 1.3: 1.2019 (x2) and 1.3017 (x2).
    @test modes.frequency ≈ EXACT[3:6] rtol = 0.02
    # Nearest in frequency, not in ω²: from 0.98, TM110 (1.118, 0.138 away) beats TE101 (0.833, 0.147 away).
    @test maxwell_eigenmodes(box_model(3); nev = 1, target_frequency = 0.98).frequency[1] ≈ EXACT[2] rtol = 0.01
end

@testset "null-space cutoff: not relative to a high target, and the search widens past what it drops" begin
    # box_model(1) has 93 modes outside the gradient null space; the lowest is 0.83, below 1e-3 of the target.
    small = box_model(1)
    high = maxwell_eigenmodes(small; nev = 93, target_frequency = 1000.0)
    @test length(high.frequency) == 93
    @test high.frequency[1] ≈ maxwell_eigenmodes(small; nev = 1, target_frequency = 0.9).frequency[1] rtol = 1.0e-8
    # The cutoff is relative to the spectrum scale ‖K‖₁/‖M‖₁ = 1412.4 of this model; nullspace_tol = 0.0537 makes it
    # 75.85 = 12σ of a target of 0.4, which drops every mode below 1.39 Hz: the 2nev = 6 nearest candidates are all
    # dropped, and the search must go on to find the next three rather than stop with fewer.
    model = box_model(2)
    krylov = maxwell_eigenmodes(model; nev = 3, target_frequency = 0.4, nullspace_tol = 0.0537)
    dense = maxwell_eigenmodes(model; nev = 3, target_frequency = 0.4, nullspace_tol = 0.0537, krylovdim = 10_000)
    @test length(krylov.frequency) == 3
    @test all(>(sqrt(12) * 0.4), krylov.frequency)
    @test krylov.frequency ≈ dense.frequency rtol = 1.0e-8
end

@testset "null-space cutoff: static modes are rejected for a low target too" begin
    grid = generate_grid(Tetrahedron, (2, 2, 2), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    model = MaxwellModel(grid; pec = ["left", "right"], eps0 = 1.0, mu0 = 1.0)
    # First three nonzero eigenfrequencies of the generalised problem K u = λ M u on the free dofs.
    truth = [0.4695153184, 0.4907674018, 0.6775122733]
    krylov = maxwell_eigenmodes(model; nev = 3, target_frequency = 1.0e-5)
    dense = maxwell_eigenmodes(model; nev = 3, target_frequency = 1.0e-7, krylovdim = 10_000)
    @test krylov.frequency ≈ truth rtol = 1.0e-8
    @test dense.frequency ≈ truth rtol = 1.0e-8
end

@testset "a target exactly on a discrete resonance" begin
    # One free edge dof with K / M = 20 = mu0 eps0 ω², so the shifted matrix is exactly [0.0].
    model = MaxwellModel(
        generate_grid(Tetrahedron, (1, 1, 1), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0));
        pec = CUBE_FACES, eps0 = 20.0, mu0 = 1.0
    )
    modes = maxwell_eigenmodes(model; nev = 1, target_frequency = 1 / (2π))
    @test modes.frequency[1] ≈ 1 / (2π) rtol = 1.0e-6
end

@testset "strongly contrasting materials are warned about" begin
    grid = unit_cube(3)
    addcellset!(grid, "left", x -> x[1] <= 0.5)
    model = MaxwellModel(grid; materials = Dict("left" => MaxwellMaterial(; mu_r = 1.0e-12)), pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
    @test_logs (:warn, r"differ by a factor 1\.0e12") match_mode = :any maxwell_eigenmodes(model; nev = 1, target_frequency = 0.5)
    @test_logs min_level = Base.CoreLogging.Warn maxwell_eigenmodes(box_model(2); nev = 1, target_frequency = 1.0)
end

@testset "physical modes below the material-warning cutoff are kept" begin
    # mu_r = 5e-6 on half the cube puts ‖K‖₁/‖M‖₁ near 3.5e7, so a cutoff of 1e-6 of it (35.4) is above the first
    # three physical eigenvalues, which are still resolved: the generalised eigenvalues are checked against a dense solve.
    grid = unit_cube(3)
    addcellset!(grid, "left", x -> x[1] <= 0.5)
    model = MaxwellModel(grid; materials = Dict("left" => MaxwellMaterial(; mu_r = 5.0e-6)), pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
    free = setdiff(1:ndofs(model.dh), FiniteElementMethod._pec_dofs(model.dh, model.grid, model.pec))
    λ = eigvals(Symmetric(Matrix(model.blocks.curlcurl[free, free])), Symmetric(Matrix(model.blocks.eps_real[free, free])))
    λ_physical = filter(>(1.0e4 * eps() * maximum(λ)), λ)
    @test λ_physical[1:3] ≈ [22.48440197, 29.81579099, 32.07272248] rtol = 1.0e-7
    truth = sqrt(λ_physical[1]) / 2π
    for kwargs in ((;), (; krylovdim = 10_000))
        found = maxwell_eigenmodes(model; nev = 1, target_frequency = 0.5, kwargs...)
        @test found.frequency[1] ≈ truth rtol = 1.0e-8
    end
end

@testset "repeated eigenvalues of disconnected cavities keep their multiplicity" begin
    # 32 disconnected PEC unit cubes of one free edge each, with εr cycling through 1, 2, 4, 8: the eigenvalues are
    # 20/εr, eight times each, and the nearest to a low target is 20/8 (0.25164606 Hz) eight times over.
    cube = generate_grid(Tetrahedron, (1, 1, 1), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    ncell, nnode = length(cube.cells), length(cube.nodes)
    nodes = [Node(Vec(x.x[1] + 2i, x.x[2], x.x[3])) for i in 0:31 for x in cube.nodes]
    cells = [Tetrahedron(ntuple(k -> c.nodes[k] + nnode * i, 4)) for i in 0:31 for c in cube.cells]
    grid = Grid(cells, nodes)
    for (k, eps_r) in enumerate((1.0, 2.0, 4.0, 8.0))
        addcellset!(grid, "eps$k", Set(ncell * i + c for i in 0:31 if mod(i, 4) == k - 1 for c in 1:ncell))
    end
    rim = union(values(cube.facetsets)...)  # the six outer faces of one cube
    addfacetset!(grid, "wall", Set(FacetIndex(ncell * i + f[1], f[2]) for i in 0:31 for f in rim))
    materials = Dict("eps$k" => MaxwellMaterial(; eps_r) for (k, eps_r) in enumerate((1.0, 2.0, 4.0, 8.0)))
    model = MaxwellModel(grid; materials, pec = ["wall"], eps0 = 1.0, mu0 = 1.0)
    f(eps_r) = sqrt(20 / eps_r) / 2π
    found = maxwell_eigenmodes(model; nev = 2, target_frequency = 0.01)
    @test found.frequency ≈ fill(f(8), 2) rtol = 1.0e-8
    @test maxwell_eigenmodes(model; nev = 3, target_frequency = 0.01).frequency ≈ fill(f(8), 3) rtol = 1.0e-8
    @test maxwell_eigenmodes(model; nev = 9, target_frequency = 0.01).frequency ≈ [fill(f(8), 8); f(4)] rtol = 1.0e-8
    @test issorted(maxwell_eigenmodes(model; nev = 12, target_frequency = 0.01).frequency)
end

@testset "restarts widen past the discarded static modes like the first search" begin
    # 64 disconnected one-free-edge cubes with the eigenvalues 1..6 once, 10 and 20 twice and 54 distinct ones above, and
    # a cutoff of 8: the six lowest are discarded, so a restart asking for a few candidates may see only those.
    eigenvalues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 10.0, 20.0, 20.0, (40.0 + 3i for i in 1:54)...]
    cube = generate_grid(Tetrahedron, (1, 1, 1), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    ncell, nnode = length(cube.cells), length(cube.nodes)
    ncubes = length(eigenvalues)
    grid = Grid(
        [Tetrahedron(ntuple(k -> c.nodes[k] + nnode * i, 4)) for i in 0:(ncubes - 1) for c in cube.cells],
        [Node(Vec(x.x[1] + 2i, x.x[2], x.x[3])) for i in 0:(ncubes - 1) for x in cube.nodes],
    )
    rim = union(values(cube.facetsets)...)
    addfacetset!(grid, "wall", Set(FacetIndex(ncell * i + f[1], f[2]) for i in 0:(ncubes - 1) for f in rim))
    materials = Dict{String, MaxwellMaterial}()
    for λ in unique(eigenvalues)
        addcellset!(grid, "λ$λ", Set(ncell * (i - 1) + c for (i, v) in enumerate(eigenvalues) if v == λ for c in 1:ncell))
        materials["λ$λ"] = MaxwellMaterial(; eps_r = 20 / λ)
    end
    model = MaxwellModel(grid; materials, pec = ["wall"], eps0 = 1.0, mu0 = 1.0)
    free = setdiff(1:ndofs(model.dh), FiniteElementMethod._pec_dofs(model.dh, model.grid, model.pec))
    K, M = model.blocks.curlcurl[free, free], model.blocks.eps_real[free, free]
    nullspace_tol = 8 / (opnorm(K, 1) / opnorm(M, 1))
    f(λ) = sqrt(λ) / 2π
    found = maxwell_eigenmodes(model; nev = 2, target_frequency = 0.001, nullspace_tol)
    @test found.frequency ≈ fill(f(10.0), 2) rtol = 1.0e-8
    @test maxwell_eigenmodes(model; nev = 3, target_frequency = 0.001, nullspace_tol).frequency ≈ [f(10.0), f(10.0), f(20.0)] rtol = 1.0e-8
end

@testset "restart deflation works with an indefinite mass form" begin
    basis = FiniteElementMethod._orthonormal_basis
    # A negative mass norm, an isotropic vector (zero norm), a vector already in the span and a complex one.
    B, U = basis([[1.0]], reshape([-1.0], 1, 1))
    @test transpose(U) * B ≈ [1.0;;]
    M = [1.0 0.0; 0.0 -1.0]
    B, U = basis([[1.0, 1.0], [2.0, 2.0], [1.0, -1.0]], M)
    @test size(B, 2) == 2 && transpose(U) * B ≈ [1.0 0.0; 0.0 1.0]
    @test B - B * (transpose(U) * B) ≈ zeros(2, 2)
    B, U = basis([[1.0 + 0.5im, 2.0]], [-2.0 0.0; 0.0 1.0])
    @test transpose(U) * B ≈ [1.0;;]
    # Lossless negative-permittivity cavities: 63 cubes of eigenvalues 2..64 and one with εr = -200 (λ = -0.1).
    cube = generate_grid(Tetrahedron, (1, 1, 1), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    ncell, nnode = length(cube.cells), length(cube.nodes)
    epss = [-200.0; [20 / λ for λ in 2.0:64.0]]
    ncubes = length(epss)
    grid = Grid(
        [Tetrahedron(ntuple(k -> c.nodes[k] + nnode * i, 4)) for i in 0:(ncubes - 1) for c in cube.cells],
        [Node(Vec(x.x[1] + 2i, x.x[2], x.x[3])) for i in 0:(ncubes - 1) for x in cube.nodes],
    )
    rim = union(values(cube.facetsets)...)
    addfacetset!(grid, "wall", Set(FacetIndex(ncell * i + f[1], f[2]) for i in 0:(ncubes - 1) for f in rim))
    materials = Dict{String, MaxwellMaterial}()
    for (i, eps_r) in enumerate(epss)
        addcellset!(grid, "cube$i", Set(ncell * (i - 1) + c for c in 1:ncell))
        materials["cube$i"] = MaxwellMaterial(; eps_r)
    end
    model = MaxwellModel(grid; materials, pec = ["wall"], eps0 = 1.0, mu0 = 1.0)
    free = setdiff(1:ndofs(model.dh), FiniteElementMethod._pec_dofs(model.dh, model.grid, model.pec))
    λ = real.(eigvals(Matrix(model.blocks.curlcurl[free, free]), Matrix(model.blocks.eps_real[free, free])))
    ω = sqrt.(complex.(sort(λ)))
    target = 0.01
    nearest = sort(ω; by = x -> abs(x - 2π * target))
    for nev in 1:3
        found = maxwell_eigenmodes(model; nev, target_frequency = target)
        @test sort(abs.(found.omega .- 2π * target)) ≈ sort(abs.(nearest[1:nev] .- 2π * target)) rtol = 1.0e-8
    end
end

@testset "lossy: nearest in complex ω, and the candidate bound holds" begin
    # eps_r = 1 - 100i scales ω by 1/√(1 - 100i): the lowest modes sit at Re f = 0.0592, 0.0794, 0.0854 with
    # Im f about as large. From 0.0854 the nearest in complex ω is the first, and shift-invert ranks the first
    # two ahead of the third in λ, so a bound on Re ω alone would stop too early.
    grid = box_grid(2)
    addcellset!(grid, "fill", x -> true)
    model = MaxwellModel(
        grid; materials = Dict("fill" => MaxwellMaterial(; eps_r = 1 - 100im)), pec = CUBE_FACES,
        eps0 = 1.0, mu0 = 1.0,
    )
    ωt = 2π * 0.085405
    one = maxwell_eigenmodes(model; nev = 1, target_frequency = 0.085405)
    six = maxwell_eigenmodes(model; nev = 6, target_frequency = 0.085405)
    @test issorted(six.frequency)
    @test one.omega[1] ≈ six.omega[argmin(abs.(six.omega .- ωt))]
    @test one.omega[1] ≈ 2π * EXACT[1] / sqrt(1 - 100im) rtol = 0.02
end

@testset "no PEC: the constant potential is removed from the gradient basis" begin
    # Natural (magnetic-wall) conditions everywhere: the dual of the PEC cube, lowest at √2/2 (x3).
    single = MaxwellModel(generate_grid(Tetrahedron, (1, 1, 1)); eps0 = 1.0, mu0 = 1.0)
    @test length(maxwell_eigenmodes(single; nev = 1, target_frequency = 0.4).frequency) == 1
    f = maxwell_eigenmodes(MaxwellModel(unit_cube(4); eps0 = 1.0, mu0 = 1.0); nev = 3, target_frequency = 0.4).frequency
    @test f ≈ fill(sqrt(2) / 2, 3) rtol = 0.02
    # PEC on one face only: the potential is pinned there, so no node is removed and the basis is still full.
    f = maxwell_eigenmodes(MaxwellModel(unit_cube(4); pec = ["left"], eps0 = 1.0, mu0 = 1.0); nev = 2, target_frequency = 0.4).frequency
    @test length(f) == 2 && all(>(0.1), f)
end

@testset "eigenvectors are fields: PEC trace, field_values, residual" begin
    model = box_model(3)
    modes = maxwell_eigenmodes(model; nev = 1, target_frequency = 0.8)
    sol = modes.modes[1]
    @test sol isa MaxwellSolution
    @test sol.omega ≈ 2π * modes.frequency[1]
    # Zero on the PEC dofs, and A(ω) u = 0 at the eigenfrequency.
    @test all(iszero, sol.u[model.ch.prescribed_dofs])
    A = maxwell_operator(model, sol.omega)
    free = setdiff(1:ndofs(model.dh), model.ch.prescribed_dofs)
    @test norm((A * sol.u)[free]) / norm(model.blocks.curlcurl * sol.u) < 1.0e-6
    # TE101: E along y only, E_y ∝ sin(πx/a) sin(πz/d).
    E = field_values(sol, [Vec(0.5, 0.25, 0.375), Vec(0.25, 0.25, 0.375)]).E
    @test abs(E[1][2]) > 10 * max(abs(E[1][1]), abs(E[1][3]))
    @test abs(E[2][2]) / abs(E[1][2]) ≈ sin(π / 4) rtol = 0.05
end

@testset "uniform dielectric fill scales every frequency by 1/sqrt(eps_r)" begin
    vacuum = maxwell_eigenmodes(box_model(3); nev = 6, target_frequency = 0.4).frequency
    grid = box_grid(3)
    addcellset!(grid, "fill", x -> true)
    filled = MaxwellModel(
        grid; materials = Dict("fill" => MaxwellMaterial(; eps_r = 2.2)), pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0
    )
    f = maxwell_eigenmodes(filled; nev = 6, target_frequency = 0.4 / sqrt(2.2)).frequency
    @test f ./ vacuum ≈ fill(1 / sqrt(2.2), 6) rtol = 1.0e-8
end

@testset "lossy fill: Q = 1/tan δ" begin
    tand = 1.0e-3
    grid = box_grid(3)
    addcellset!(grid, "fill", x -> true)
    lossy = MaxwellModel(
        grid; materials = Dict("fill" => MaxwellMaterial(; eps_r = 2.2 * (1 - im * tand))),
        pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0,
    )
    modes = maxwell_eigenmodes(lossy; nev = 3, target_frequency = 0.5)
    @test all(q -> isapprox(q, 1 / tand; rtol = 0.01), modes.Q)
    @test all(ω -> imag(ω) > 0, modes.omega)  # exp(+iωt) decays
    @test modes.frequency ≈ real.(modes.omega) ./ 2π
end

@testset "SI units: 100 x 50 x 75 mm box" begin
    model = MaxwellModel(box_grid(3; scale = 100.0); pec = CUBE_FACES, length_scale = 1.0e-3)
    f1 = maxwell_eigenmodes(model; nev = 1, target_frequency = 2.0e9).frequency[1]
    @test f1 ≈ FiniteElementMethod.C0 / 2 * sqrt(1 / 0.1^2 + 1 / 0.075^2) rtol = 0.01
    @test round(FiniteElementMethod.C0 / 2 * sqrt(1 / 0.1^2 + 1 / 0.075^2) / 1.0e9; digits = 3) == 2.498
end

@testset "degenerate sizes: no gradient basis, dense fallback, too many modes" begin
    # One PEC hexahedron of six tets: every node is on the PEC boundary, the body diagonal is the one free edge.
    cube = MaxwellModel(generate_grid(Tetrahedron, (1, 1, 1)); pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
    f = maxwell_eigenmodes(cube; nev = 1, target_frequency = 0.4).frequency
    @test length(f) == 1 && isfinite(f[1]) && f[1] > 0
    @test_throws "exceeds the 1 modes" maxwell_eigenmodes(cube; nev = 2, target_frequency = 0.4)
    # A problem smaller than the Krylov dimension is solved densely, and agrees with the Krylov path.
    small = box_model(1)
    dense = maxwell_eigenmodes(small; nev = 4, target_frequency = 0.9, krylovdim = 10_000)
    krylov = maxwell_eigenmodes(small; nev = 4, target_frequency = 0.9, krylovdim = 30)
    @test dense.frequency ≈ krylov.frequency rtol = 1.0e-8
    # A target on a resonance makes one θ huge; the gradient modes are still told apart by their eigenvectors.
    cube2 = MaxwellModel(generate_grid(Tetrahedron, (2, 2, 2)); pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
    f0 = maxwell_eigenmodes(cube2; nev = 3, target_frequency = 0.3, krylovdim = 10_000).frequency
    ft = f0[1] * (1 + 1.0e-10)
    near = maxwell_eigenmodes(cube2; nev = 3, target_frequency = ft, krylovdim = 10_000).frequency
    @test length(near) == 3
    @test near ≈ maxwell_eigenmodes(cube2; nev = 3, target_frequency = ft, krylovdim = 30).frequency rtol = 1.0e-8
    @test near[1] ≈ f0[1] rtol = 1.0e-10
    # A single tetrahedron with every face PEC has no free dof at all.
    nodes = Node.([Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0), Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)])
    tet = Grid([Tetrahedron((1, 2, 3, 4))], nodes)
    addfacetset!(tet, "all", x -> true)
    @test_throws "no modes" maxwell_eigenmodes(MaxwellModel(tet; pec = ["all"]); nev = 1, target_frequency = 1.0)
    @test_throws "nev must be positive" maxwell_eigenmodes(cube; nev = 0, target_frequency = 0.4)
    @test_throws "finite" maxwell_eigenmodes(cube; nev = 1, target_frequency = Inf)
end

@testset "argument checks" begin
    model = box_model(2)
    @test_throws ArgumentError maxwell_eigenmodes(model; nev = 2, target_frequency = 0.0)
    grid = box_grid(2)
    addcellset!(grid, "all", x -> true)
    conductor = MaxwellModel(grid; pec = CUBE_FACES, materials = Dict("all" => MaxwellMaterial(; sigma = 1.0)))
    @test_throws "conductivity" maxwell_eigenmodes(conductor; nev = 1, target_frequency = 1.0e9)
    # Without KrylovKit the eigensolver hook has only its fallback, which names the package to load.
    @test_throws "KrylovKit" FiniteElementMethod._eigsolve(nothing, identity, zeros(2), 1)
end
