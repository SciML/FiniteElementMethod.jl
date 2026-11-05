# Lumped ports against transmission-line closed forms (exp(+iωt)): a parallel-plate line with PMC sides,
# x ∈ [0, W], y ∈ [0, h], z ∈ [0, len] mm, PEC plates at y = 0 and y = h, carries an exact TEM wave of
# impedance Zc = η0 h / (W sqrt(εr)), and lumped ports across the plates (direction +y) are its terminals.
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Test

const ETA0 = FiniteElementMethod.MU0 * FiniteElementMethod.C0
const F3 = 2π * 3.0e9
const PLATE_W, PLATE_H, PLATE_LEN = 10.0, 5.0, 20.0  # mm

function plates(hz; ncross = 2, len = PLATE_LEN, Z0 = (50.0, 50.0), top = :port, materials = Dict{String, MaxwellMaterial}())
    grid = generate_grid(Tetrahedron, (ncross, ncross, ceil(Int, len / hz)), Vec(0.0, 0.0, 0.0), Vec(PLATE_W, PLATE_H, len))
    isempty(materials) || addcellset!(grid, "fill", x -> true)
    ports = [LumpedPort("bottom"; direction = (0, 1, 0), Z0 = Z0[1])]
    top === :port && push!(ports, LumpedPort("top"; direction = (0, 1, 0), Z0 = Z0[2]))
    pec = top === :pec ? ["front", "back", "top"] : ["front", "back"]
    return MaxwellModel(grid; pec, ports, length_scale = 1.0e-3, materials)
end

# ABCD matrix of a line of length `len` mm with relative permittivity `eps_r` (complex for loss).
function line_abcd(ω, len, eps_r = 1.0)
    θ = ω / FiniteElementMethod.C0 * sqrt(complex(eps_r)) * len * 1.0e-3
    Zc = ETA0 * PLATE_H / (PLATE_W * sqrt(complex(eps_r)))
    return [cos(θ) im * Zc * sin(θ); im * sin(θ) / Zc cos(θ)]
end
# Scattering matrix of a two-port ABCD matrix with real reference impedances Z0.
function abcd_to_s(T, Z0)
    (A, B), (C, D) = T[1, :], T[2, :]
    Z = [A / C (A * D - B * C) / C; 1 / C D / C]
    G = Diagonal(collect(float.(Z0)))
    return sqrt(G) \ (Z - G) / (Z + G) * sqrt(G)
end

lumped_S(model, ω) = solve_sweep(model, [ω]).S[:, :, 1]
passive(S) = maximum(eigvals(Hermitian(S' * S))) <= 1 + 1.0e-8

@testset "transmission-line closed forms of the self-check" begin
    # The numbers quoted for W = 10 mm, h = 5 mm, Z0 = 50 Ω, L = 20 mm at 3 GHz.
    Zc = ETA0 * PLATE_H / PLATE_W
    @test Zc ≈ 188.365 atol = 1.0e-3
    S = abcd_to_s(line_abcd(F3, PLATE_LEN), (50, 50))
    @test S[1, 1] ≈ 0.84651 + 0.136im atol = 1.0e-5
    @test S[2, 1] ≈ 0.08165 - 0.5082im atol = 1.0e-5
    θ = F3 / FiniteElementMethod.C0 * PLATE_LEN * 1.0e-3
    Γ, P = (Zc - 50) / (Zc + 50), exp(-im * θ)
    @test S[1, 1] ≈ Γ * (1 - P^2) / (1 - Γ^2 * P^2) && S[2, 1] ≈ P * (1 - Γ^2) / (1 - Γ^2 * P^2)
    @test im * Zc * tan(θ) ≈ 581.45im atol = 1.0e-2
end

@testset "parallel-plate two-port" begin
    expected = abcd_to_s(line_abcd(F3, PLATE_LEN), (50, 50))
    errs = Float64[]
    # Refined in all three directions: the dispersion of the tetrahedra depends on their transverse size too.
    for (ncross, hz) in ((2, 4.0), (4, 2.0), (8, 1.0))
        model = plates(hz; ncross)
        S = lumped_S(model, F3)
        push!(errs, maximum(abs, S - expected))
        @info "PMC parallel plates, lumped ports" hz ndofs(model.dh) S[1, 1] S[2, 1] errs[end]
        @test S[1, 2] ≈ S[2, 1] rtol = 1.0e-8
        @test S[1, 1] ≈ S[2, 2] rtol = 1.0e-6
        @test passive(S)
        # Lossless: the two Z0 loads take all the source power.
        @test abs2(S[1, 1]) + abs2(S[2, 1]) ≈ 1 atol = 1.0e-8
    end
    @test errs[end] < 2.0e-4
    @test errs[1] / errs[3] > 10  # dispersion error O(h²)
    # Z = G^½(I + S)(I - S)⁻¹G^½ recovers the line's open-circuit impedances, -iZc cot θ and -iZc / sin θ.
    model = plates(1.0)
    T = line_abcd(F3, PLATE_LEN)
    Z = FiniteElementMethod.z_parameters(model, lumped_S(model, F3))
    @test Z ≈ [T[1, 1] / T[2, 1] 1 / T[2, 1]; 1 / T[2, 1] T[2, 2] / T[2, 1]] rtol = 3.0e-3
end

@testset "matched and short-circuited lines" begin
    # Ports referred to Zc: no reflection, transmission exp(-ik0 L).
    Zc = ETA0 * PLATE_H / PLATE_W
    model = plates(1.0; Z0 = (Zc, Zc))
    sweep = solve_sweep(model, [F3]; fields = true)
    S = sweep.S[:, :, 1]
    @info "matched parallel plates" S
    @test abs(S[1, 1]) < 2.0e-3 && abs(S[2, 2]) < 2.0e-3
    @test S[2, 1] ≈ exp(-im * F3 / FiniteElementMethod.C0 * PLATE_LEN * 1.0e-3) atol = 2.0e-3
    # The driven port sees Zc: V = Zc I_in with I_in = 1 A - V/Z0 = 1/2 A.
    u = sweep.fields[1][:, 1]
    V = FiniteElementMethod.port_voltage(model, 1, u)
    @test V ≈ Zc / 2 rtol = 2.0e-3
    @test FiniteElementMethod.port_current(model, 1, u; source = 1) ≈ 0.5 rtol = 2.0e-3
    @test abs(FiniteElementMethod.port_voltage(model, 2, u)) ≈ Zc / 2 rtol = 2.0e-3
    # One port, PEC short at z = L: Zin = iZc tan(k0 L).
    model = plates(1.0; top = :pec)
    Zin = only(FiniteElementMethod.z_parameters(model, lumped_S(model, F3)))
    @info "shorted parallel plates" Zin
    @test Zin ≈ im * Zc * tan(F3 / FiniteElementMethod.C0 * PLATE_LEN * 1.0e-3) rtol = 5.0e-3
end

@testset "lossy line with unequal reference impedances" begin
    lossy = MaxwellMaterial(; eps_r = 2.2 - 0.3im, sigma = 0.05)
    model = plates(1.0; Z0 = (50.0, 75.0), materials = Dict("fill" => lossy))
    S = lumped_S(model, F3)
    eps_eff = lossy.eps_r - im * lossy.sigma / (F3 * FiniteElementMethod.EPS0)
    expected = abcd_to_s(line_abcd(F3, PLATE_LEN, eps_eff), (50, 75))
    @info "lossy parallel plates" S expected
    @test S[1, 2] ≈ S[2, 1] rtol = 1.0e-8
    @test passive(S) && maximum(eigvals(Hermitian(S' * S))) < 0.9
    @test S ≈ expected atol = 5.0e-3
end

@testset "lumped port inside the mesh" begin
    # Port 2 on the plane z = L inside the line, listed from the cells on both sides; beyond it an open stub
    # of length d (the natural condition at z = L + d is a PMC: an open circuit) in parallel with its load.
    len, d, hz = PLATE_LEN, 6.0, 1.0
    grid = generate_grid(Tetrahedron, (2, 2, round(Int, (len + d) / hz)), Vec(0.0, 0.0, 0.0), Vec(PLATE_W, PLATE_H, len + d))
    plane = FacetIndex[]
    for cell in 1:getncells(grid), (k, facet) in pairs(Ferrite.reference_facets(RefTetrahedron))
        x = getcoordinates(grid, cell)
        all(i -> x[i][3] ≈ len, facet) && push!(plane, FacetIndex(cell, k))
    end
    addfacetset!(grid, "gap", plane)
    ports = [LumpedPort("bottom"; direction = (0, 1, 0)), LumpedPort("gap"; direction = (0, 1, 0))]
    model = MaxwellModel(grid; pec = ["front", "back"], ports, length_scale = 1.0e-3)
    gap = model.ports[2]
    @test length(plane) == 2 * 8  # 2 × 2 squares of 2 triangles, each from both sides
    @test gap.length ≈ PLATE_H && gap.width ≈ PLATE_W && abs(gap.normal[3]) ≈ 1
    @test length(FiniteElementMethod._unique_facets(grid, plane)) == 8
    S = lumped_S(model, F3)
    Zc = ETA0 * PLATE_H / PLATE_W
    Ystub = im * tan(F3 / FiniteElementMethod.C0 * d * 1.0e-3) / Zc
    expected = abcd_to_s(line_abcd(F3, len) * [1 0; Ystub 1], (50, 50))
    @info "interior lumped port with an open stub" S expected
    @test S[1, 2] ≈ S[2, 1] rtol = 1.0e-8
    @test S ≈ expected atol = 3.0e-3
end

@testset "mixed wave and lumped ports in one sweep" begin
    # WR-90, 30 mm long: the TE10 wave port at z = 0, a lumped port across the guide (along y) at z = L.
    a, b, len = 22.86, 10.16, 30.0
    grid = generate_grid(Tetrahedron, (8, 4, 15), Vec(0.0, 0.0, 0.0), Vec(a, b, len))
    ports = [WavePort("bottom"; a_direction = (1, 0, 0)), LumpedPort("top"; direction = (0, 1, 0), Z0 = 100.0)]
    model = MaxwellModel(grid; pec = ["left", "right", "front", "back"], ports, length_scale = 1.0e-3)
    ωs = 2π .* [9.0e9, 10.0e9, 11.0e9]
    calls = Symbol[]
    function counting(A, B, cache)
        push!(calls, cache === nothing ? :lu : :refactor)
        return FiniteElementMethod.umfpack_solver(A, B, cache)
    end
    sweep = solve_sweep(model, ωs; fields = true, solver = counting)
    @test calls == [:lu, :refactor, :refactor]
    for (k, ω) in pairs(ωs)
        S = sweep.S[:, :, k]
        @info "WR-90 wave port to lumped port" ω / 2π S abs2(S[1, 1]) + abs2(S[2, 1])
        # Reciprocity holds only if both kinds of port are normalised to the same power waves.
        @test S[1, 2] ≈ S[2, 1] rtol = 1.0e-8
        @test passive(S)
        @test abs2(S[1, 1]) + abs2(S[2, 1]) ≈ 1 atol = 1.0e-2
        # Every column from the one factorisation equals an independent solve.
        A = maxwell_operator(model, ω)
        B = FiniteElementMethod.port_excitations(model, ω)
        for j in axes(B, 2)
            bj = B[:, j]
            Aj = copy(A)
            apply!(Aj, bj, model.ch)
            @test Aj \ bj ≈ sweep.fields[k][:, j] rtol = 1.0e-10
        end
    end
    @test_throws "not a lumped port" FiniteElementMethod.z_parameters(model, sweep.S[:, :, 1])
    @test_throws "not a wave port" FiniteElementMethod.propagation_constant(model, 2, ωs[1])
end

@testset "lumped port arguments" begin
    grid = generate_grid(Tetrahedron, (2, 2, 2))
    @test_throws "must be positive" LumpedPort("bottom"; direction = (0, 1, 0), Z0 = 0)
    @test_throws "not in the port plane" MaxwellModel(grid; ports = [LumpedPort("bottom"; direction = (0, 0, 1))])
    @test_throws "not a rectangle" MaxwellModel(grid; ports = [LumpedPort("bottom"; direction = (1, 1, 0))])
    @test_throws "no facet set" MaxwellModel(grid; ports = [LumpedPort("nope"; direction = (0, 1, 0))])
    # Lumped ports widen the pattern; every block and further term shares it.
    model = MaxwellModel(grid; ports = [LumpedPort("bottom"; direction = (0, 1, 0))])
    @test nnz(model.blocks.curlcurl) > nnz(allocate_matrix(model.dh))
    @test model.ports[1].block.colptr == model.blocks.sigma.colptr
    @test FiniteElementMethod.tangential_mass(model, ["top"]).colptr == model.blocks.curlcurl.colptr
end
