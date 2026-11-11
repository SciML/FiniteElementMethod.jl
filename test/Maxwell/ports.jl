# WR-90 waveguide checks of the TE10 wave ports, the first-order absorbing boundary and the frequency sweep,
# against closed forms (exp(+iωt), ports at z = 0 and z = L, broad wall along x).
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Test

const WR90_A, WR90_B = 22.86, 10.16  # mm
const F10 = 2π * 10.0e9
const BETA10 = 158.2383  # rad/m, TE10 at 10 GHz
const K010 = F10 / FiniteElementMethod.C0

# A WR-90 section of length `len` mm with cells of about `h` mm; the bottom face z = 0 is a wave port, the top
# face a wave port too unless `top` names another termination.
function wr90(h, len; top = :port, kwargs...)
    n = ceil.(Int, (WR90_A, WR90_B, len) ./ h)
    grid = generate_grid(Tetrahedron, Tuple(n), Vec(0.0, 0.0, 0.0), Vec(WR90_A, WR90_B, float(len)))
    walls = ["left", "right", "front", "back"]
    ports = [WavePort("bottom"; a_direction = (1, 0, 0), kwargs...)]
    top === :port && push!(ports, WavePort("top"; a_direction = (1, 0, 0), kwargs...))
    return MaxwellModel(
        grid; pec = top === :pec ? [walls; "top"] : walls, absorbing = top === :abc ? ["top"] : String[],
        ports, length_scale = 1.0e-3,
    )
end

phase_deg(z) = rad2deg(angle(z))
wrap_deg(x) = mod(x + 180, 360) - 180

@testset "facet blocks share the cell pattern" begin
    model = wr90(5.0, 10.0)
    B = FiniteElementMethod.tangential_mass(model, ["bottom", "top"])
    @test B.colptr == model.blocks.curlcurl.colptr && rowvals(B) == rowvals(model.blocks.curlcurl)
    @test issymmetric(B)
    # Each port block is the tangential mass of its own face.
    @test model.ports[1].block + model.ports[2].block ≈ B
    # The norm of the projected mode approaches ∮ e⋅e = ab/2 from below.
    port = model.ports[1]
    @test port.a ≈ WR90_A && port.b ≈ WR90_B && port.normal ≈ Vec(0.0, 0.0, -1.0)
    @test 0.97 * WR90_A * WR90_B / 2 < port.norm < WR90_A * WR90_B / 2
end

@testset "TE10 mode constants" begin
    model = wr90(5.0, 10.0)
    @test FiniteElementMethod.cutoff_frequency(model, 1) ≈ 6.5571e9 rtol = 1.0e-4
    @test FiniteElementMethod.propagation_constant(model, 1, F10) ≈ BETA10 rtol = 1.0e-5
    # Below cutoff the mode decays away from the port: β = -iα.
    β5 = FiniteElementMethod.propagation_constant(model, 2, 2π * 5.0e9)
    @test real(β5) == 0 && imag(β5) ≈ -sqrt((π / 22.86e-3)^2 - (2π * 5.0e9 / FiniteElementMethod.C0)^2)
    @test_throws "not in the port plane" wr90(5.0, 10.0; a_direction = (0, 0, 1))
    @test_throws "not a planar rectangle" MaxwellModel(
        generate_grid(Tetrahedron, (2, 2, 2)); ports = [WavePort("bottom"; a_direction = (1, 1, 0))]
    )
    @test_throws "no facet set" MaxwellModel(generate_grid(Tetrahedron, (2, 2, 2)); absorbing = ["nope"])
end

@testset "empty WR-90 guide, two ports" begin
    len = 50.0
    for (h, tol_deg) in ((2.0, 2.0), (1.5, 1.0))
        model = wr90(h, len)
        S = solve_sweep(model, [F10]).S[:, :, 1]
        err = wrap_deg(phase_deg(S[2, 1]) - wrap_deg(-rad2deg(BETA10 * len * 1.0e-3)))
        @info "WR-90 straight guide" h ndofs(model.dh) S[1, 1] abs(S[2, 1]) phase_deg(S[2, 1]) err
        @test abs(S[1, 1]) < 0.05 && abs(S[2, 2]) < 0.05
        @test abs(S[2, 1]) ≈ 1 atol = 0.01
        @test abs(err) < tol_deg
        @test S[1, 2] ≈ S[2, 1] rtol = 1.0e-8
        @test abs2(S[1, 1]) + abs2(S[2, 1]) ≈ 1 atol = 1.0e-3
    end
    # De-embedding both reference planes to the middle of the guide leaves no transmission phase.
    model = wr90(2.0, len; deembed = len / 2)
    S = solve_sweep(model, [F10]).S[:, :, 1]
    @test abs(S[2, 1] - 1) < 0.05
end

@testset "exactly at cutoff the wave impedance is infinite: a clear error, not NaN" begin
    # With eps0 = mu0 = 1 and unit scale, ω = kc is exactly the cutoff and β is exactly zero.
    grid = generate_grid(Tetrahedron, (2, 1, 2), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 2.0))
    model = MaxwellModel(
        grid; pec = ["left", "right", "front", "back"], eps0 = 1.0, mu0 = 1.0,
        ports = [WavePort("bottom"; a_direction = (1, 0, 0)), WavePort("top"; a_direction = (1, 0, 0))],
    )
    ωc = model.ports[1].kc
    @test FiniteElementMethod.propagation_constant(model, 1, ωc) == 0
    @test_throws "cutoff of wave port 1" solve_sweep(model, [ωc])
    @test_throws "cutoff of wave port 1" FiniteElementMethod.port_excitations(model, ωc)
    @test all(isfinite, solve_sweep(model, [ωc * (1 + 1.0e-6)]).S)
end

@testset "evanescent TE10 below cutoff" begin
    len = 10.0
    model = wr90(2.0, len)
    ω = 2π * 5.0e9
    α = -imag(FiniteElementMethod.propagation_constant(model, 1, ω))
    S = solve_sweep(model, [ω]).S[:, :, 1]
    @info "WR-90 below cutoff" S exp(-α * len * 1.0e-3)
    @test S[2, 1] ≈ exp(-α * len * 1.0e-3) rtol = 0.03
    @test abs(S[1, 1]) < 0.03
end

@testset "absorbing boundary terminating a guide" begin
    len = 30.0
    model = wr90(2.0, len; top = :abc)
    S11 = only(solve_sweep(model, [F10]).S)
    Γ = (BETA10 - K010) / (BETA10 + K010)
    expected = Γ * exp(-2im * BETA10 * len * 1.0e-3)
    @info "WR-90 with ABC" S11 abs(S11) Γ expected wrap_deg(phase_deg(S11) - phase_deg(expected))
    @test Γ ≈ -0.1396 atol = 1.0e-4
    @test abs(S11) ≈ abs(Γ) rtol = 0.03
    @test abs(wrap_deg(phase_deg(S11) - phase_deg(expected))) < 3
    # A PEC short instead reflects everything: S11 = -exp(-2iβL).
    S11 = only(solve_sweep(wr90(2.0, len; top = :pec), [F10]).S)
    @test abs(S11) ≈ 1 atol = 1.0e-3
    @test abs(wrap_deg(phase_deg(S11) - phase_deg(-exp(-2im * BETA10 * len * 1.0e-3)))) < 3
end

@testset "sweep equals single-frequency solves" begin
    model = wr90(3.0, 30.0)
    ωs = 2π .* [8.0e9, 10.0e9, 12.0e9]
    calls = Symbol[]
    function counting(A, B, cache)
        push!(calls, cache === nothing ? :lu : :refactor)
        return FiniteElementMethod.umfpack_solver(A, B, cache)
    end
    sweep = solve_sweep(model, ωs; fields = true, solver = counting)
    @test calls == [:lu, :refactor, :refactor]
    @test size(sweep.S) == (2, 2, 3) && length(sweep.fields) == 3
    @test solve_sweep(model, ωs).fields === nothing
    for (k, ω) in pairs(ωs)
        # An independent solve: a fresh operator, constraints and factorisation.
        A = maxwell_operator(model, ω)
        B = FiniteElementMethod.port_excitations(model, ω)
        for j in axes(B, 2)
            b = B[:, j]
            Aj = copy(A)
            apply!(Aj, b, model.ch)
            @test Aj \ b ≈ sweep.fields[k][:, j] rtol = 1.0e-10
        end
        @test solve_sweep(model, [ω]).S[:, :, 1] ≈ sweep.S[:, :, k] rtol = 1.0e-10
    end
    # The driven field is the TE10 mode, E_y ∝ sin(πx/a).
    sol = MaxwellSolution(model, ωs[2], sweep.fields[2][:, 1])
    E = field_values(sol, [Vec(WR90_A / 2, WR90_B / 2, 0.0), Vec(WR90_A / 8, WR90_B / 2, 0.0)]).E
    @test abs(E[1][2]) ≈ abs(1 + sweep.S[1, 1, 2]) rtol = 0.05  # incident plus reflected at the port face
    @test abs(E[2][2]) / abs(E[1][2]) ≈ sin(π / 8) rtol = 0.1
    @test_throws "no wave ports" solve_sweep(MaxwellModel(generate_grid(Tetrahedron, (1, 1, 1))), [1.0])
end

@testset "port rim must be PEC" begin
    grid = generate_grid(Tetrahedron, (4, 2, 4), Vec(0.0, 0.0, 0.0), Vec(WR90_A, WR90_B, 20.0))
    @test_throws "is not PEC" MaxwellModel(grid; ports = [WavePort("bottom"; a_direction = (1, 0, 0))])
    @test_throws "is not PEC" MaxwellModel(
        grid; pec = ["left", "right", "front"], ports = [WavePort("bottom"; a_direction = (1, 0, 0))]
    )
    @test_throws "cannot also be in `pec`" MaxwellModel(
        generate_grid(Tetrahedron, (2, 2, 2)); pec = ["left", "right", "front", "back", "bottom"],
        ports = [WavePort("bottom"; a_direction = (1, 0, 0))]
    )
end

# Transmission-line answer for a vacuum guide (port at z = 0) with a lossy filling over the last `len2` mm and
# the ABC of admittance `y_abc` (k0 sqrt(εc) for the correct ABC) at its end; E-field admittances scale with β.
function lossy_abc_s11(len1, len2, εc, y_abc = K010 * sqrt(εc))
    β2 = sqrt(complex(K010^2 * εc - (π / (WR90_A * 1.0e-3))^2))
    β2 = imag(β2) > 0 ? -β2 : β2
    Γ2 = (β2 - y_abc) / (β2 + y_abc) * exp(-2im * β2 * len2 * 1.0e-3)
    y = β2 * (1 - Γ2) / (1 + Γ2)
    return (BETA10 - y) / (BETA10 + y) * exp(-2im * BETA10 * len1 * 1.0e-3)
end

@testset "absorbing boundary on a conducting material" begin
    len1, len2, σ = 20.0, 10.0, 0.2
    εc = 1 - im * σ / (F10 * FiniteElementMethod.EPS0)
    grid = generate_grid(Tetrahedron, (12, 6, 15), Vec(0.0, 0.0, 0.0), Vec(WR90_A, WR90_B, len1 + len2))
    addcellset!(grid, "lossy", x -> x[3] >= len1 - 1.0e-9)
    model = MaxwellModel(
        grid; materials = Dict("lossy" => MaxwellMaterial(; sigma = σ)), pec = ["left", "right", "front", "back"],
        absorbing = ["top"], ports = [WavePort("bottom"; a_direction = (1, 0, 0))], length_scale = 1.0e-3,
    )
    S11 = only(solve_sweep(model, [F10]).S)
    expected = lossy_abc_s11(len1, len2, εc)
    ignoring_σ = lossy_abc_s11(len1, len2, εc, K010)  # an ABC blind to the conductivity of its cells
    @info "WR-90 ABC on a conducting load" εc S11 expected abs(S11 - expected) abs(ignoring_σ - expected)
    @test abs(S11 - expected) < 0.01
    @test abs(ignoring_σ - expected) > 0.03
end

@testset "sweep keeps an independent field per frequency, even from a solver that reuses its output" begin
    model = wr90(5.0, 10.0)
    ωs = F10 .* [0.9, 1.1]
    reused = Matrix{ComplexF64}(undef, 0, 0)
    function one_buffer_solver(A, B, cache)
        U, cache = FiniteElementMethod.umfpack_solver(A, B, cache)
        size(reused) == size(U) || (reused = similar(U))
        copyto!(reused, U)
        return reused, cache
    end
    sweep = solve_sweep(model, ωs; fields = true, solver = one_buffer_solver)
    fresh = [solve_sweep(model, [ω]; fields = true).fields[1] for ω in ωs]
    @test sweep.fields[1] !== sweep.fields[2]
    @test sweep.fields[1] ≈ fresh[1] && sweep.fields[2] ≈ fresh[2]
end

@testset "an empty frequency vector gives an empty sweep" begin
    sweep = solve_sweep(wr90(5.0, 10.0), Float64[]; fields = true)
    @test isempty(sweep.omegas) && size(sweep.S) == (2, 2, 0) && isempty(sweep.fields)
end

@testset "the absorbing operator is finite at ω = 0" begin
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    addcellset!(grid, "all", x -> true)
    for material in (MaxwellMaterial(), MaxwellMaterial(; sigma = 0.5))
        m = MaxwellModel(grid; absorbing = ["top"], materials = Dict("all" => material), eps0 = 1.0, mu0 = 1.0)
        @test all(isfinite, maxwell_operator(m, 0.0).nzval)
    end
end

@testset "lossy port materials are rejected" begin
    grid = generate_grid(Tetrahedron, (2, 1, 2), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 2.0))
    addcellset!(grid, "lossy", x -> x[3] >= 0.0)
    walls = ["left", "right", "front", "back"]
    @test_throws "lossy" MaxwellModel(
        grid; pec = walls, eps0 = 1.0, mu0 = 1.0, materials = Dict("lossy" => MaxwellMaterial(; eps_r = 1 - 2im)),
        ports = [WavePort("bottom"; a_direction = (1, 0, 0))]
    )
end

@testset "a facet set cannot be both absorbing and a wave port" begin
    grid = generate_grid(Tetrahedron, (2, 1, 2), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 2.0))
    addfacetset!(grid, "bottom_alias", x -> x[3] ≈ 0.0)
    walls = ["left", "right", "front", "back"]
    for name in ("bottom", "bottom_alias")
        @test_throws "absorbing and a wave port" MaxwellModel(
            grid; pec = walls, eps0 = 1.0, mu0 = 1.0, absorbing = [name],
            ports = [WavePort("bottom"; a_direction = (1, 0, 0))]
        )
    end
end

@testset "tetrahedron facet tables match Ferrite" begin
    @test FiniteElementMethod._TET_FACET_NODES == Ferrite.reference_facets(RefTetrahedron)
end
