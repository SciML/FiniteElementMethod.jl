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
