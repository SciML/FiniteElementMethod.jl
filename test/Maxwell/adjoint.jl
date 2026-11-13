# Discrete adjoint material sensitivities of the WR-90 dielectric slab: against central finite differences of
# the forward sweep (both discrete, so they agree to many digits) and against the closed-form slab.
using FiniteElementMethod, Ferrite, LinearAlgebra, SparseArrays, Test

const SLAB_A, SLAB_B = 22.86, 10.16  # mm
const SLAB_LEAD, SLAB_D = 10.0, 10.0  # mm of vacuum guide on each side of the slab, and the slab
const SLAB_F = 2π * 10.0e9
const SLAB_WALLS = ["left", "right", "front", "back"]

# The slab fills the cross-section over SLAB_LEAD <= z <= SLAB_LEAD + SLAB_D; ports at both ends, de-embedded to
# the slab faces. With `abc = true` the guide ends in an absorbing boundary just behind the slab instead.
function slab_grid(h; abc = false)
    len = abc ? SLAB_LEAD + SLAB_D : 2SLAB_LEAD + SLAB_D
    n = ceil.(Int, (SLAB_A, SLAB_B, len) ./ h)
    grid = generate_grid(Tetrahedron, Tuple(n), Vec(0.0, 0.0, 0.0), Vec(SLAB_A, SLAB_B, len))
    addcellset!(grid, "slab", x -> SLAB_LEAD - 1.0e-9 <= x[3] <= SLAB_LEAD + SLAB_D + 1.0e-9; all = true)
    addcellset!(grid, "lead", x -> x[3] <= SLAB_LEAD + 1.0e-9; all = true)
    return grid
end

function slab_model(grid, material; abc = false)
    ports = [WavePort("bottom"; a_direction = (1, 0, 0), deembed = SLAB_LEAD)]
    abc || push!(ports, WavePort("top"; a_direction = (1, 0, 0), deembed = SLAB_LEAD))
    return MaxwellModel(
        grid; materials = Dict("slab" => material), pec = SLAB_WALLS, absorbing = abc ? ["top"] : String[], ports,
        length_scale = 1.0e-3,
    )
end

function shifted(m::MaxwellMaterial, kind, δ)
    kind === :eps_r && return MaxwellMaterial(m.eps_r + δ, m.mu_r, m.sigma)
    kind === :eps_r_imag && return MaxwellMaterial(m.eps_r + im * δ, m.mu_r, m.sigma)
    kind === :mu_r && return MaxwellMaterial(m.eps_r, m.mu_r + δ, m.sigma)
    return MaxwellMaterial(m.eps_r, m.mu_r, m.sigma + δ)
end

# Central differences of S with respect to the slab's `kind`, step δ.
function fd_dS(grid, material, kind, ωs, δ; abc = false)
    S(δ) = solve_sweep(slab_model(grid, shifted(material, kind, δ); abc), ωs).S
    return (S(δ) - S(-δ)) / 2δ
end

relerr(a, b) = norm(a - b) / norm(b)

# Closed form for a lossless slab εr in an otherwise empty guide, referenced to the slab faces.
function slab_closed_form(εr; ω = SLAB_F)
    k0, kc = ω / FiniteElementMethod.C0, π / (SLAB_A * 1.0e-3)
    β, β2 = sqrt(k0^2 - kc^2), sqrt(εr * k0^2 - kc^2)
    Γ, P = (β - β2) / (β + β2), exp(-im * β2 * SLAB_D * 1.0e-3)
    return (S11 = Γ * (1 - P^2) / (1 - Γ^2 * P^2), S21 = P * (1 - Γ^2) / (1 - Γ^2 * P^2))
end

const SLAB_KINDS = (:eps_r, :eps_r_imag, :mu_r, :sigma)
const SLAB_STEPS = Dict(:eps_r => 1.0e-4, :eps_r_imag => 1.0e-4, :mu_r => 1.0e-4, :sigma => 1.0e-5)

@testset "closed-form slab" begin
    cf = slab_closed_form(2.2)
    @test cf.S11 ≈ -0.07985 + 0.18593im atol = 1.0e-4
    @test cf.S21 ≈ -0.89984 - 0.38644im atol = 1.0e-4
    δ = 1.0e-6
    dcf = (slab_closed_form(2.2 + δ).S11 - slab_closed_form(2.2 - δ).S11) / 2δ
    grid = slab_grid(2.0)
    sens = s_parameter_sensitivities(slab_model(grid, MaxwellMaterial(; eps_r = 2.2)), [SLAB_F], [(:eps_r, "slab")])
    S11, dS11 = sens.S[1, 1, 1], sens.dS[1, 1, 1, 1]
    @info "slab against the closed form" S11 cf.S11 dS11 dcf
    @test abs(S11 - cf.S11) < 0.01
    @test abs(dS11 - dcf) < 0.05 * abs(dcf)
end

@testset "adjoint against finite differences: $label" for (label, material) in (
        ("lossless slab", MaxwellMaterial(; eps_r = 2.2)),
        ("lossy slab", MaxwellMaterial(; eps_r = 2.2 - 0.05im, mu_r = 1.3, sigma = 0.1)),
    )
    grid = slab_grid(2.0)
    ωs = 2π .* [9.0e9, 11.0e9]
    params = [(kind, "slab") for kind in SLAB_KINDS]
    sens = s_parameter_sensitivities(slab_model(grid, material), ωs, params)
    @test size(sens.dS) == (2, 2, 2, 4) && sens.params == params
    for (l, kind) in pairs(SLAB_KINDS)
        fd = fd_dS(grid, material, kind, ωs, SLAB_STEPS[kind])
        err = relerr(sens.dS[:, :, :, l], fd)
        @info "dS/d$kind, $label" sens.dS[:, :, 1, l] err
        @test err < 1.0e-6
        @test sens.dS[1, 2, :, l] ≈ sens.dS[2, 1, :, l] rtol = 1.0e-8  # reciprocity
    end
    # The derivative with respect to the imaginary part of εr is i times that of its real part (S is holomorphic in εr).
    @test sens.dS[:, :, :, 2] ≈ im .* sens.dS[:, :, :, 1] rtol = 1.0e-10
    # Lossless: dS is then also constrained by unitarity, S'ᴴS + SᴴS' = 0 for real-εr perturbations.
    if iszero(material.sigma)
        for k in 1:2
            S, dS = sens.S[:, :, k], sens.dS[:, :, k, 1]
            @test norm(dS' * S + S' * dS) < 1.0e-3 * norm(dS)
        end
    end
end

@testset "absorbing boundary on the parameter set" begin
    grid = slab_grid(2.0; abc = true)
    material = MaxwellMaterial(; eps_r = 2.2 - 0.05im, mu_r = 1.3, sigma = 0.1)
    params = [(kind, "slab") for kind in SLAB_KINDS]
    sens = s_parameter_sensitivities(slab_model(grid, material; abc = true), [SLAB_F], params)
    for (l, kind) in pairs(SLAB_KINDS)
        fd = fd_dS(grid, material, kind, [SLAB_F], SLAB_STEPS[kind]; abc = true)
        err = relerr(sens.dS[:, :, :, l], fd)
        @info "dS11/d$kind with the ABC on the slab" only(sens.dS[:, :, :, l]) err
        @test err < 1.0e-6
    end
end

@testset "adjoint gradient of a field functional" begin
    grid = slab_grid(2.0)
    material = MaxwellMaterial(; eps_r = 2.2 - 0.05im, mu_r = 1.3, sigma = 0.1)
    params = [(kind, "slab") for kind in SLAB_KINDS]
    # J = Σ|u|² over the field with port 1 driven: dJ/du = ∂J/∂Re u + i ∂J/∂Im u = 2u.
    J(model) = sum(abs2, solve_sweep(model, [SLAB_F]; fields = true).fields[1][:, 1])
    model = slab_model(grid, material)
    u = solve_sweep(model, [SLAB_F]; fields = true).fields[1][:, 1]
    grad = adjoint_gradient(model, SLAB_F, u, 2 .* u, params)
    for (l, kind) in pairs(SLAB_KINDS)
        δ = SLAB_STEPS[kind]
        fd = (J(slab_model(grid, shifted(material, kind, δ))) - J(slab_model(grid, shifted(material, kind, -δ)))) / 2δ
        @info "dJ/d$kind" grad[l] fd
        @test grad[l] ≈ fd rtol = 1.0e-6
    end
    # The same through a supplied factorisation of the constrained operator.
    A = maxwell_operator(model, SLAB_F)
    apply!(A, model.ch)
    F = lu(A)
    @test adjoint_gradient(model, SLAB_F, u, 2 .* u, params; solver = (A, B, F) -> (F \ B, F), cache = F) ≈ grad
end

@testset "sensitivity inputs" begin
    grid = slab_grid(3.0)
    model = slab_model(grid, MaxwellMaterial(; eps_r = 2.2))
    @test_throws "no cell set" s_parameter_sensitivities(model, [SLAB_F], [(:eps_r, "nope")])
    @test_throws "unknown material parameter" s_parameter_sensitivities(model, [SLAB_F], [(:epsilon, "slab")])
    @test_throws "touches wave port" s_parameter_sensitivities(model, [SLAB_F], [(:eps_r, "lead")])
    sweep = solve_sweep(model, [SLAB_F])
    @test_throws "no fields" s_parameter_sensitivities(model, sweep, [(:eps_r, "slab")])
    @test_throws "do not match" s_parameter_sensitivities(
        model, MaxwellSweep(sweep.omegas, sweep.S, [zeros(ComplexF64, 3, 2)]), [(:eps_r, "slab")]
    )
    @test size(s_parameter_sensitivities(model, [SLAB_F], []).dS) == (2, 2, 1, 0)
    @test adjoint_gradient(model, SLAB_F, zeros(ComplexF64, ndofs(model.dh)), zeros(ndofs(model.dh)), []) == Float64[]
    # A sweep with fields gives the same as solving inside; a set with no free dofs has zero sensitivity.
    sweep = solve_sweep(model, [SLAB_F]; fields = true)
    @test s_parameter_sensitivities(model, sweep, [("eps_r", "slab")]).dS ≈
        s_parameter_sensitivities(model, [SLAB_F], [(:eps_r, "slab")]).dS
    addcellset!(grid, "empty", Int[])
    @test iszero(s_parameter_sensitivities(model, sweep, [(:sigma, "empty")]).dS)
end

@testset "sensitivities reject a model with a nonsymmetric term" begin
    model = slab_model(slab_grid(3.0), MaxwellMaterial(; eps_r = 2.2))
    extra = copy(model.blocks.sigma)
    nonzeros(extra) .= rand(nnz(extra))
    FiniteElementMethod.add_term!(model, extra, ω -> 1.0)
    # A tiny block with a huge coefficient is as nonsymmetric as a large one.
    tiny = copy(model.blocks.sigma)
    nonzeros(tiny) .= 0
    rows, cols = findnz(tiny)
    scaled = slab_model(slab_grid(3.0), MaxwellMaterial(; eps_r = 2.2))
    i = findfirst(k -> rows[k] != cols[k] && rows[k] ∉ scaled.ch.prescribed_dofs && cols[k] ∉ scaled.ch.prescribed_dofs, eachindex(rows))
    nonzeros(tiny)[i] = 1.0e-13
    FiniteElementMethod.add_term!(scaled, tiny, ω -> 1.0e13)
    @test_throws "nonsymmetric" s_parameter_sensitivities(scaled, [SLAB_F], [(:eps_r, "slab")])
    u = zeros(ComplexF64, ndofs(model.dh))
    @test_throws "nonsymmetric" s_parameter_sensitivities(model, [SLAB_F], [(:eps_r, "slab")])
    @test_throws "nonsymmetric" adjoint_gradient(model, SLAB_F, u, zeros(ndofs(model.dh)), [(:eps_r, "slab")])
    @test_throws "nonsymmetric" s_parameter_sensitivities(
        model, MaxwellSweep([SLAB_F], zeros(ComplexF64, 2, 2, 1), [zeros(ComplexF64, ndofs(model.dh), 2)]), [(:eps_r, "slab")]
    )
end

@testset "the symmetry check ignores constrained dofs" begin
    grid = generate_grid(Tetrahedron, (4, 2, 6), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 3.0))
    ports = [WavePort("bottom"; a_direction = (1, 0, 0)), WavePort("top"; a_direction = (1, 0, 0))]
    addcellset!(grid, "core", x -> 1.0 - 1.0e-9 <= x[3] <= 2.0 + 1.0e-9; all = true)
    model = MaxwellModel(grid; pec = SLAB_WALLS, ports, eps0 = 1.0, mu0 = 1.0)
    ω = 2.5
    # A nonsymmetric block on the free dofs, padded with a huge symmetric entry on a PEC dof.
    padded = copy(model.blocks.eps_real)
    rows = rowvals(padded)
    for j in axes(padded, 2), k in nzrange(padded, j)
        nonzeros(padded)[k] *= 1 + 0.7 * sin(rows[k])
    end
    pec_dof = first(model.ch.prescribed_dofs)
    padded[pec_dof, pec_dof] = 1.0e16
    FiniteElementMethod.add_term!(model, padded, _ -> 3.0)
    u = zeros(ComplexF64, ndofs(model.dh))
    @test_throws "nonsymmetric" s_parameter_sensitivities(model, [ω], [(:eps_r, "core")])
    @test_throws "nonsymmetric" s_parameter_sensitivities(
        model, MaxwellSweep([ω], zeros(ComplexF64, 2, 2, 1), [zeros(ComplexF64, ndofs(model.dh), 2)]), [(:eps_r, "core")]
    )
    @test_throws "nonsymmetric" adjoint_gradient(model, ω, u, zeros(ndofs(model.dh)), [(:eps_r, "core")])
end

@testset "the absorbing boundary's derivative is finite at ω = 0" begin
    grid = slab_grid(3.0; abc = true)
    model = slab_model(grid, MaxwellMaterial(; eps_r = 2.2); abc = true)
    n = ndofs(model.dh)
    # The static operator is singular, so the adjoint solve is stubbed: only dA/dp u is under test.
    stub(A, B, cache) = (zeros(ComplexF64, size(B)), cache)
    for kind in SLAB_KINDS
        g = adjoint_gradient(model, 0.0, ones(ComplexF64, n), ones(n), [(kind, "slab")]; solver = stub)
        @test all(isfinite, g)
    end
    for kind in SLAB_KINDS, term in FiniteElementMethod._parameter_terms(model, (kind, "slab"))
        @test isfinite(term.coefficient(0.0))
    end
end
