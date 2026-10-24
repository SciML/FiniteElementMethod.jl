# Stage e (spec §5.1, §5.2): Galerkin projection and term classification, checked term for term against
# the verified dauntless prototype outputs (test_linear.out, test_nonlinear.out, test_multifield.out,
# test_testspace.out).
using FiniteElementMethod: weak_form, classify, TestSlots
using Symbolics
using Test

include("strongform_seam.jl")
using .StrongFormSeam: analyse, BC, StateSlots

"`a` and `b` agree at several random values of every variable they contain."
function numeq(a, b)
    vars = union(Symbolics.get_variables(Num(a)), Symbolics.get_variables(Num(b)))
    return all(1:3) do _
        vals = Dict(v => 0.5 + rand() for v in vars)
        va, vb = (Float64(Symbolics.value(substitute(Num(ex), vals; fold = Val(true)))) for ex in (a, b))
        isapprox(va, vb; rtol = 1.0e-12, atol = 1.0e-12)
    end
end

"Every kind expected is present with a matching integrand, and nothing else."
function matches(terms, expected)
    got = Dict(term.kind => term.integrand for term in terms)
    return Set(keys(got)) == Set(keys(expected)) && all(k -> numeq(got[k], expected[k]), keys(expected))
end
volume(wf, e) = filter(term -> term.eq == e, wf.volume)
facet(wf, piece, e = 1) = filter(term -> term.piece == piece && term.eq == e, wf.facet)

@variables t x y k α β h
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)
const RIGHT, TOP = [1.0, 0.0], [0.0, 1.0]
scalar_wf(eq, bcs; params = [k, α, β, h]) = weak_form(analyse([eq], bcs; depvars = [U], t, sivs = [x, y], params))

dir = BC("left", U ~ 0.1 * y)
rhs = x * y + 1

@testset "case (i): constant k, Neumann rescaled by k, Robin split" begin
    wf = scalar_wf(
        Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + rhs,
        [dir, BC("right", Dx(U) ~ h, RIGHT), BC("top", k * Dy(U) + α * U ~ β, TOP)]
    )
    s, (; V, W) = wf.analysis.slots, wf.test
    G, X, P = s.G, s.X, s.P
    @test matches(
        volume(wf, 1), Dict(
            :mass => s.Ud[1] * V, :bilinear => P[1] * (G[1] * W[1] + G[2] * W[2]), :source => -V - V * X[1] * X[2]
        )
    )
    @test matches(facet(wf, "right"), Dict(:source => -P[1] * P[4] * V))
    @test matches(facet(wf, "top"), Dict(:bilinear => P[2] * s.U[1] * V, :source => -P[3] * V))
    @test length(wf.analysis.essential) == 1
    ess = only(wf.analysis.essential)
    @test ess.piece == "left" && ess.field == 1 && numeq(ess.g, 0.1 * X[2])
end

@testset "case (ii): variable k(x, y) in divergence form" begin
    kxy = 1 + x^2 * y
    wf = scalar_wf(
        Dt(U) ~ Dx(kxy * Dx(U)) + Dy(kxy * Dy(U)) + rhs,
        [dir, BC("right", Dx(U) ~ h, RIGHT), BC("top", kxy * Dy(U) + α * U ~ β, TOP)]
    )
    s, (; V, W) = wf.analysis.slots, wf.test
    G, X, P = s.G, s.X, s.P
    kX = 1 + X[1]^2 * X[2]
    @test matches(
        volume(wf, 1), Dict(
            :mass => s.Ud[1] * V, :bilinear => kX * (G[1] * W[1] + G[2] * W[2]), :source => -V - V * X[1] * X[2]
        )
    )
    @test matches(facet(wf, "right"), Dict(:source => -P[4] * V * kX))
    @test matches(facet(wf, "top"), Dict(:bilinear => P[2] * s.U[1] * V, :source => -P[3] * V))
end

@testset "case (ii-b): non-divergence a(x, y)Δu with advection and reaction" begin
    axy = 1 + x^2 + y
    wf = scalar_wf(
        Dt(U) ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) - 0.7 * Dx(U) - 0.3 * Dy(U) - 2.5 * U + rhs,
        [dir, BC("right", Dx(U) ~ h, RIGHT), BC("top", axy * Dy(U) + α * U ~ β, TOP)]
    )
    s, (; V, W) = wf.analysis.slots, wf.test
    G, X, P = s.G, s.X, s.P
    aX = 1 + X[1]^2 + X[2]
    bilinear = 0.7G[1] * V + 1.3G[2] * V + 2.5s.U[1] * V + 2G[1] * V * X[1] + aX * (G[1] * W[1] + G[2] * W[2])
    @test matches(volume(wf, 1), Dict(:mass => s.Ud[1] * V, :bilinear => bilinear, :source => -V - V * X[1] * X[2]))
    @test matches(facet(wf, "right"), Dict(:source => -P[4] * V * aX))
    @test matches(facet(wf, "top"), Dict(:bilinear => P[2] * s.U[1] * V, :source => -P[3] * V))
end

@testset "case (ii-c): anisotropic tensor with cross terms" begin
    k11, k12, k22 = 2 + x, 0.4 + 0.1y, 1 + y^2
    eq = Dt(U) ~ Dx(k11 * Dx(U) + k12 * Dy(U)) + Dy(k12 * Dx(U) + k22 * Dy(U)) + rhs
    wf = scalar_wf(eq, [dir, BC("top", k12 * Dx(U) + k22 * Dy(U) + α * U ~ β, TOP)])
    s, (; V, W) = wf.analysis.slots, wf.test
    G, X, P = s.G, s.X, s.P
    bilinear = (2 + X[1]) * G[1] * W[1] + (0.4 + 0.1X[2]) * (G[1] * W[2] + G[2] * W[1]) + (1 + X[2]^2) * G[2] * W[2]
    @test matches(volume(wf, 1), Dict(:mass => s.Ud[1] * V, :bilinear => bilinear, :source => -V - V * X[1] * X[2]))
    @test isempty(facet(wf, "right"))
    @test matches(facet(wf, "top"), Dict(:bilinear => P[2] * s.U[1] * V, :source => -P[3] * V))
    # ∂ₙu = h is not proportional to the conormal k11 uₓ + k12 u_y on x = 1
    @test_throws "not a conormal" scalar_wf(eq, [BC("right", Dx(U) ~ h, RIGHT)])
end

@testset "case (iii): k(u) = 1 + u², nonlinear volume and facet terms" begin
    ustar = exp(x) * sin(y) / 2
    kstar = 1 + ustar^2
    fsym = expand_derivatives(-(Dx(kstar * Dx(ustar)) + Dy(kstar * Dy(ustar))))
    hsym = expand_derivatives(Dx(ustar))
    eq = 0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + fsym
    bcs = [BC("left", U ~ ustar), BC("bottom", U ~ ustar), BC("top", U ~ ustar), BC("right", Dx(U) ~ hsym, RIGHT)]
    wf = weak_form(analyse([eq], bcs; depvars = [U], t, sivs = [x, y]))
    s, (; V, W) = wf.analysis.slots, wf.test
    G, X, U1 = s.G, s.X, s.U[1]
    gw = G[1] * W[1] + G[2] * W[2]
    source = (1 // 4) * V * cos(X[2])^2 * exp(X[1])^3 * sin(X[2]) + (1 // 4) * V * exp(X[1])^3 * sin(X[2])^3
    @test matches(volume(wf, 1), Dict(:bilinear => gw, :source => source, :nonlinear => U1^2 * gw))
    @test matches(
        facet(wf, "right"), Dict(
            :source => (-1 // 2) * V * exp(X[1]) * sin(X[2]),
            :nonlinear => (-1 // 2) * U1^2 * V * exp(X[1]) * sin(X[2])
        )
    )
    @test length(wf.analysis.essential) == 3
end

@testset "two fields with cross-diffusion, routed natural conditions" begin
    @variables Du Dw χ a b
    @variables w(..)
    Wf = w(t, x, y)
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a - U + U^2 * Wf,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U^2 * Wf,
    ]
    bcs = [BC("right", Du * Dx(U) + χ * Dx(Wf) ~ 0.1, RIGHT), BC("right", Dx(Wf) ~ 0.2, RIGHT)]
    wf = weak_form(analyse(eqs, bcs; depvars = [U, Wf], t, sivs = [x, y], params = [Du, Dw, χ, a, b]))
    s, (; V, W) = wf.analysis.slots, wf.test
    G, P, U1, U2 = s.G, s.P, s.U[1], s.U[2]
    gu, gw = G[1] * W[1] + G[2] * W[2], G[3] * W[1] + G[4] * W[2]
    @test matches(
        volume(wf, 1), Dict(
            :mass => s.Ud[1] * V, :bilinear => U1 * V + P[1] * gu + P[3] * gw,
            :source => -P[4] * V, :nonlinear => -U1^2 * U2 * V
        )
    )
    @test matches(
        volume(wf, 2), Dict(
            :mass => s.Ud[2] * V, :bilinear => P[2] * gw, :source => -P[5] * V, :nonlinear => U1^2 * U2 * V
        )
    )
    @test matches(facet(wf, "right", 1), Dict(:source => -0.1V))
    @test matches(facet(wf, "right", 2), Dict(:source => -0.2P[2] * V))
end

@testset "timedep and paramdep flags" begin
    wf = scalar_wf(Dt(U) ~ (1 + t) * (Dx(Dx(U)) + Dy(Dy(U))) + k * x, [BC("right", Dx(U) ~ sin(t), RIGHT)])
    bykind = Dict(term.kind => term for term in wf.volume)
    @test (bykind[:mass].timedep, bykind[:mass].paramdep) == (false, false)
    @test (bykind[:bilinear].timedep, bykind[:bilinear].paramdep) == (true, false)
    @test (bykind[:source].timedep, bykind[:source].paramdep) == (false, true)
    rightsrc = only(facet(wf, "right"))
    @test rightsrc.kind === :source && rightsrc.timedep && !rightsrc.paramdep
end

@testset "classification ranges over the declared slot set (Hessian slots)" begin
    s = StateSlots(1, 2, 2)
    (; V, W) = TestSlots(2)
    G, H, X, P = s.G, s.H, s.X, s.P
    # SUPG integrand from test_testspace.out: the τ·(b·∇v)·εΔu part lives in Hessian slots only
    supg = -V + G[1] * V + 0.5G[2] * V - V * X[1] + P[1] * (G[1] * W[1] + G[2] * W[2]) +
        (-1 + G[1] + 0.5G[2] - X[1] - (H[1] + H[4]) * P[1]) * P[2] * (W[1] + 0.5W[2])
    source = -V - V * X[1] - (1 + X[1]) * P[2] * (W[1] + 0.5W[2])
    classes = classify(supg, s, TestSlots(2))
    @test [c.kind for c in classes] == [:bilinear, :source]
    @test matches(classes, Dict(:bilinear => supg - source, :source => source))
    @test all(c -> c.paramdep && !c.timedep, classes)
    # Without H declared the Hessian term is state-free and lands in :source: the slot set decides.
    noH = (; s.U, s.G, H = nothing, s.T, s.P)
    @test matches(
        classify(supg, noH, TestSlots(2)),
        Dict(
            :bilinear => supg - source + (H[1] + H[4]) * P[1] * P[2] * (W[1] + 0.5W[2]),
            :source => source - (H[1] + H[4]) * P[1] * P[2] * (W[1] + 0.5W[2])
        )
    )
end
