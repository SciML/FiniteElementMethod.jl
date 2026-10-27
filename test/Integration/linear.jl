# End to end, linear single-field classes (spec §10.1 cases (i), (ii), (ii-b), (ii-c); Appendix A accepted rows):
# Symbolics equations and conditions -> StrongFormSeam.analyse -> weak_form -> emit_kernels -> assemble_linear,
# against hand-written Ferrite assembly of K, M and F on the raw arrays, and the Dirichlet dofs and values of
# emit_dirichlet against a hand-written ConstraintHandler.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y k α β h
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)
const PARAMS = [k, α, β, h]
const PV = [2.0, 3.0, 0.5, 1.5]
const TOP = [0.0, 1.0]
const TOL = 1.0e-14
scalar_chain(eq, bcs; params = PARAMS) = chain([eq], bcs; depvars = [U], t, sivs = [x, y], params)

opaque(z) = 1 + z^2 + sin(z)   # registered without a derivative rule
@register_symbolic opaque(z)
fsrc(X) = X[1] * X[2] + 1
const DIR = BC("left", U ~ 0.1 * y)
const DIR_REF = [("left", :u, (X, t) -> 0.1 * X[2])]
const ROBIN_REF = ("top", 1, 1, X -> PV[2], X -> PV[3])

# Compare the chain's K, M, F and Dirichlet data with the hand-written reference.
function check_linear(setup, c; block, source = (e, X) -> fsrc(X), facets, dirichlet = DIR_REF, p = PV)
    A = assemble_linear(setup.ctx, c.kernels, p)
    check_operators(A, linear_reference(setup; block, source, facets)...; tol = TOL)
    ch, ch_ref = constraints(setup, c.dirichlet, p), constraints_reference(setup, dirichlet)
    @test ch.prescribed_dofs == ch_ref.prescribed_dofs
    @test dirichlet_values(ch) ≈ dirichlet_values(ch_ref) rtol = 1.0e-15
    return A
end

const SETUP = mesh_setup(Triangle, 1, 10)

@testset "(i) constant isotropic diffusion: Dirichlet, Neumann, Robin" begin
    c = scalar_chain(
        Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1,
        [DIR, BC("right", Dx(U) ~ h, RIGHT), BC("top", k * Dy(U) + α * U ~ β, TOP)]
    )
    A = check_linear(
        SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(PV[1]), m = 1.0),
        facets = [("right", 1, 1, X -> 0.0, X -> PV[1] * PV[4]), ROBIN_REF]
    )
    @test norm(A.K - A.K') ≤ TOL * norm(A.K)
    bilinear = only(v for v in c.wf.volume if v.kind === :bilinear)
    @test !bilinear.timedep && bilinear.paramdep
end

@testset "(i) on $(celltype) P$(order)" for (celltype, order) in
    ((Triangle, 1), (Triangle, 2), (Quadrilateral, 1), (Quadrilateral, 2))
    setup = mesh_setup(celltype, order, 4)
    c = scalar_chain(
        Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1,
        [DIR, BC("right", Dx(U) ~ h, RIGHT), BC("top", k * Dy(U) + α * U ~ β, TOP)]
    )
    check_linear(
        setup, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(PV[1]), m = 1.0),
        facets = [("right", 1, 1, X -> 0.0, X -> PV[1] * PV[4]), ROBIN_REF]
    )
end

@testset "(i) on Line P$(order)" for order in (1, 2)
    @variables u1(..)
    U1 = u1(t, x)
    setup = mesh_setup(Line, order, 8)
    c = chain(
        [Dt(U1) ~ Dx(k * Dx(U1)) + x + 1], [BC("left", U1 ~ 0.1), BC("right", k * Dx(U1) + α * U1 ~ β, [1.0])];
        depvars = [U1], t, sivs = [x], params = PARAMS
    )
    check_linear(
        setup, c; block = (e, b, X) -> (; zero_block(1)..., κ = PV[1] * one(Tensor{2, 1}), m = 1.0),
        source = (e, X) -> X[1] + 1, facets = [("right", 1, 1, X -> PV[2], X -> PV[3])],
        dirichlet = [("left", :u, (X, t) -> 0.1)]
    )
end

@testset "(i) on mesh pieces: NormalDerivative with the symbolic normal" begin
    # The same conditions as (i), written on the pieces of a MeshDomain over the grid's facet sets with
    # NormalDerivative, so PDEBase matches them against k∇u·N with the symbolic normal N. Needs MeshDomain pieces and
    # NormalDerivative slotting (PDEBase#154, the M0 stack); broken without them, and on the seam path.
    if PDEBASE_MESH
        Ω = mesh_domain(SETUP)
        left, right, top = (PDEBase.boundarypiece(Ω, name) for name in (:left, :right, :top))
        Dn(piece) = PDEBase.NormalDerivative(piece)
        c = chain(
            [Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1],
            [u(t, left) ~ 0.1 * y, Dn(right)(u(t, right)) ~ h, k * Dn(top)(u(t, top)) + α * u(t, top) ~ β];
            depvars = [U], t, sivs = [x, y], params = PARAMS, Ω
        )
        @test [b.piece for b in c.an.natural] == ["right", "top"]
        check_linear(
            SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(PV[1]), m = 1.0),
            facets = [("right", 1, 1, X -> 0.0, X -> PV[1] * PV[4]), ROBIN_REF]
        )
    else
        @test_broken PDEBASE_MESH
    end
end

@testset "(ii) variable isotropic diffusion k = 1 + x²y" begin
    kxy = 1 + x^2 * y
    kf(X) = 1 + X[1]^2 * X[2]
    c = scalar_chain(
        Dt(U) ~ Dx(kxy * Dx(U)) + Dy(kxy * Dy(U)) + x * y + 1,
        [DIR, BC("right", Dx(U) ~ h, RIGHT), BC("top", kxy * Dy(U) + α * U ~ β, TOP)]
    )
    check_linear(
        SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(kf(X)), m = 1.0),
        facets = [("right", 1, 1, X -> 0.0, X -> kf(X) * PV[4]), ROBIN_REF]
    )
end

@testset "(ii-b) non-divergence a(x,y)Δu with linear advection and reaction" begin
    axy = 1 + x^2 + y
    af(X) = 1 + X[1]^2 + X[2]
    c = scalar_chain(
        Dt(U) ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) - 0.7 * Dx(U) - 0.3 * Dy(U) - 2.5 * U + x * y + 1,
        [DIR, BC("right", Dx(U) ~ h, RIGHT), BC("top", axy * Dy(U) + α * U ~ β, TOP)]
    )
    # The product rule moves ∇a·∇u into the advection: β_eff = (0.7, 0.3) + ∇a.
    A = check_linear(
        SETUP, c;
        block = (e, b, X) -> (; κ = iso(af(X)), β = Vec(0.7 + 2X[1], 0.3 + 1.0), c = 2.5, m = 1.0),
        facets = [("right", 1, 1, X -> 0.0, X -> af(X) * PV[4]), ROBIN_REF]
    )
    @test norm(A.K - A.K') > 1.0e-3 * norm(A.K)
end

@testset "(ii-c) anisotropic with cross terms" begin
    k11, k12, k22 = 2 + x, 0.4 + 0.1y, 1 + y^2
    Kt(X) = Tensor{2, 2}((2 + X[1], 0.4 + 0.1X[2], 0.4 + 0.1X[2], 1 + X[2]^2))
    c = scalar_chain(
        Dt(U) ~ Dx(k11 * Dx(U) + k12 * Dy(U)) + Dy(k12 * Dx(U) + k22 * Dy(U)) + x * y + 1,
        [DIR, BC("top", k12 * Dx(U) + k22 * Dy(U) + α * U ~ β, TOP)]
    )
    check_linear(
        SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = Kt(X), m = 1.0), facets = [ROBIN_REF]
    )
end

@testset "mixed partial Dx(Dy(u))" begin
    c = scalar_chain(Dt(U) ~ Dx(Dy(U)) + Dx(Dx(U)) + Dy(Dy(U)) + x * y + 1, [DIR])
    κ = Tensor{2, 2}((1.0, 0.0, 1.0, 1.0))     # ∇v·κ∇w = v_x (w_x + w_y) + v_y w_y
    A = check_linear(SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ, m = 1.0), facets = [])
    @test norm(A.K - A.K') > 1.0e-3 * norm(A.K)
end

@testset "opaque registered coefficient in divergence form (accepted)" begin
    c = scalar_chain(Dt(U) ~ Dx(opaque(x) * Dx(U)) + Dy(opaque(x) * Dy(U)) + x * y + 1, [DIR])
    check_linear(SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(opaque(X[1])), m = 1.0), facets = [])
end

@testset "§10.1 regression: piecewise ifelse coefficient accepted in divergence form" begin
    # The rejected half (non-divergence ifelse·Δu) is @test_broken in rejections.jl until PDEBase's smoothness check.
    apw = ifelse(x < 0.5, 1.0, 3.0)
    c = scalar_chain(Dt(U) ~ Dx(apw * Dx(U)) + Dy(apw * Dy(U)) + x * y + 1, [DIR])
    check_linear(
        SETUP, c; block = (e, b, X) -> (; zero_block(2)..., κ = iso(X[1] < 0.5 ? 1.0 : 3.0), m = 1.0), facets = []
    )
end

@testset "Dirichlet: constant and time-dependent, at two times" begin
    c = scalar_chain(Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)), [BC("left", U ~ k), BC("bottom", U ~ sin(t) * x + h)])
    # Both paths emit conditions in input order; the corner dof both pieces share takes the value of the Dirichlet
    # the handler applies last, so the reference uses the same order.
    ref = [("left", :u, (X, t) -> PV[1]), ("bottom", :u, (X, t) -> sin(t) * X[1] + PV[4])]
    @test [(piece, field) for (piece, field, _) in c.dirichlet] == [("left", 1), ("bottom", 1)]
    for tv in (0.3, 1.1)
        ch, ch_ref = constraints(SETUP, c.dirichlet, PV; t = tv), constraints_reference(SETUP, ref; t = tv)
        @test ch.prescribed_dofs == ch_ref.prescribed_dofs
        @test dirichlet_values(ch) ≈ dirichlet_values(ch_ref) rtol = 1.0e-15
    end
    ch = constraints(SETUP, c.dirichlet, PV; t = 0.3)
    before = dirichlet_values(ch)
    update!(ch, 1.1)
    @test dirichlet_values(ch) ≈ dirichlet_values(constraints_reference(SETUP, ref; t = 1.1)) rtol = 1.0e-15
    @test !(dirichlet_values(ch) ≈ before)
end

@testset "facet terms are non-vanishing: sum(F) shifts by k·h·|Γ|" begin
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    with_h = assemble_linear(SETUP.ctx, scalar_chain(eq, [BC("right", Dx(U) ~ h, RIGHT)]).kernels, PV).F
    without_h = assemble_linear(SETUP.ctx, scalar_chain(eq, BC[]).kernels, PV).F
    @test sum(with_h) - sum(without_h) ≈ PV[1] * PV[4] * 2.0 rtol = 1.0e-13   # |right| = 2 on [−1, 1]²
end

@testset "time- and parameter-dependent coefficients: flags and reassembly" begin
    D = k * (1 + t * x^2)
    c = scalar_chain(Dt(U) ~ D * (Dx(Dx(U)) + Dy(Dy(U))) + exp(-t) * x, BC[])
    kinds = Dict(v.kind => v for v in c.wf.volume)
    @test kinds[:bilinear].timedep && kinds[:bilinear].paramdep
    @test kinds[:source].timedep && !kinds[:source].paramdep
    @test !kinds[:mass].timedep && !kinds[:mass].paramdep
    operators = map(((0.0, PV), (0.7, [3.5, 0.0, 0.0, 0.0]))) do (tv, p)
        A = assemble_linear(SETUP.ctx, c.kernels, p; t = tv)
        Df(X) = p[1] * (1 + tv * X[1]^2)
        Kr, Mr, Fr = linear_reference(
            SETUP; block = (e, b, X) -> (; κ = iso(Df(X)), β = Vec(2p[1] * tv * X[1], 0.0), c = 0.0, m = 1.0),
            source = (e, X) -> exp(-tv) * X[1]
        )
        check_operators(A, Kr, Mr, Fr; tol = TOL)
        A
    end
    @test relerr(operators[1].K, operators[2].K) > 0.1
    @test relerr(operators[1].F, operators[2].F) > 0.1
end

@testset "§10.1 regression: ProductRule vs Weighted(r²) on the spherical Laplacian" begin
    # ∂ₜu = r⁻² ∂ᵣ(r² ∂ᵣu) on r ∈ [0.5, 1.5], ∂ᵣu = 2 at r = 1.5. ProductRule gives the non-symmetric
    # ∫u'v' − ∫(2/r)u'v; Weighted(r²) gives the symmetric ∫r²u'v' with mass r² and facet datum r²·2.
    @variables r
    @variables radial(..)
    Dr = Differential(r)
    Ur = radial(t, r)
    setup = mesh_setup(Line, 1, 16; corners = (Vec(0.5), Vec(1.5)))
    eq = Dt(Ur) ~ 1 / r^2 * Dr(r^2 * Dr(Ur))
    bcs = [BC("right", Dr(Ur) ~ 2.0, [1.0])]
    one1 = one(Tensor{2, 1})
    domain = [(0.5, 1.5)]
    product = chain([eq], bcs; depvars = [Ur], t, sivs = [r], domain, ibp = Dict(1 => ProductRule()))
    weighted = chain([eq], bcs; depvars = [Ur], t, sivs = [r], domain, ibp = Dict(1 => Weighted(r^2)))
    Ap, Aw = (assemble_linear(setup.ctx, c.kernels, [0.0]) for c in (product, weighted))
    Kp, Mp, Fp = linear_reference(
        setup; block = (e, b, X) -> (; κ = one1, β = Vec(-2 / X[1]), c = 0.0, m = 1.0), source = (e, X) -> 0.0,
        facets = [("right", 1, 1, X -> 0.0, X -> 2.0)]
    )
    Kw, Mw, Fw = linear_reference(
        setup; block = (e, b, X) -> (; κ = X[1]^2 * one1, β = Vec(0.0), c = 0.0, m = X[1]^2),
        source = (e, X) -> 0.0, facets = [("right", 1, 1, X -> 0.0, X -> 2.0 * X[1]^2)]
    )
    check_operators(Ap, Kp, Mp, Fp; tol = TOL)
    check_operators(Aw, Kw, Mw, Fw; tol = TOL)
    @test norm(Aw.K - Aw.K') ≤ TOL * norm(Aw.K)
    @test norm(Ap.K - Ap.K') > 1.0e-2 * norm(Ap.K)
    # Spec §5.1: ProductRule warns exactly once when it generates a correction term, Weighted never. The warning
    # belongs to PDEBase's analyse; the seam does not emit it, so the first check is broken on the seam path.
    warnings(f) = count(log -> log.level == Base.CoreLogging.Warn, first(Test.collect_test_logs(f)))
    @test_pdebase warnings(() -> chain([eq], bcs; depvars = [Ur], t, sivs = [r], domain, ibp = Dict(1 => ProductRule()))) == 1
    @test warnings(() -> chain([eq], bcs; depvars = [Ur], t, sivs = [r], domain, ibp = Dict(1 => Weighted(r^2)))) == 0
end

@testset "periodic conditions are carried through weak_form untouched" begin
    # Checks only that weak_form passes periodic data through with its terms unchanged. Periodic pairing is
    # constraint-level (PeriodicDirichlet in M4), and rejecting periodic conditions before M4 belongs to the
    # discretizer (M2/M4), so this is not coverage of the periodic BC kind.
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    bcs = [BC("top", k * Dy(U) + α * U ~ β, TOP)]
    periodic = [PeriodicBC("left", "right", 1, [2.0, 0.0])]
    plain = scalar_chain(eq, bcs)
    paired = chain([eq], bcs; depvars = [U], t, sivs = [x, y], params = PARAMS, periodic)
    fields(bcs) = [(b.master, b.mirror, b.field, b.shift) for b in bcs]
    @test fields(paired.wf.analysis.periodic) == fields(periodic)
    terms(wf) = [(v.eq, v.kind, v.integrand, v.timedep, v.paramdep) for v in wf.volume]
    facet_terms(wf) = [(f.eq, f.piece, f.kind, f.integrand, f.timedep, f.paramdep) for f in wf.facet]
    @test isequal(terms(paired.wf), terms(plain.wf))
    @test isequal(facet_terms(paired.wf), facet_terms(plain.wf))
    A, B = (assemble_linear(SETUP.ctx, c.kernels, PV) for c in (paired, plain))
    @test A.K == B.K && A.M == B.M && A.F == B.F
    # The same pairing written as a trace equation, classified by PDEBase's stage d (seam path: broken).
    function traced()
        c = chain([eq], [bcs; u(t, -1.0, y) ~ u(t, 1.0, y)]; depvars = [U], t, sivs = [x, y], params = PARAMS)
        return fields(c.wf.analysis.periodic) == [("left", "right", 1, 2.0)] &&
            isequal(terms(c.wf), terms(plain.wf)) && isequal(facet_terms(c.wf), facet_terms(plain.wf))
    end
    @test_pdebase PDEBASE_PATH && traced()
end
