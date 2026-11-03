# End to end, multi-field classes (spec §10.1 two-field row; Appendix A coupled flux, algebraic rows, wave as a
# first-order system): block K, M, F against hand-written Ferrite assembly, the full residual against a
# hand-written residual, the tangent against central differences, and Dirichlet routing to the second field.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y Du Dw χ a b
@variables u(..) w(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U, Wf = u(t, x, y), w(t, x, y)
block_norms(A, setup) = [norm(A[I, J]) for I in field_dofs.(Ref(setup), setup.fields), J in field_dofs.(Ref(setup), setup.fields)]

@testset "two-field reaction-diffusion with cross-diffusion (Schnakenberg)" begin
    params, pv = [Du, Dw, χ, a, b], [1.0, 20.0, 0.3, 0.1, 0.9]
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a - U + U^2 * Wf,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U^2 * Wf,
    ]
    bcs = [
        BC("right", Du * Dx(U) + χ * Dx(Wf) ~ 0.1, RIGHT),   # conormal of equation 1
        BC("right", Dx(Wf) ~ 0.2, RIGHT),                    # ∂ₙw, routed to equation 2 and rescaled by Dw
        BC("left", Wf ~ 0.9),                                # essential, routed to field 2
    ]
    c = chain(eqs, bcs; depvars = [U, Wf], t, sivs = [x, y], params)
    @test [(f.eq, f.kind) for f in c.wf.facet] == [(1, :source), (2, :source)]
    setup = mesh_setup(Triangle, 1, 16; fields = [:u, :w])
    Dup, Dwp, χp, ap, bp = pv

    blocks = Dict(
        (1, 1) => (; κ = iso(Dup), β = zero(Vec{2}), c = 1.0, m = 1.0),
        (1, 2) => (; κ = iso(χp), β = zero(Vec{2}), c = 0.0, m = 0.0),
        (2, 1) => zero_block(2),
        (2, 2) => (; κ = iso(Dwp), β = zero(Vec{2}), c = 0.0, m = 1.0),
    )
    A = assemble_linear(setup.ctx, c.kernels, pv)
    Kr, Mr, Fr = linear_reference(
        setup; block = (e, f, X) -> blocks[(e, f)], source = (e, X) -> e == 1 ? ap : bp,
        facets = [("right", 1, 1, X -> 0.0, X -> 0.1), ("right", 2, 2, X -> 0.0, X -> 0.2Dwp)]
    )
    check_operators(A, Kr, Mr, Fr)
    # Block norms printed by scratch/dauntless/test_multifield.out on the same 16×16 P1 mesh.
    @test all(isapprox.(block_norms(A.K, setup), [69.6 20.9; 0.0 1390.0]; rtol = 5.0e-3, atol = 1.0e-12))
    @test all(isapprox.(block_norms(A.M, setup), [0.131 0.0; 0.0 0.131]; rtol = 5.0e-3, atol = 1.0e-12))

    u_rand = rand(Xoshiro(2), ndofs(setup.dh))
    rg, Jg = chain_residual(setup, c.kernels, u_rand, pv)
    rr = residual_reference(
        setup, u_rand;
        flux = (e, Uq, ∇Uq, X) -> e == 1 ? Dup * ∇Uq[1] + χp * ∇Uq[2] : Dwp * ∇Uq[2],
        source = (e, Uq, ∇Uq, X) -> e == 1 ? -(ap - Uq[1] + Uq[1]^2 * Uq[2]) : -(bp - Uq[1]^2 * Uq[2]),
        facets = [("right", 1, (Uq, X) -> -0.1), ("right", 2, (Uq, X) -> -0.2Dwp)]
    )
    @test relerr(rg, rr) ≤ 1.0e-14
    @test tangent_error(v -> chain_residual(setup, c.kernels, v, pv)[1], Jg, u_rand) ≤ 1.0e-9

    ch = constraints(setup, c.dirichlet, pv)
    ch_ref = constraints_reference(setup, [("left", :w, (X, t) -> 0.9)])
    @test ch.prescribed_dofs == ch_ref.prescribed_dofs
    @test dirichlet_values(ch) ≈ dirichlet_values(ch_ref) rtol = 1.0e-15
    @test issubset(ch.prescribed_dofs, field_dofs(setup, :w))
end

@testset "algebraic row 0 ~ Δv + e⁻ᵗ sin x: zero mass block, time-dependent source" begin
    eqs = [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + Wf, 0 ~ Dx(Dx(Wf)) + Dy(Dy(Wf)) + exp(-t) * sin(x)]
    c = chain(eqs, [BC("left", U ~ 0.0), BC("left", Wf ~ 0.0)]; depvars = [U, Wf], t, sivs = [x, y])
    @test count(v -> v.kind === :mass, c.wf.volume) == 1
    @test only(v for v in c.wf.volume if v.kind === :source).timedep
    setup = mesh_setup(Quadrilateral, 1, 8; fields = [:u, :w])
    blocks = Dict(
        (1, 1) => (; zero_block(2)..., κ = iso(1.0), m = 1.0),
        (1, 2) => (; zero_block(2)..., c = -1.0),
        (2, 1) => zero_block(2),
        (2, 2) => (; zero_block(2)..., κ = iso(1.0)),
    )
    w_dofs = field_dofs(setup, :w)
    sources = map((0.0, 0.5)) do tv
        A = assemble_linear(setup.ctx, c.kernels, P0; t = tv)
        Kr, Mr, Fr = linear_reference(
            setup; block = (e, f, X) -> blocks[(e, f)], source = (e, X) -> e == 2 ? exp(-tv) * sin(X[1]) : 0.0
        )
        check_operators(A, Kr, Mr, Fr)
        @test norm(A.M[w_dofs, :]) == 0 && norm(A.M[:, w_dofs]) == 0
        A.F
    end
    @test sources[2] ≈ exp(-0.5) * sources[1] rtol = 1.0e-14
end

@testset "wave equation as a first-order system, 1D P$(order)" for order in (1, 2)
    @variables p1(..) q1(..)
    P1, Q1 = p1(t, x), q1(t, x)
    c = chain(
        [Dt(P1) ~ Q1, Dt(Q1) ~ Dx(Dx(P1))], [BC("left", P1 ~ 0.0), BC("right", P1 ~ 0.0)];
        depvars = [P1, Q1], t, sivs = [x]
    )
    setup = mesh_setup(Line, order, 12; fields = [:p, :q])
    one1 = one(Tensor{2, 1})
    blocks = Dict(
        (1, 1) => (; zero_block(1)..., m = 1.0),
        (1, 2) => (; zero_block(1)..., c = -1.0),
        (2, 1) => (; zero_block(1)..., κ = one1),
        (2, 2) => (; zero_block(1)..., m = 1.0),
    )
    A = assemble_linear(setup.ctx, c.kernels, P0)
    Kr, Mr, Fr = linear_reference(setup; block = (e, f, X) -> blocks[(e, f)], source = (e, X) -> 0.0)
    check_operators(A, Kr, Mr, Fr)
    @test norm(A.F) == 0
    ch = constraints(setup, c.dirichlet, P0)
    ch_ref = constraints_reference(setup, [("left", :p, (X, t) -> 0.0), ("right", :p, (X, t) -> 0.0)])
    @test ch.prescribed_dofs == ch_ref.prescribed_dofs
end

@testset "declared field order [u, w] survives an equation that mentions w first" begin
    # Regression: PDEBase once numbered fields in the order the equations mention them, which put w first here and
    # sent u's equation to w's dofs. Fields must follow the declared depvars, so the u block is diffusion 2 with
    # mass, and the w block diffusion 5 coupled to u, with no mass.
    eqs = [Dt(U) ~ Wf + 2 * (Dx(Dx(U)) + Dy(Dy(U))), 0 ~ 5 * (Dx(Dx(Wf)) + Dy(Dy(Wf))) + U]
    c = chain(eqs, BC[]; depvars = [U, Wf], t, sivs = [x, y])
    s = c.an.slots
    @test isequal(c.an.mass[1], s.Ud[1]) && isequal(c.an.mass[2], 0)
    setup = mesh_setup(Triangle, 1, 6; fields = [:u, :w])
    blocks = Dict(
        (1, 1) => (; zero_block(2)..., κ = iso(2.0), m = 1.0),
        (1, 2) => (; zero_block(2)..., c = -1.0),
        (2, 1) => (; zero_block(2)..., c = -1.0),
        (2, 2) => (; zero_block(2)..., κ = iso(5.0)),
    )
    A = assemble_linear(setup.ctx, c.kernels, P0)
    check_operators(A, linear_reference(setup; block = (e, f, X) -> blocks[(e, f)], source = (e, X) -> 0.0)...)
end
