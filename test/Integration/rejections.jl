# End to end, every Appendix A rejected class the chain can raise, asserted by error type and message substring.
# Equation-level errors come from PDEBase.StrongForm.analyse when it is available (see references.jl) and from the
# test-only seam otherwise, so each substring is chosen per path by `wording(seam, pdebase)`; the PDEBase wording
# follows spec Appendix A. Boundary-condition errors still come from the seam's stage d on both paths, and the last
# testset's errors from the assembly driver.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))
using DomainSets: DomainSets

@variables t x y
@variables u(..) w(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U, Wf = u(t, x, y), w(t, x, y)

opaque(z) = 1 + z^2   # registered without a derivative rule
@register_symbolic opaque(z)

run_chain(eqs, bcs = BC[]; depvars = [U]) = chain(eqs, bcs; depvars, t, sivs = [x, y])
rejects(needle, eqs, bcs = BC[]; kw...) = raises(() -> run_chain(eqs, bcs; kw...), needle)
wording(seam, pdebase) = PDEBASE_PATH ? pdebase : seam

@testset "rejected equation classes" begin
    order_message = wording("operator needs mixed/C¹ formulation", "operator needs a mixed or C¹ formulation")
    @test rejects(order_message, [Dt(U) ~ Dx(Dx(Dx(U)))])
    @test rejects(order_message, [Dt(U) ~ -Dx(Dx(Dx(Dx(U)))) - Dy(Dy(Dy(Dy(U))))])
    # Spec m2 (KdV): the message names the actual order. The seam says "fourth-or-higher" for every order above two.
    @test_pdebase rejects("third-order", [Dt(U) ~ Dx(Dx(Dx(U)))])
    @test_pdebase rejects("fourth-order", [Dt(U) ~ -Dx(Dx(Dx(Dx(U))))])
    @test rejects(wording("not ∂ₜu × coefficient", "second order in time; write as a first-order system"), [Differential(t, 2)(U) ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U)^2 ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U) ~ Dx(Dt(U)) + Dx(Dx(U))])
    @test rejects("state-dependent mass", [(1 + U^2) * Dt(U) ~ Dx(Dx(U))])
    @test rejects("needs residual-form DAE", [(1 + U^2) * Dt(U) ~ Dx(Dx(U))])
    @test rejects("second derivative not in a divergence-compatible position", [Dt(U) ~ sin(Dx(Dx(U)))])
    # Appendix A files ∇u-dependent coefficients and products of second-order factors under one message.
    @test rejects(wording("coefficient depends on ∇u", "second derivative not in a divergence-compatible position"), [Dt(U) ~ Dx(U) * Dx(Dx(U))])
    @test rejects(wording("product of two second-order factors", "second derivative not in a divergence-compatible position"), [Dt(U) ~ Dx(Dx(U)) * Dy(Dy(U))])
end

@testset "opaque registered functions" begin
    # Accepted in divergence form with a coordinate argument: covered end to end in linear.jl.
    @test run_chain([Dt(U) ~ Dx(opaque(x) * Dx(U)) + Dy(opaque(x) * Dy(U))]) isa NamedTuple
    # State reaches the argument of a function without a derivative rule (PDEBase: the operator whitelist).
    @test rejects(
        wording("registered function without a derivative rule is applied to state", "is not supported by the strong-form analysis"),
        [Dt(U) ~ opaque(U) * Dx(U) + Dx(Dx(U))]
    )
    # Non-divergence: the product rule needs ∂ₓopaque, which has no rule (the smoothness check).
    @test rejects(wording("unresolved symbols", "is not differentiable"), [Dt(U) ~ opaque(x) * (Dx(Dx(U)) + Dy(Dy(U)))])
end

@testset "rejected boundary conditions" begin
    # Box-face conditions go through PDEBase's stage d on the PDEBase path and the seam's otherwise.
    eq = [Dt(U) ~ Dx((2 + x) * Dx(U) + 0.4 * Dy(U)) + Dy(0.4 * Dx(U) + (1 + y^2) * Dy(U))]
    @test rejects(
        wording("not a conormal (natural) condition", "is not a conormal condition for any equation's flux; write the conormal condition"),
        eq, [BC("right", Dx(U) ~ 1.0, [1.0, 0.0])]
    )
    @test rejects(
        wording("nonlinear Dirichlet BC", "Dirichlet data must be linear in one field"), [Dt(U) ~ Dx(Dx(U))], [BC("left", U^2 ~ 1.0)]
    )
    # Spec m1: an essential condition coupling two fields. The seam accepts it silently.
    two_field = [Dt(U) ~ Dx(Dx(U)), Dt(Wf) ~ Dy(Dy(Wf))]
    @test_pdebase rejects("Dirichlet data depends on another field", two_field, [BC("left", U + Wf ~ 1)]; depvars = [U, Wf])
end

@testset "rejections only PDEBase's analyse raises" begin
    @test_pdebase rejects("not differentiable", [Dt(U) ~ ifelse(x < 0.5, 1.0, 3.0) * (Dx(Dx(U)) + Dy(Dy(U)))])
    integral = Symbolics.Integral(x in DomainSets.ClosedInterval(0.0, 1.0))
    @test_pdebase rejects("is not supported", [Dt(U) ~ Dx(Dx(U)) + integral(U)])
end

# A raw trace equation reaches only PDEBase's boundary analysis; the seam takes interior-form conditions on pieces,
# so these rows are broken on the seam path.
rejects_trace(needle, bc) = PDEBASE_PATH && rejects(needle, [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [bc])

@testset "trace conditions on box faces" begin
    # Box-face form of "operator and trace on different pieces": the x-derivative of the left trace against the
    # right trace.
    @test_pdebase rejects_trace("relates derivatives on two faces", Dx(u(t, -1.0, y)) ~ u(t, 1.0, y))
    # Periodic traces whose faces are not related by a translation along one axis.
    @test_pdebase rejects_trace("periodic pairing needs a translation", u(t, -1.0, y) ~ u(t, x, 1.0))
end

# Rows that need piece and NormalDerivative slotting, which comes in PDEBase strongform PR 5 (not in 4f536e4). The
# probe flips (an unexpected pass) when NormalDerivative exists, the signal to write the real end-to-end rows.
has_normal_derivative() = isdefined(Symbolics, :NormalDerivative) || isdefined(PDEBase, :NormalDerivative)

@testset "rows awaiting PDEBase strongform PR 5 (piece and NormalDerivative slotting)" begin
    # Symbolic normal slot N in a facet kernel, from Dn_Γ(u) on a mesh piece; the seam only takes constant normals.
    @test_broken has_normal_derivative()
    # "NormalDerivative(Γa) applied to a trace on Γb" on mesh pieces.
    @test_broken has_normal_derivative()
end

@testset "boundary piece with no facet set (assembly driver)" begin
    setup = mesh_setup(Triangle, 1, 4)
    c = run_chain([Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [BC("inlet", Dx(U) ~ 1.0, [-1.0, 0.0])])
    @test_throws "no facet set is given for piece inlet" assemble_linear(setup.ctx, c.kernels, [0.0])
    @test_throws "no facet set named" AssemblyContext(setup.dh, [:u], setup.cv, setup.fv, Dict("inlet" => "inlet"))
end
