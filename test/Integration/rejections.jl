# End to end, every Appendix A rejected class the chain can raise, asserted by error type and message substring.
# Equation-level errors come from PDEBase.StrongForm.analyse (see references.jl), worded as spec Appendix A.
# Errors for seam `BC`s on facet sets that are not box faces come from the seam's stage d, and the last testset's from the assembly driver.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))
using DomainSets: DomainSets

@variables t x y
@variables u(..) w(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U, Wf = u(t, x, y), w(t, x, y)

opaque(z) = 1 + z^2   # registered without a derivative rule
@register_symbolic opaque(z)

run_chain(eqs, bcs = BC[]; depvars = [U], kw...) = chain(eqs, bcs; depvars, t, sivs = [x, y], kw...)
rejects(needle, eqs, bcs = BC[]; kw...) = raises(() -> run_chain(eqs, bcs; kw...), needle)

@testset "rejected equation classes" begin
    order_message = "operator needs a mixed or C¹ formulation"
    @test rejects(order_message, [Dt(U) ~ Dx(Dx(Dx(U)))])
    @test rejects(order_message, [Dt(U) ~ -Dx(Dx(Dx(Dx(U)))) - Dy(Dy(Dy(Dy(U))))])
    # Spec m2 (KdV): the message names the actual order.
    @test rejects("third-order", [Dt(U) ~ Dx(Dx(Dx(U)))])
    @test rejects("fourth-order", [Dt(U) ~ -Dx(Dx(Dx(Dx(U))))])
    @test rejects("second order in time; write as a first-order system", [Differential(t, 2)(U) ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U)^2 ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U) ~ Dx(Dt(U)) + Dx(Dx(U))])
    @test rejects("state-dependent mass", [(1 + U^2) * Dt(U) ~ Dx(Dx(U))])
    @test rejects("needs residual-form DAE", [(1 + U^2) * Dt(U) ~ Dx(Dx(U))])
    @test rejects("second derivative not in a divergence-compatible position", [Dt(U) ~ sin(Dx(Dx(U)))])
    # Appendix A files ∇u-dependent coefficients and products of second-order factors under one message.
    @test rejects("second derivative not in a divergence-compatible position", [Dt(U) ~ Dx(U) * Dx(Dx(U))])
    @test rejects("second derivative not in a divergence-compatible position", [Dt(U) ~ Dx(Dx(U)) * Dy(Dy(U))])
end

@testset "opaque registered functions" begin
    # Accepted in divergence form with a coordinate argument: covered end to end in linear.jl.
    @test run_chain([Dt(U) ~ Dx(opaque(x) * Dx(U)) + Dy(opaque(x) * Dy(U))]) isa NamedTuple
    # State reaches the argument of a function without a derivative rule (PDEBase: the operator whitelist).
    @test rejects(
        "is not supported by the strong-form analysis",
        [Dt(U) ~ opaque(U) * Dx(U) + Dx(Dx(U))]
    )
    # Non-divergence: the product rule needs ∂ₓopaque, which has no rule (the smoothness check).
    @test rejects("is not differentiable", [Dt(U) ~ opaque(x) * (Dx(Dx(U)) + Dy(Dy(U)))])
end

@testset "rejected boundary conditions" begin
    # Box-face conditions are analysed by PDEBase, which needs the chain's domain to read left and right as faces.
    box = [(-1.0, 1.0), (-1.0, 1.0)]
    eq = [Dt(U) ~ Dx((2 + x) * Dx(U) + 0.4 * Dy(U)) + Dy(0.4 * Dx(U) + (1 + y^2) * Dy(U))]
    @test rejects(
        "is not a conormal condition for any equation's flux; write the conormal condition",
        eq, [BC("right", Dx(U) ~ 1.0, [1.0, 0.0])]; domain = box
    )
    @test rejects(
        "Dirichlet data must be linear in one field", [Dt(U) ~ Dx(Dx(U))], [BC("left", U^2 ~ 1.0)]; domain = box
    )
    # Spec m1: an essential condition coupling two fields.
    two_field = [Dt(U) ~ Dx(Dx(U)), Dt(Wf) ~ Dy(Dy(Wf))]
    @test rejects("Dirichlet data depends on another field", two_field, [BC("left", U + Wf ~ 1)]; depvars = [U, Wf], domain = box)
end

@testset "rejections only PDEBase's analyse raises" begin
    @test rejects("not differentiable", [Dt(U) ~ ifelse(x < 0.5, 1.0, 3.0) * (Dx(Dx(U)) + Dy(Dy(U)))])
    integral = Symbolics.Integral(x in DomainSets.ClosedInterval(0.0, 1.0))
    @test rejects("is not supported", [Dt(U) ~ Dx(Dx(U)) + integral(U)])
end

# A raw trace equation reaches PDEBase's boundary analysis; the seam takes interior-form conditions on pieces.
rejects_trace(needle, bc) = rejects(needle, [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [bc])

@testset "trace conditions on box faces" begin
    # Box-face form of "operator and trace on different pieces": the x-derivative of the left trace against the
    # right trace.
    @test rejects_trace("relates derivatives on two faces", Dx(u(t, -1.0, y)) ~ u(t, 1.0, y))
    # Periodic-style traces on faces of different axes, not related by a translation along one axis.
    @test rejects_trace("relates values on two faces of different axes", u(t, -1.0, y) ~ u(t, x, 1.0))
end

# Conditions on the pieces of a MeshDomain go through PDEBase's piece and NormalDerivative slotting. `makebc(Γ, Dn)` builds the condition from piece and operator constructors.
function mesh_rejects(needle, makebc)
    Ω = mesh_domain(mesh_setup(Triangle, 1, 2))
    bc = makebc(name -> PDEBase.boundarypiece(Ω, name), PDEBase.NormalDerivative)
    return rejects(needle, [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [bc]; Ω)
end

@testset "conditions on mesh pieces" begin
    # "NormalDerivative(Γa) applied to a trace on Γb" (Appendix A).
    @test mesh_rejects("not on Γ_right", (Γ, Dn) -> Dn(Γ(:right))(u(t, Γ(:left))) ~ 1.0)
    # A Cartesian derivative of a trace on a mesh piece is not its normal derivative.
    @test mesh_rejects("write the normal derivative", (Γ, Dn) -> Dx(u(t, Γ(:right))) ~ 1.0)
end

@testset "boundary piece with no facet set (assembly driver)" begin
    setup = mesh_setup(Triangle, 1, 4)
    Ω = mesh_domain(setup; names = [:left, :right, :bottom, :top, :inlet])
    inlet = PDEBase.boundarypiece(Ω, :inlet)
    c = run_chain([Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [PDEBase.NormalDerivative(inlet)(u(t, inlet)) ~ 1.0]; Ω)
    # FEM-owned errors: exact type and message.
    @test_throws ArgumentError("no facet set is given for piece inlet: add it to the pieces of the AssemblyContext") assemble_linear(setup.ctx, c.kernels, [0.0])
    @test_throws ArgumentError("piece inlet: the grid has no facet set named \"inlet\"") AssemblyContext(setup.dh, [:u], setup.cv, setup.fv, Dict("inlet" => "inlet"))
end
