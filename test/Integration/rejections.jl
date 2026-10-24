# End to end, every Appendix A rejected class the current chain can raise, asserted by error type and message
# substring. Except for the last testset (the assembly driver's piece check), each error comes from the
# test-only StrongFormSeam and MOVES TO PDEBase.StrongForm.analyse when the seam is swapped; the messages there
# follow spec Appendix A, so the substrings below will need updating at the swap.
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

@testset "rejected equation classes (seam; moves to PDEBase analyse)" begin
    @test_throws ErrorException run_chain([Dt(U) ~ Dx(Dx(Dx(U)))])
    # Third order: spec m2 wants the message to name the actual order ("third-order"); the seam says
    # "fourth-or-higher" for every order above two.
    @test rejects("needs mixed/C¹ formulation", [Dt(U) ~ Dx(Dx(Dx(U)))])
    @test rejects("needs mixed/C¹ formulation", [Dt(U) ~ -Dx(Dx(Dx(Dx(U)))) - Dy(Dy(Dy(Dy(U))))])
    @test rejects("not ∂ₜu × coefficient", [Differential(t, 2)(U) ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U)^2 ~ Dx(Dx(U))])
    @test rejects("not ∂ₜu × coefficient", [Dt(U) ~ Dx(Dt(U)) + Dx(Dx(U))])
    @test rejects("state-dependent mass", [(1 + U^2) * Dt(U) ~ Dx(Dx(U))])
    @test rejects("second derivative not in a divergence-compatible position", [Dt(U) ~ sin(Dx(Dx(U)))])
    @test rejects("coefficient depends on ∇u", [Dt(U) ~ Dx(U) * Dx(Dx(U))])
    @test rejects("product of two second-order factors", [Dt(U) ~ Dx(Dx(U)) * Dy(Dy(U))])
end

@testset "opaque registered functions (seam; moves to PDEBase analyse)" begin
    # Accepted in divergence form with a coordinate argument: covered end to end in linear.jl.
    @test run_chain([Dt(U) ~ Dx(opaque(x) * Dx(U)) + Dy(opaque(x) * Dy(U))]) isa NamedTuple
    # State reaches the argument of a function without a derivative rule.
    @test rejects("registered function without a derivative rule is applied to state", [Dt(U) ~ opaque(U) * Dx(U) + Dx(Dx(U))])
    # Non-divergence: the product rule needs ∂ₓopaque, which has no rule (the smoothness check).
    @test rejects("unresolved symbols", [Dt(U) ~ opaque(x) * (Dx(Dx(U)) + Dy(Dy(U)))])
end

@testset "rejected boundary conditions (seam; moves to PDEBase analyse)" begin
    eq = [Dt(U) ~ Dx((2 + x) * Dx(U) + 0.4 * Dy(U)) + Dy(0.4 * Dx(U) + (1 + y^2) * Dy(U))]
    @test rejects("not a conormal (natural) condition", eq, [BC("right", Dx(U) ~ 1.0, [1.0, 0.0])])
    @test rejects("nonlinear Dirichlet BC", [Dt(U) ~ Dx(Dx(U))], [BC("left", U^2 ~ 1.0)])
end

@testset "rejections PDEBase must add that the seam does not raise" begin
    # Each is accepted silently today; these flip to passing when PDEBase.StrongForm.analyse is swapped in.
    two_field = [Dt(U) ~ Dx(Dx(U)), Dt(Wf) ~ Dy(Dy(Wf))]
    @test_broken rejects("depends on another field", two_field, [BC("left", U + Wf ~ 1)]; depvars = [U, Wf])
    @test_broken rejects("not differentiable", [Dt(U) ~ ifelse(x < 0.5, 1.0, 3.0) * (Dx(Dx(U)) + Dy(Dy(U)))])
    integral = Symbolics.Integral(x in DomainSets.ClosedInterval(0.0, 1.0))
    @test_broken rejects("is not supported", [Dt(U) ~ Dx(Dx(U)) + integral(U)])
end

@testset "boundary piece with no facet set (assembly driver)" begin
    setup = mesh_setup(Triangle, 1, 4)
    c = run_chain([Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))], [BC("inlet", Dx(U) ~ 1.0, [-1.0, 0.0])])
    @test_throws "no facet set is given for piece inlet" assemble_linear(setup.ctx, c.kernels, [0.0])
    @test_throws "no facet set named" AssemblyContext(setup.dh, [:u], setup.cv, setup.fv, Dict("inlet" => "inlet"))
end
