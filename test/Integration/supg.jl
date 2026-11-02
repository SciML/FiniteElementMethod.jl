# End to end, SUPG on an advection-dominated boundary layer (spec §1 goal 4, §10.1 SUPG row): steady
# −εu″ + u′ = 0 on (0, 1), u(0) = 0, u(1) = 1, exact u = (e^{x/ε} − 1)/(e^{1/ε} − 1), with cell Péclet number
# Pe = |b|h/(2ε) > 1. Galerkin P1 oscillates; SUPG P1 with the optimal τ = h/(2|b|)(coth Pe − 1/Pe) is nodally
# exact and monotone.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))
using FiniteElementMethod: SUPG

@variables t x
@variables u(..)
Dx = Differential(x)
U = u(t, x)
const EPS = 0.01
exact(X) = expm1(X[1] / EPS) / expm1(1 / EPS)

function boundary_layer(nel, test_space)
    c = chain(
        [0 ~ EPS * Dx(Dx(U)) - Dx(U)], [BC("left", U ~ 0.0), BC("right", U ~ 1.0)];
        depvars = [U], t, sivs = [x], test_space
    )
    setup = mesh_setup(Line, 1, nel; corners = (Vec(0.0), Vec(1.0)), hessians = true)
    return setup, linear_solve(setup, c.kernels, constraints(setup, c.dirichlet, P0), P0)
end

optimal_τ(hK, b) = (Pe = abs(b[1]) * hK / (2EPS); hK / (2abs(b[1])) * (coth(Pe) - 1 / Pe))
nodal(setup, uh) = evaluate_at_grid_nodes(setup.dh, uh, :u)

@testset "Galerkin oscillates at cell Péclet number 2.5" begin
    setup, uh = boundary_layer(20, Galerkin())
    values = nodal(setup, uh)
    @test minimum(values) < -0.1                  # undershoot upstream of the layer
    @test nodal_error(setup, uh, exact) > 0.1
end

@testset "SUPG with the optimal τ is monotone and nodally exact, nel = $nel" for nel in (10, 20, 40, 80)
    setup, uh = boundary_layer(nel, SUPG(optimal_τ))
    values = nodal(setup, uh)
    @test minimum(values) ≥ -1.0e-10 && maximum(values) ≤ 1 + 1.0e-10
    @test nodal_error(setup, uh, exact) ≤ 1.0e-10
end

@testset "SUPG with τ = h/(2|b|) converges in L2 and stays monotone" begin
    τ(hK, b) = hK / (2abs(b[1]))
    errs = map((20, 40, 80, 160)) do nel
        setup, uh = boundary_layer(nel, SUPG(τ))
        values = nodal(setup, uh)
        @test minimum(values) ≥ -1.0e-10 && maximum(values) ≤ 1 + 1.0e-10
        l2_error(setup, uh, exact)
    end
    @test issorted(errs; rev = true)
    @test errs[end] < errs[1] / 4
end
