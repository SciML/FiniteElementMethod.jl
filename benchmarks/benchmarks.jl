# AirspeedVelocity suite for the "Benchmark this PR" workflow: DAEProblem construction of a heat equation on
# P1 triangles at small sizes (see construction.jl for the full N = 1e4 study).
using BenchmarkTools, FiniteElementMethod, ModelingToolkit, DomainSets, Ferrite
using SciMLBase: discretize
using ModelingToolkit: t_nounits as t

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

function heat_system(side)
    Ω = MeshDomain(FiniteElementMethod.rectangle_grid(Triangle, (side - 1, side - 1)))
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x + 2y)
    bcs = [[u(t, Γ) ~ (1 + x + 2y) * (1 + t) for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ 1 + x + 2y]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    return pdesys, FEMDiscretization(Ω, t)
end

const SUITE = BenchmarkGroup()
SUITE["construction"] = BenchmarkGroup()
for side in (10, 20)
    pdesys, disc = heat_system(side)
    SUITE["construction"]["heat P1 $(side)x$(side)"] = @benchmarkable discretize($pdesys, $disc)
end
