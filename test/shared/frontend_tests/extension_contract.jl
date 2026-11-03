# The Extension contract page (docs/src/extension_contract.md) states the keyword sets, messages and types of the
# front end; these tests keep the page and the code in step.
using Test
using FiniteElementMethod: FiniteElementMethod, FEMMetadata, rectangle_grid
using Ferrite: Triangle

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using PDEBase: PDEBase

const PAGE = read(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "extension_contract.md"), String)

message(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : rethrow()
end

@testset verbose = true "extension contract page" begin
    reserved = FiniteElementMethod._RESERVED_KWARGS
    derived = keys(FiniteElementMethod._DERIVED_KWARGS)
    @test occursin(join(("`$k`" for k in reserved), ", ", " and "), PAGE)
    @test all(k -> occursin("| `$k`", PAGE), derived)
    @test length(reserved) + length(derived) == 10
    @test occursin("Ten keywords are reserved", PAGE)

    @variables x y u(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))
    bcs = [[u(t, Γ) ~ 0 for Γ in boundarypieces(Ω)]; u(0, x, y) ~ 0]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    disc = FEMDiscretization(Ω, t)
    @test occursin(message(() -> FEMDiscretization(Ω, t; tspan = (0.0, 2.0))), PAGE)
    @test occursin(message(() -> discretize(pdesys, disc, 1)), PAGE)
    @test startswith(message(() -> discretize(1, disc)), "FEMDiscretization discretizes a ModelingToolkit.PDESystem; got ")
    @test occursin("FEMDiscretization discretizes a ModelingToolkit.PDESystem; got ...", PAGE)

    @test FEMMetadata <: SciMLBase.AbstractDiscretizationMetadata
    @test discretize(pdesys, disc).problem_type isa FEMMetadata
    @test FEMDiscretization <: PDEBase.AbstractMeshDiscretization <: SciMLBase.AbstractDiscretization
end
