# Device assembly end to end on one backend: `discretize` with `backend` set gives the operators of the host assembly,
# and the returned problem solves to the host solution.
using Test
# The four triggers of FiniteElementMethodKAExt.
using KernelAbstractions: KernelAbstractions
using Adapt: Adapt
using GPUArrays: GPUArrays
using GPUArraysCore: GPUArraysCore
using FiniteElementMethod: FiniteElementMethod as FEM, FEMDiscretization, MeshDomain, boundarypiece
using ModelingToolkit: ModelingToolkit, PDESystem, Differential, @parameters, @named, t_nounits as t
using DomainSets: Interval
using Ferrite: Ferrite, generate_grid, Hexahedron, Triangle, Vec, Lagrange, RefTriangle
using OrdinaryDiffEqBDF: DFBDF, solve, discretize
using Symbolics: @variables

# The operators of `_lower` before the Dirichlet rows are replaced, and the Dirichlet dofs.
function lowered(pdesys, disc)
    parts = FEM._lower(pdesys, disc)
    return (; parts.raw.M, parts.raw.K, F = parts.raw.load, parts.raw.prescribed)
end

relerr(a, b) = maximum(abs, a - b) / max(maximum(abs, b), eps())

# Against the host Float64 assembly; Float32 rounding of the coefficients and shape functions is about 1e-7.
optol(::Type{Float32}) = 1.0e-5
optol(::Type{Float64}) = 1.0e-12
# Solutions of the host and the device problem, both solved with reltol 1e-8.
soltol(::Type{Float32}) = 1.0e-4
soltol(::Type{Float64}) = 1.0e-7

@variables x y z u(..) v(..)
@parameters k
const Dx, Dy, Dz, Dt = Differential(x), Differential(y), Differential(z), Differential(t)

# 3D P1 on hexahedra: a variable conductivity with a parameter, a source, Dirichlet data on one face.
function heat3d(g; n = 3)
    Ω = MeshDomain(generate_grid(Hexahedron, (n, n, n), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0)))
    U = u(t, x, y, z)
    c = 1 + x^2 + k * y * z
    eq = Dt(U) ~ Dx(c * Dx(U)) + Dy(c * Dy(U)) + Dz(c * Dz(U)) + x * y + 1
    bcs = [u(t, boundarypiece(Ω, :left)) ~ g(y, z), u(0, x, y, z) ~ x * (1 - x)]
    @named pdesys = PDESystem([eq], bcs, [(x, y, z) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y, z], [U], [k => 0.3])
    return pdesys, Ω, Dict(), (0.4, 0.3, 0.6)
end

# 2D P2 on triangles.
function heat2d_p2(g; n = 4)
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    U = u(t, x, y)
    eq = Dt(U) ~ Dx((1 + k * x) * Dx(U)) + Dy((1 + k * x) * Dy(U)) + sin(x) * y
    bcs = [u(t, boundarypiece(Ω, :left)) ~ g(y, 0), u(t, boundarypiece(Ω, :right)) ~ 0.5, u(0, x, y) ~ x * y]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [U], [k => 0.3])
    return pdesys, Ω, Dict(u => Lagrange{RefTriangle, 2}()), (0.4, 0.7)
end

# Two fields of different orders coupled by linear terms, as in a Taylor–Hood pair.
function two_field(g; n = 3)
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    U, V = u(t, x, y), v(t, x, y)
    eqs = [Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + Dx(V) + 1, Dt(V) ~ 0.5 * (Dx(Dx(V)) + Dy(Dy(V))) - U + k * Dy(U)]
    left = boundarypiece(Ω, :left)
    bcs = [u(t, left) ~ g(y, 0), v(t, left) ~ 0, u(0, x, y) ~ 0, v(0, x, y) ~ x]
    @named pdesys = PDESystem(eqs, bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.1)], [t, x, y], [U, V], [k => 0.3])
    return pdesys, Ω, Dict(u => Lagrange{RefTriangle, 2}(), v => Lagrange{RefTriangle, 1}()), (0.4, 0.7)
end

const CONSTANT_G = (a, b) -> 1.0
const TIMEDEP_G = (a, b) -> sin(10t) * (1 + a) + b
const ASSEMBLY_CASES = [
    "3D P1 hexahedron" => heat3d,
    "2D P2 triangle" => heat2d_p2,
    "two fields, P2 and P1" => two_field,
]

# Every case with constant and time-dependent Dirichlet data, in both precisions and both assembly modes: the
# operators against the host assembly, and the solution at a point against the host solution.
function test_device_assembly(backend; precisions = (Float32, Float64), worst = Dict{Any, Float64}())
    @testset "$name, $gname Dirichlet data on $(nameof(typeof(backend)))" for (name, case) in ASSEMBLY_CASES,
            (gname, g) in (("constant", CONSTANT_G), ("time-dependent", TIMEDEP_G))
        pdesys, Ω, spaces, point = case(g)
        host_disc = FEMDiscretization(Ω, t; spaces, boundary_check = :off)
        host = lowered(pdesys, host_disc)
        host_sol = solve(discretize(pdesys, host_disc), DFBDF(); reltol = 1.0e-8, abstol = 1.0e-10)
        dvs = pdesys.dvs
        for T in precisions, assembly in (:colored, :atomic)
            disc = FEMDiscretization(Ω, t; spaces, boundary_check = :off, backend, precision = T, assembly)
            dev = lowered(pdesys, disc)
            @test dev.prescribed == host.prescribed
            for (op, a, b) in (("M", dev.M, host.M), ("K", dev.K, host.K), ("F", dev.F, host.F))
                err = relerr(a, b)
                worst[(op, T)] = max(get(worst, (op, T), 0.0), err)
                @test err < optol(T)
            end
            sol = solve(discretize(pdesys, disc), DFBDF(); reltol = 1.0e-8, abstol = 1.0e-10)
            for dv in dvs
                ref = host_sol(0.1, point...; dv)
                err = abs(sol(0.1, point...; dv) - ref) / max(abs(ref), 1.0)
                worst[("solution", T)] = max(get(worst, ("solution", T), 0.0), err)
                @test err < soltol(T)
            end
        end
    end
    return worst
end
