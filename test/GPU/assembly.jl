# Device assembly through `discretize` on KernelAbstractions.CPU() and JLArrays: the operators equal the host
# assembly and the problems solve to the host solution (`common/assembly.jl`). Both backends run on the host, so they
# test the portable code and host-array discipline, not that the kernels compile for a GPU (see README.md).
using JLArrays: JLBackend
using GenericSparseArrays: GenericSparseArrays
include(joinpath(@__DIR__, "common", "assembly.jl"))

GPUArraysCore.allowscalar(false)

@testset "KernelAbstractions.CPU()" begin
    println("largest relative errors on KernelAbstractions.CPU(): ", test_device_assembly(KernelAbstractions.CPU()))
end

@testset "JLArrays" begin
    println("largest relative errors on JLArrays: ", test_device_assembly(JLBackend()))
end

@testset "colored assembly gives the same operators on every run" begin
    pdesys, Ω, spaces, _ = heat3d(CONSTANT_G)
    disc = FEMDiscretization(Ω, t; spaces, boundary_check = :off, backend = JLBackend(), precision = Float32)
    a, b = lowered(pdesys, disc), lowered(pdesys, disc)
    @test a.M == b.M && a.K == b.K && a.F == b.F
end

@testset "a steady problem assembled on the device solves to the host solution" begin
    Ω = MeshDomain(generate_grid(Triangle, (4, 4), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    U = u(x, y)
    @named pdesys = PDESystem(
        [0 ~ Dx(Dx(U)) + Dy(Dy(U)) + 1 + x], [u(boundarypiece(Ω, :left)) ~ 0, u(boundarypiece(Ω, :right)) ~ y],
        [(x, y) ∈ Ω], [x, y], [U]
    )
    host = solve(discretize(pdesys, FEMDiscretization(Ω)))
    for backend in (KernelAbstractions.CPU(), JLBackend())
        dev = solve(discretize(pdesys, FEMDiscretization(Ω; backend)))
        @test dev(0.4, 0.7; dv = U) ≈ host(0.4, 0.7; dv = U) rtol = 1.0e-10
    end
end
