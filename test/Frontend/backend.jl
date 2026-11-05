# Device assembly needs FiniteElementMethodKAExt, which this group never loads: a `backend` is refused at construction
# with what to load, and `backend = nothing` is the host path. Device assembly itself is tested in test/GPU.
using FiniteElementMethod, Test
using FiniteElementMethod: rectangle_grid
using Ferrite: Triangle
using ModelingToolkit: t_nounits as t

Ω = MeshDomain(rectangle_grid(Triangle, (2, 2)))

@testset "a backend without the KernelAbstractions extension raises what to load" begin
    @test Base.get_extension(FiniteElementMethod, :FiniteElementMethodKAExt) === nothing
    @test_throws "backend = :rocm needs the KernelAbstractions extension of FiniteElementMethod: load KernelAbstractions, Adapt, GPUArrays and GPUArraysCore, and the vendor package (AMDGPU for ROCBackend()), with a Ferrite that has GPU support (Ferrite.jl PR #1496)" FEMDiscretization(Ω, t; backend = :rocm)
    @test_throws "backend = :rocm needs the KernelAbstractions extension" FiniteElementMethod.device_assemble_linear!(:rocm)
end

@testset "the default backend = nothing assembles on the host" begin
    disc = FEMDiscretization(Ω, t)
    @test disc.backend === nothing && disc.precision === nothing && disc.assembly === :colored
end
