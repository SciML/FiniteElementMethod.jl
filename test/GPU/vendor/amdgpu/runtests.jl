# Run by hand on an AMD GPU, never in CI, and never alongside another GPU run:
#     julia --project=test/GPU/vendor/amdgpu test/GPU/vendor/amdgpu/runtests.jl
# LocalPreferences.toml caps AMDGPU's pool at 6 GiB (soft) and 8 GiB (hard): the device memory is system RAM here.
using AMDGPU
include(joinpath(@__DIR__, "..", "..", "common", "backend.jl"))

const Ext = Base.get_extension(FEM, :FiniteElementMethodKAExt)
AMDGPU.versioninfo()
println("KernelAbstractions ", pkgversion(KernelAbstractions), ", AMDGPU ", pkgversion(AMDGPU), ", device ", AMDGPU.device())

@testset "AMDGPU" begin
    @test AMDGPU.functional()
    backend = ROCBackend()
    worst = test_backend(backend, ROCArray)
    println("largest relative error against the host Float64 kernels: ", worst)

    @testset "a Dict-allocating registered function fails to compile, with an error naming it" begin
        rows, np = kernel_rows(registered(badf), Galerkin())
        pts, us = host_points()
        checked = 0
        for (role, ex, g, sig) in rows
            callees = Ext.foreign_callees(ex)
            isempty(callees) && continue
            dk = FEM.device_kernel(ex, Float32)
            nrows = size(reference(g, sig, np, pts, us), 1)
            launch() = run_on(backend, ROCArray, dk, sig, np, pts, us, nrows, Float32)
            @test_throws "a generated kernel failed to compile for ROCBackend; it calls the user functions $(only(callees))" Ext.launch_checked(launch, backend, callees)
            try
                Ext.launch_checked(launch, backend, callees)
            catch err
                println(role, ": ", sprint(showerror, err))
            end
            checked += 1
        end
        @test checked > 0
    end
end

# Device assembly through `discretize` on ROCBackend(), in its own module: `common/assembly.jl` names its own symbolic
# variables. Float64 runs at about 1/53 of the Float32 rate on Radeon 8060S, so the cases are small.
module DeviceAssemblyAMDGPU
    using AMDGPU: AMDGPU, ROCBackend
    include(joinpath(@__DIR__, "..", "..", "common", "assembly.jl"))
    @testset "device assembly on AMDGPU" begin
        @test Base.get_extension(FEM, :FiniteElementMethodAMDGPUExt) isa Module
        @test Base.get_extension(Ferrite, :FerriteAMDGPUExt) isa Module
        @test FEM.device_matrix_type(ROCBackend(), Float32, Int) <: AMDGPU.rocSPARSE.ROCSparseMatrixCSC
        println("largest relative errors on ROCBackend(): ", test_device_assembly(ROCBackend()))
    end
end
