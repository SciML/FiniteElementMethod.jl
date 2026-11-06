# The end-to-end check of this layer on one backend: real PDESystem, emitted kernels, the device pass, a
# KernelAbstractions kernel, and its result against the host Float64 kernel.
using Test
# The four triggers of FiniteElementMethodKAExt.
using KernelAbstractions: KernelAbstractions
using Adapt: Adapt
using GPUArrays: GPUArrays
using GPUArraysCore: GPUArraysCore

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "harness.jl"))

# The largest relative error per precision over all kernels.
function test_backend(backend, toarr; rows = all_rows())
    pts, us = host_points()
    worst = Dict(Float32 => 0.0, Float64 => 0.0)
    @testset "$name/$role on $(nameof(typeof(backend)))" for (name, role, ex, g, sig, np) in rows
        ref = reference(g, sig, np, pts, us)
        for T in (Float32, Float64)
            out = run_on(backend, toarr, FEM.device_kernel(ex, T), sig, np, pts, us, size(ref, 1), T)
            worst[T] = max(worst[T], relerr(Float64.(out), ref))
            @test relerr(Float64.(out), ref) < rtol(T)
        end
    end
    return worst
end
