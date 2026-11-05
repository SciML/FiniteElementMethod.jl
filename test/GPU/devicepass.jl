# The device pass on the host: the kernels of real PDESystems lose their Float64 literals and NaNMath.pow, compile to
# isbits functions, and in Float32 compute in Float32 to within 1e-6 of the Float64 kernels. Unsupported calls are
# refused when the kernel is prepared, and user functions are named when a kernel fails to compile.
using NaNMath: NaNMath
using RuntimeGeneratedFunctions: drop_expr
include(joinpath(@__DIR__, "common", "backend.jl"))

const Ext = Base.get_extension(FEM, :FiniteElementMethodKAExt)
const ROWS = all_rows()
kernel(body) = Expr(:function, :((x, T)), Expr(:block, body))
eltypes(r::Tuple) = Set(typeof.(r))
eltypes(r) = Set([typeof(r)])

@testset "the extension loads with KernelAbstractions, Adapt, GPUArrays and GPUArraysCore" begin
    @test Ext isa Module
end

@testset "the raw kernels hold Float64 literals and NaNMath.pow, and promote Float32 arguments to Float64" begin
    before = [Ext.device_violations(ex, Float32) for (_, _, ex) in ROWS]
    @test any(v -> any(startswith("the Float64 literal"), v), before)
    @test any(v -> any(startswith("NaNMath.pow"), v), before)
    promoted = map(ROWS) do (_, _, _, g, sig, np)
        Float64 in eltypes(callk(drop_expr(g), Val(sig), vec2(Float32, 0.4, 0.6), 0.8f0, fixed_args(sig, Float32, np)))
    end
    @test count(promoted) > length(ROWS) ÷ 2
end

@testset "after the pass $name/$role passes the device check, is isbits and computes in Float32" for (name, role, ex, g, sig, np) in ROWS
    @test isempty(Ext.device_violations(Ext.retype(ex, Float32), Float32))
    @test isempty(Ext.device_violations(Ext.retype(ex, Float64), Float64))
    dk = FEM.device_kernel(ex, Float32)
    @test isbits(dk)
    @test Float64 ∉ eltypes(callk(dk, Val(sig), vec2(Float32, 0.4, 0.6), 0.8f0, fixed_args(sig, Float32, np)))
    pts, us = host_points()
    @test relerr(Float64.(reference(dk, sig, np, pts, us, Float32)), reference(g, sig, np, pts, us)) < rtol(Float32)
end

@testset "prepared coefficient kernels in a LinearBlock are isbits; a KernelSpec is not" begin
    rows = Dict(role => ex for (name, role, ex) in ROWS if name == "heat")
    coeffs(kind) = FEM.device_kernel(rows["$kind.coeffs"], Float32)
    @test isbits(FEM.LinearBlock(1, coeffs(:mass), coeffs(:bilinear), coeffs(:source)))
    @test !isbits(FEM.KernelSpec(1, :source, nothing, nothing, nothing, coeffs(:source)))
end

@testset "NaNMath.pow becomes nanpow: NaN for a negative base, where ^ throws a DomainError on the host" begin
    @test_throws DomainError (-1.0f0)^1.5f0
    @test Ext.nanpow(-1.0f0, 1.5f0) isa Float32 && isnan(Ext.nanpow(-1.0f0, 1.5f0))
    for (b, e) in ((-1.0, 1.5), (-2.0, 2.0), (-2.0, 3), (2, 0.5), (3.0f0, 2), (2.0f0, 1.5f0))
        @test isequal(Ext.nanpow(b, e), NaNMath.pow(b, e))
    end
    # The exponent here is the kernel's argument named T, which the pass must leave alone.
    for pow in (NaNMath.pow, GlobalRef(NaNMath, :pow))
        dk = FEM.device_kernel(kernel(Expr(:call, pow, :x, :T)), Float32)
        @test dk(3.0f0, 2.0f0) === 9.0f0
        @test isnan(FEM.device_kernel(kernel(Expr(:call, pow, :x, 1.5)), Float32)(-1.0f0, 0.0f0))
    end
    @test FEM.device_kernel(kernel(:(x * -1 // 2 + 0.25)), Float32)(1.0f0, 0.0f0) === -0.25f0
end

@testset "library ccalls, closures and ccalls are refused when the kernel is prepared" begin
    msg = "a generated kernel cannot run on a device: it contains NaNMath.lgamma, a library ccall with no device implementation"
    @test_throws msg FEM.device_kernel(kernel(Expr(:call, NaNMath.lgamma, :x)), Float32)
    @test_throws msg FEM.device_kernel(kernel(Expr(:call, GlobalRef(NaNMath, :lgamma), :x)), Float64)
    @test_throws "it contains a closure" FEM.device_kernel(kernel(:(map(z -> z + 1, x))), Float32)
    @test_throws "it contains a ccall" FEM.device_kernel(kernel(:(ccall(:sin, Float64, (Float64,), x))), Float32)
    @test_throws "must be a floating-point type, got Int64" FEM.device_kernel(kernel(:x), Int)
end

@testset "user-registered functions are listed, and named when a kernel fails to compile" begin
    callees = unique(reduce(vcat, [Ext.foreign_callees(ex) for (_, ex) in first(kernel_rows(registered(badf), Galerkin()))]))
    @test length(callees) == 1 && endswith(only(callees), ".badf")
    # FEM's own calls (Base, NaNMath, StructuralZero) are never listed.
    listed = unique(reduce(vcat, [Ext.foreign_callees(ex) for (_, _, ex) in ROWS]))
    @test length(listed) == 1 && endswith(only(listed), ".goodf")
    err = ErrorException("InvalidIRError: unsupported call to an unknown function")
    cpu = KernelAbstractions.CPU()
    msg = "a generated kernel failed to compile for CPU; it calls the user functions $(only(callees)), which must be " *
        "GPU-compilable (no allocation, Dict, String or ccall). Original error: InvalidIRError: unsupported call"
    @test_throws msg Ext.launch_checked(() -> throw(err), cpu, callees)
    @test_throws err Ext.launch_checked(() -> throw(err), cpu, String[])
    @test Ext.launch_checked(() -> 1, cpu, callees) == 1
end
