# Loading and refusals of device assembly: the FEM and Ferrite extensions it needs, the keywords' validation, the
# error when a Ferrite without GPU support is loaded, and the features device assembly does not cover yet.
using JLArrays: JLBackend
using GenericSparseArrays: GenericSparseArrays
include(joinpath(@__DIR__, "common", "assembly.jl"))

const Ext = Base.get_extension(FEM, :FiniteElementMethodKAExt)

@testset "the FEM and Ferrite KernelAbstractions extensions are both loaded" begin
    @test Ext isa Module
    @test Base.get_extension(FEM, :FiniteElementMethodGenericSparseArraysExt) isa Module
    @test Base.get_extension(Ferrite, :FerriteKAExt) isa Module
    @test FEM.is_device_backend(KernelAbstractions.CPU()) && FEM.is_device_backend(JLBackend())
    @test FEM.device_matrix_type(JLBackend(), Float32, Int) <: GenericSparseArrays.GenericSparseMatrixCSC
end

@testset "the error when Ferrite has no GPU support names what is missing" begin
    @test_throws "device assembly needs a Ferrite with GPU support (the FerriteKAExt extension, Ferrite.jl PR #1496)" Ext.check_ferrite_gpu(nothing)
end

@testset "the keywords are validated at construction" begin
    pdesys, Ω, spaces, _ = heat2d_p2(CONSTANT_G)
    @test_throws "backend must be a KernelAbstractions.Backend, such as ROCBackend() or KernelAbstractions.CPU(), or nothing; got :rocm" FEMDiscretization(Ω, t; backend = :rocm)
    @test_throws "precision must be Float32, Float64 or nothing, got Float16" FEMDiscretization(Ω, t; backend = JLBackend(), precision = Float16)
    @test_throws "precision = Float32 sets the precision of device assembly, so it needs a backend" FEMDiscretization(Ω, t; precision = Float32)
    @test_throws "assembly must be :colored or :atomic, got :serial" FEMDiscretization(Ω, t; backend = JLBackend(), assembly = :serial)
    # neither is forwarded to the problem
    prob = discretize(pdesys, FEMDiscretization(Ω, t; spaces, backend = JLBackend(), precision = Float32))
    @test !any(in((:backend, :precision, :assembly)), keys(prob.kwargs))
end

@testset "features device assembly does not cover raise an ArgumentError naming them" begin
    Ω = MeshDomain(generate_grid(Triangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right = boundarypiece(Ω, :left), boundarypiece(Ω, :right)
    U = u(t, x, y)
    sys(eq, bcs) = PDESystem([eq], [bcs; u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], []; name = :p)
    diffusion = Dx(Dx(U)) + Dy(Dy(U))
    dev(; kw...) = FEMDiscretization(Ω, t; backend = JLBackend(), kw...)
    msg(what) = "backend = $(JLBackend()): device assembly does not support $what yet; remove the backend keyword to assemble on the host"
    @test_throws msg("facet terms (Neumann and Robin conditions)") discretize(sys(Dt(U) ~ diffusion, [u(t, left) ~ 0, FEM.NormalDerivative(right)(u(t, right)) ~ 1]), dev())
    @test_throws msg("nonlinear terms") discretize(sys(Dt(U) ~ diffusion - U^3, [u(t, left) ~ 0]), dev())
    @test_throws msg("sources that depend on time") discretize(sys(Dt(U) ~ diffusion + sin(t), [u(t, left) ~ 0]), dev())
    @test_throws msg("dirichlet = :eliminate (use the default dirichlet = :algebraic)") discretize(sys(Dt(U) ~ diffusion, [u(t, left) ~ 0]), dev(dirichlet = :eliminate))
    @test_throws msg("the SUPG test space") discretize(
        sys(Dt(U) ~ 0.01 * diffusion - Dx(U), [u(t, left) ~ 0, u(t, right) ~ 1]),
        dev(test_space = FEM.SUPG((hK, b) -> hK / 2))
    )
end
