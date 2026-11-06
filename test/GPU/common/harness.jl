# A KernelAbstractions kernel that evaluates one generated kernel per point with driver-shaped arguments, and the
# comparison of a backend's result with the host Float64 kernel.
using FiniteElementMethod: StructuralZero
using Ferrite: Vec
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using Random: Random

const NPTS = 256

vec2(::Type{T}, a, b) where {T} = Vec{2, T}((T(a), T(b)))

# The arguments of a signature other than X and u, in precision T, with the `np` parameters as an NTuple.
function fixed_args(sig, ::Type{T}, np) where {T}
    P = ntuple(i -> T(0.5 + 0.1i), np)
    t, N = T(0.3), vec2(T, 0.6, 0.8)
    sig === :coeffs && return (t, N, P)
    sig === :cell && return (t, N, P, T(0.05))
    sig === :dirichlet && return (t, P)
    return ((T(0.2), T(-0.4)), (T(0.7),), T(0.9), vec2(T, 0.3, -0.5), t, N, P)   # G, Ud, V, W, T, N, P
end
callk(f, ::Val{:full}, X, u, (G, Ud, V, W, t, N, P)) = f((u,), G, Ud, V, W, X, t, N, P)
callk(f, ::Val, X, u, fixed) = f(X, fixed...)

num(::StructuralZero, ::Type{T}) where {T} = zero(T)
num(x, ::Type{T}) where {T} = T(x)
@inline function store!(out, i, r::Tuple)
    vals = map(v -> num(v, eltype(out)), r)
    for j in 1:length(vals)
        @inbounds out[j, i] = vals[j]
    end
    return nothing
end
@inline store!(out, i, r) = (@inbounds out[1, i] = num(r, eltype(out)); nothing)

@kernel function eval_kernel!(out, f, @Const(pts), @Const(us), fixed, sig)
    i = @index(Global)
    X = Vec{2, eltype(out)}((pts[1, i], pts[2, i]))
    store!(out, i, callk(f, sig, X, us[i], fixed))
end

nout(r::Tuple) = length(r)
nout(r) = 1

# Points in (0.1, 1.1)^2, so fractional powers, log and sqrt of the coordinates stay real; u in (0.5, 1.5).
host_points() = (Random.seed!(1); (0.1 .+ rand(2, NPTS), 0.5 .+ rand(NPTS)))

# The kernel `g` evaluated pointwise on the host in precision T.
function reference(g, sig, np, pts, us, ::Type{T} = Float64) where {T}
    fixed = fixed_args(sig, T, np)
    at(i) = callk(g, Val(sig), vec2(T, pts[1, i], pts[2, i]), T(us[i]), fixed)
    out = zeros(T, nout(at(1)), NPTS)
    foreach(i -> store!(out, i, at(i)), 1:NPTS)
    return out
end

# JLArrays 0.3.3 (the last release for KernelAbstractions 0.9) defines no `synchronize(::JLBackend)`; its launches are
# synchronous.
sync(b) = hasmethod(KernelAbstractions.synchronize, Tuple{typeof(b)}) ? KernelAbstractions.synchronize(b) : nothing

# The device kernel `dk` evaluated on `backend`, with arrays moved there by `toarr`; the output copied to the host.
function run_on(backend, toarr, dk, sig, np, pts, us, nrows, ::Type{T}) where {T}
    out = KernelAbstractions.zeros(backend, T, nrows, NPTS)
    eval_kernel!(backend, 64)(out, dk, toarr(T.(pts)), toarr(T.(us)), fixed_args(sig, T, np), Val(sig); ndrange = NPTS)
    sync(backend)
    return Array(out)
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

# Tolerances against the host Float64 kernel: Float32 kernels measured within 5.5e-7 on Radeon 8060S.
rtol(::Type{Float32}) = 1.0e-6
rtol(::Type{Float64}) = 1.0e-13
