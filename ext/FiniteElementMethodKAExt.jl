module FiniteElementMethodKAExt

using FiniteElementMethod: FiniteElementMethod
const FEM = FiniteElementMethod
using Adapt: adapt
using Ferrite: Ferrite
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, @ndrange
const KA = KernelAbstractions
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, RuntimeGeneratedFunction, drop_expr
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, rowvals

# NaNMath is not a dependency, so its functions are recognised by module and function name.
callee(f::Function) = (parentmodule(f), nameof(f))
callee(g::GlobalRef) = (g.mod, g.name)
callee(_) = nothing
function callee_is(c, mod, name)
    m = callee(c)
    return m !== nothing && nameof(m[1]) === mod && m[2] === name
end

# Library ccalls with no device implementation (NaNMath 1.1.5, SpecialFunctions 2): GPUCompiler rejects them.
const NO_DEVICE_METHOD = Set(
    [
        (:NaNMath, :lgamma);
        [(:SpecialFunctions, f) for f in (:erfc, :erfcx, :erfi, :dawson, :besselj0, :besselj1, :bessely0, :bessely1)]
    ]
)

# `NaNMath.pow` as NaNMath 1.1.5 implements it in Julia; earlier NaNMath releases call libm, which no device has. It is
# NaN for a negative base with a non-integer exponent, where `^` throws a DomainError on the host.
function nanpow(x::Real, y::Real)
    fx, fy = map(float, promote(x, y))
    return fx >= 0 || isinteger(y) ? fx^fy : oftype(fx, NaN)
end
nanpow(x::Real, y::Integer) = x^y
nanpow(x, y) = x^y

# The kernel's own argument named `T` (the time slot) is never touched: only literal values are replaced.
retype(ex, ::Type) = ex
retype(x::Union{Float64, Rational}, ::Type{T}) where {T} = T(x)
function retype(ex::Expr, ::Type{T}) where {T}
    args = Any[retype(a, T) for a in ex.args]
    ex.head === :call && callee_is(args[1], :NaNMath, :pow) && return Expr(:call, nanpow, args[2:end]...)
    return Expr(ex.head, args...)
end

function device_violations(kernel::Expr, ::Type{T}) where {T}
    found = String[]
    function visit(e)
        e isa Float64 && T !== Float64 && push!(found, "the Float64 literal $e")
        e isa Expr || return
        e.head in (:->, :function) && push!(found, "a closure")
        (e.head === :foreigncall || (e.head === :call && e.args[1] in (:ccall, :cglobal))) && push!(found, "a ccall")
        if e.head === :call
            c = callee(e.args[1])
            callee_is(e.args[1], :NaNMath, :pow) && push!(found, "NaNMath.pow, a libm ccall before NaNMath 1.1.5")
            c !== nothing && (nameof(c[1]), c[2]) in NO_DEVICE_METHOD &&
                push!(found, "$(c[1]).$(c[2]), a library ccall with no device implementation")
        end
        return foreach(visit, e.args)
    end
    # The kernel is itself a function expression; only functions inside its body are closures.
    foreach(visit, kernel.head in (:->, :function) ? kernel.args : (kernel,))
    return unique(found)
end

function FiniteElementMethod.device_kernel(ex::Expr, ::Type{T}) where {T}
    T <: AbstractFloat || throw(ArgumentError("the precision of a device kernel must be a floating-point type, got $T"))
    kernel = retype(ex, T)
    found = device_violations(kernel, T)
    isempty(found) || throw(ArgumentError("a generated kernel cannot run on a device: it contains " * join(found, ", ")))
    # FiniteElementMethod's own cache and context, as Symbolics.build_function uses for the host kernels.
    f = drop_expr(RuntimeGeneratedFunction(FiniteElementMethod, FiniteElementMethod, kernel))
    @assert isbits(f)
    return f
end

# Calls into modules other than these are user-registered symbolic functions, which may not compile for a device;
# that is known only when the first launch compiles the kernel.
const DEVICE_SAFE_MODULES = (Base, Core, FiniteElementMethod, @__MODULE__)
function foreign_callees(ex)
    found = Set{String}()
    function visit(e)
        e isa Expr || return
        if e.head === :call
            c = callee(e.args[1])
            c === nothing || Base.moduleroot(c[1]) in DEVICE_SAFE_MODULES || nameof(c[1]) === :NaNMath ||
                push!(found, "$(c[1]).$(c[2])")
        end
        return foreach(visit, e.args)
    end
    visit(ex)
    return sort!(collect(found))
end

function kernel_compile_error(backend::KernelAbstractions.Backend, callees, err)
    return ArgumentError(
        "a generated kernel failed to compile for $(nameof(typeof(backend))); it calls the user functions " *
            join(callees, ", ") * ", which must be GPU-compilable (no allocation, Dict, String or ccall). " *
            "Original error: " * first(split(sprint(showerror, err), '\n'))
    )
end

function launch_checked(launch, backend::KernelAbstractions.Backend, callees)
    try
        return launch()
    catch err
        isempty(callees) && rethrow()
        throw(kernel_compile_error(backend, callees, err))
    end
end

include("FiniteElementMethodKAExt/assembly.jl")

end # module
