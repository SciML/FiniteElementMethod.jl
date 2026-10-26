"""
    AbstractTestSpace

Supertype of the test-function choices of a [`FEMDiscretization`](@ref).
"""
abstract type AbstractTestSpace end

"""
    Galerkin()

Bubnov–Galerkin testing: every equation is tested with the space of its paired field.
The default, and the only test space before SUPG lands.
"""
struct Galerkin <: AbstractTestSpace end

"""
    FEMDiscretization(Ω::MeshDomain, t = nothing; spaces = Dict(), quadrature = nothing,
        test_space = Galerkin(), ibp = Dict(), pairing = Dict(), codegen = :numeric,
        dirichlet = :algebraic, kwargs...)

Finite-element discretization of a `PDESystem` whose spatial domain is the mesh domain `Ω`
(of a `Ferrite.Grid`), with time variable `t`. `DAEProblem(pdesys, disc)` assembles the
Galerkin operators with Ferrite and returns a `DAEProblem` with one array unknown per field.

  - `spaces`: dependent variable (`u` or `u(t, x, y)`) `=>` `Ferrite.Interpolation`; the
    default is `Lagrange{RefShape, 1}()` on the mesh's reference shape. All fields must
    currently share one interpolation.
  - `quadrature`: quadrature order; `nothing` means `2p` for Lagrange order `p`.
  - `ibp`: equation (or its index) `=>` `PDEBase.StrongForm` integration-by-parts policy.
  - `pairing`: equation `=>` dependent variable it is tested with, needed for an equation
    with no time derivative of its own, such as `0 ~ Dxx(v) + f`.
  - `codegen = :mtk` and `dirichlet = :eliminate` are reserved for later milestones and
    raise; remaining keywords are passed to the `DAEProblem`.

Parameters are substituted by the values given in the `PDESystem` when the operators are
assembled, so `remake` with new parameter values does not reassemble them.
"""
struct FEMDiscretization{T} <: PDEBase.AbstractMeshDiscretization
    domain::PDEBase.MeshDomain
    time::T
    spaces::Dict{Any, Ferrite.Interpolation}
    quadrature::Union{Nothing, Int}
    test_space::AbstractTestSpace
    ibp::Dict{Any, StrongForm.IBPPolicy}
    pairing::Dict{Equation, Any}
    codegen::Symbol
    dirichlet::Symbol
    kwargs::NamedTuple
end

function FEMDiscretization(
        domain::PDEBase.MeshDomain, time = nothing; spaces = Dict(), quadrature = nothing,
        test_space = Galerkin(), ibp = Dict(), pairing = Dict(), codegen::Symbol = :numeric,
        dirichlet::Symbol = :algebraic, kwargs...
    )
    domain.mesh isa Ferrite.Grid ||
        throw(ArgumentError("FEMDiscretization needs a MeshDomain of a Ferrite.Grid, got a mesh of type $(typeof(domain.mesh))"))
    codegen === :mtk && throw(ArgumentError("codegen = :mtk lands in M3; use the default codegen = :numeric"))
    codegen === :numeric || throw(ArgumentError("codegen must be :numeric, got $(repr(codegen))"))
    dirichlet === :eliminate &&
        throw(ArgumentError("dirichlet = :eliminate lands in M4; use the default dirichlet = :algebraic"))
    dirichlet === :algebraic || throw(ArgumentError("dirichlet must be :algebraic, got $(repr(dirichlet))"))
    test_space isa Galerkin ||
        throw(ArgumentError("test space $(test_space) lands in M4; only Galerkin() is supported"))
    return FEMDiscretization{typeof(time)}(
        domain, time, Dict{Any, Ferrite.Interpolation}(_field_op(k) => ip for (k, ip) in spaces),
        quadrature, test_space, Dict{Any, StrongForm.IBPPolicy}(ibp), Dict{Equation, Any}(pairing),
        codegen, dirichlet, NamedTuple(kwargs)
    )
end

# `u`, `u(..)` and `u(t, x, y)` all name the dependent variable `u`.
function _field_op(dv)
    s = unwrap(dv)
    return SymbolicUtils.iscall(s) ? SymbolicUtils.operation(s) : s
end

PDEBase.get_time(disc::FEMDiscretization) = disc.time
