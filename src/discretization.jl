# PDEBase's backend extension point: `MeshDomain(grid)` calls these on the native mesh.
PDEBase.meshdim(grid::Ferrite.Grid) = _meshdim(grid)
PDEBase.facetset_names(grid::Ferrite.Grid) = _facetset_names(grid)
PDEBase.cellset_names(grid::Ferrite.Grid) = _cellset_names(grid)
PDEBase.contains_point(grid::Ferrite.Grid, x) = _contains_point(grid, x)

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
    raise; remaining keywords are passed to the `DAEProblem`, where a keyword given at the
    `DAEProblem`/`discretize` call wins. `iv`, `unknowns`, `du0`, `coordinate` and `name`
    are set by the discretization and rejected.

Pieces come from `boundarypiece.(Ω, names)`; on an analytic DomainSets box write
`boundarypiece.(Ref(rect), names)`, since DomainSets types do not broadcast as scalars.

Where pieces with Dirichlet conditions on the same field meet (a corner), the condition listed
last in the `PDESystem`'s boundary conditions wins, as in Ferrite's `ConstraintHandler`.
Fields are numbered in one order throughout (`DofHandler` blocks, array unknowns,
`FEMMetadata.fields`), the declared order of the `PDESystem`'s dependent variables
(`PDEBase.StrongForm.declared_fields`); equations are paired with fields
through `PDEBase.StrongForm.pair_equations`.

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

# Keywords of `numeric_dae_problem` that the front end sets itself.
const _RESERVED_KWARGS = (:iv, :unknowns, :du0, :coordinate, :name)

function _check_reserved(kwargs)
    for k in keys(kwargs)
        k in _RESERVED_KWARGS && throw(
            ArgumentError(
                "keyword `$k` is set by the FEM discretization and cannot be passed; the reserved keywords are " *
                    join(_RESERVED_KWARGS, ", ")
            )
        )
    end
    return nothing
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
    _check_reserved(kwargs)
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

function Base.show(io::IO, disc::FEMDiscretization)
    return print(
        io, "FEMDiscretization(", Ferrite.getncells(disc.domain.mesh), " cells, time = ", disc.time,
        ", codegen = :", disc.codegen, ")"
    )
end

# The SciMLBase PDE-interface checklist of this discretization (PDEBase#147). docs/make.jl renders
# it on the Discretization page and docs/src/boundary_conditions.md keeps a static copy for builds
# without the front end; test/Frontend/supported_conditions.jl keeps the two equal.
# Tracking: drop the `isdefined` guard once the PDEBase compat floor has `supported_conditions`.
@static if isdefined(PDEBase, :supported_conditions)
    PDEBase.supported_conditions(::FEMDiscretization) = PDEBase.SupportedConditions(;
        geometries = (:mesh,), dirichlet = true, neumann = true, robin = true, periodic = false,
        interface = false, time_dependent = true,
        notes = """
        Restrictions of this release:

          - Geometries are `Ferrite.Grid` meshes (1D, 2D and 3D cell shapes) wrapped as `MeshDomain(grid)`, with boundary pieces named by the grid's facet sets. Only 1D and 2D are evidenced; meshes with mixed cell types are rejected until M4.
          - Coordinates are Cartesian and named by the `(x, y) ∈ Ω` pairing. Time must be the first argument of every dependent variable.
          - Dirichlet data may depend on space, time and parameters but not on another field. Neumann and Robin conditions are written with `NormalDerivative(Γ)`, with any conormal coefficient written explicitly. Neumann and Robin data may depend on time; the Robin coefficient multiplying `u(t, Γ)` may depend only on space and parameters until M3.
          - Periodic conditions `u(t, Γa) ~ u(t, Γb)` are rejected until M4. Interface conditions between regions have no syntax; every field lives on the whole mesh.
          - Boundary pieces must be disjoint. A piece with no condition gets the natural zero-flux condition, with a warning when the operator has a flux.
          - Where two Dirichlet pieces of one field meet, the condition listed last in the `PDESystem` wins.
          - Every field with a time derivative needs an initial condition, linear in that one field and finite at every node.
          - Steady problems, `dirichlet = :eliminate`, fields with different interpolations and test spaces other than `Galerkin()` are rejected until M4. Nonlinear operators, time-dependent mass or stiffness coefficients and `codegen = :mtk` are rejected until M3.
        """
    )
end
