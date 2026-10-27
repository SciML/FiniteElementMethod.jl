# PDEBase's backend extension point: `MeshDomain(grid)` calls these on the native mesh.
PDEBase.meshdim(grid::Ferrite.Grid) = _meshdim(grid)
PDEBase.facetset_names(grid::Ferrite.Grid) = _facetset_names(grid)
PDEBase.cellset_names(grid::Ferrite.Grid) = _cellset_names(grid)
PDEBase.contains_point(grid::Ferrite.Grid, x) = _contains_point(grid, x)

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
  - `dirichlet`: how Dirichlet conditions enter the DAE (spec §7.3). The default `:algebraic`
    keeps every dof as an unknown and makes each prescribed row the algebraic equation
    `u_d = g_d(t)`. `:eliminate` removes the prescribed dofs: the unknowns are the free dofs
    `fr`, the operators are `M[fr, fr]` and `K[fr, fr]` and the load is `F[fr] - K[fr, cd] g`.
    It needs Dirichlet data that are constant in time (they may vary in space); data that
    depend on time raise `ArgumentError("dirichlet = :eliminate needs Dirichlet data that are
    constant in time, but the condition on boundary piece <name> depends on time; use the default
    dirichlet = :algebraic")` from `DAEProblem`. The solution still returns full fields, with
    `g` at the prescribed nodes, through `sol[u]`, `sol(t, x, y)`, traces and `write_vtk`;
    only `sol.original_sol` and the array unknowns hold the free dofs alone. `:algebraic` is
    recommended.
  - `codegen = :mtk` is reserved for a later milestone and raises; remaining keywords are
    passed to the `DAEProblem`, where a keyword given at the `DAEProblem`/`discretize` call
    wins. `iv`, `unknowns`, `du0`, `coordinate` and `name` are set by the discretization and
    rejected.

Pieces come from `boundarypiece.(Ω, names)`; on an analytic DomainSets box write
`boundarypiece.(Ref(rect), names)`, since DomainSets types do not broadcast as scalars.

Where pieces with Dirichlet conditions on the same field meet (a corner), the condition listed
last in the `PDESystem`'s boundary conditions wins, as in Ferrite's `ConstraintHandler`. Listing
the same piece and field twice is an error.
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
    dirichlet in (:algebraic, :eliminate) ||
        throw(ArgumentError("dirichlet must be :algebraic or :eliminate, got $(repr(dirichlet))"))
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
