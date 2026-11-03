# PDEBase's backend extension point: `MeshDomain(grid)` calls these on the native mesh.
PDEBase.meshdim(grid::Ferrite.Grid) = _meshdim(grid)
PDEBase.facetset_names(grid::Ferrite.Grid) = _facetset_names(grid)
PDEBase.cellset_names(grid::Ferrite.Grid) = _cellset_names(grid)
PDEBase.contains_point(grid::Ferrite.Grid, x) = _contains_point(grid, x)

"""
    FEMDiscretization(Ω::MeshDomain, t = nothing; spaces = Dict(), regions = Dict(),
        quadrature = nothing, test_space = Galerkin(), ibp = Dict(), pairing = Dict(),
        codegen = :numeric, dirichlet = :algebraic, algebraic_abstol = :auto, kwargs...)

Finite-element discretization of a `PDESystem` whose spatial domain is the mesh domain `Ω`
(of a `Ferrite.Grid`), with time variable `t`. `discretize(pdesys, disc)` assembles the
Galerkin operators with Ferrite and returns a `DAEProblem` with one array unknown per field.
Without `t` the discretization is steady: the `PDESystem` must have no time variable, and
`discretize` (or `NonlinearProblem(pdesys, disc)`) returns a `NonlinearProblem` whose `solve`
gives a `PDENoTimeSolution`.

  - `spaces`: dependent variable (`u` or `u(t, x, y)`) `=>` `Ferrite.Interpolation`; the
    default is `Lagrange{RefShape, 1}()` on the mesh's reference shape. Fields may have
    different orders, such as the Taylor–Hood pair `Lagrange{RefTriangle, 2}()` for the
    velocity and `Lagrange{RefTriangle, 1}()` for the pressure. The geometry is not a choice here:
    cells are mapped with the geometric interpolation of the mesh's cell type, so a mesh of
    quadratic cells (`QuadraticTriangle`, say, from a second-order Gmsh mesh) is mapped
    isoparametrically.
  - `regions`: dependent variable `=>` `Region(Ω, name)`, a cell set of the mesh on which the
    field lives (a `Ferrite.SubDofHandler`); elsewhere it has no dofs, its equation is not
    assembled, it enters other equations as zero and the solution reads `NaN`. Its
    conditions must be on pieces whose facets belong to the region's cells.
  - `quadrature`: quadrature order shared by all fields; `nothing` means `2p` for the
    highest Lagrange order `p`.
  - `test_space`: `Galerkin()` or [`SUPG`](@ref)`(τ; b)`, streamline-upwind Petrov–Galerkin
    stabilisation of every equation; linear terms only, and `τ` cannot reference parameters.
  - `ibp`: equation (or its index) `=>` `PDEBase.StrongForm` integration-by-parts policy.
  - `pairing`: equation `=>` dependent variable it is tested with, needed for an equation
    with no time derivative of its own, such as `0 ~ Dxx(v) + f`. In a steady system the
    equations not named here take the remaining dependent variables in declared order.
  - `algebraic_abstol`: the absolute tolerance the `DAEProblem` carries for its algebraic rows
    (those with `differential_vars` false: algebraic fields and Dirichlet rows), so that they
    are left out of the solver's error control. `:auto` sets `1e8` when the system has index
    2, that is when an algebraic field's own unknowns are absent from its equation (the
    pressure in `0 ~ Dx(u1) + Dy(u2)`; a nonlinear term in the field, as in `0 ~ v^2 - u`,
    counts as present), and nothing otherwise; a positive number sets that
    value on the algebraic rows of any system; `nothing` sets nothing. The differential rows
    get the `abstol` passed to `FEMDiscretization` or `DAEProblem` if it is a number, else
    OrdinaryDiffEq's default `1e-6`; an array `abstol` passed there is used as it is. An
    `abstol` passed to `solve` replaces the problem's whole vector. The keyword belongs to the
    time-dependent (DAE) path: a steady discretization ignores `:auto` and `nothing` and
    rejects a number. This keyword is not in the spec (§7.1, §7.6): without it, `DFBDF`
    drives the step size of an index-2 system below `eps` near the initial time, since the
    local error of the index-2 unknowns does not shrink with the step.
  - `dirichlet`: how Dirichlet conditions enter the DAE (spec §7.3). The default `:algebraic`
    keeps every dof as an unknown and makes each prescribed row the algebraic equation
    `u_d = g_d(t)`. `:eliminate` removes the prescribed dofs: the unknowns are the free dofs
    `fr`, the operators are `M[fr, fr]` and `K[fr, fr]` and the load is `F[fr] - K[fr, cd] g`;
    nonlinear terms are evaluated at the full state with `u_cd = g` and keep their rows `fr`.
    It needs Dirichlet data that are constant in time (they may vary in space); data that
    depend on time raise `ArgumentError("dirichlet = :eliminate needs Dirichlet data that are
    constant in time, but the condition on boundary piece <name> depends on time; use the default
    dirichlet = :algebraic")` from `discretize`. A field whose dofs are all prescribed has no
    array unknown (its `FieldInfo.unknown` is `nothing`); if every dof of the problem is
    prescribed, `discretize` raises `ArgumentError("dirichlet = :eliminate leaves no unknowns
    because every dof is prescribed; use the default dirichlet = :algebraic")`. The solution
    still returns full fields, with `g` at the prescribed nodes, through `sol[u]`,
    `sol(t, x, y)`, traces and `write_vtk`; only `sol.original_sol` and the array unknowns
    hold the free dofs alone. `:algebraic` is recommended.
  - `codegen = :mtk` is reserved and raises; remaining keywords are
    passed to the `DAEProblem` (or `NonlinearProblem`), where a keyword given at the
    `discretize` call wins. `iv`, `unknowns`, `du0`, `coordinate`, `name` and `parameters` are set
    by the discretization and rejected.

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
assembled, and are parameters of the problem at those values. The operators are not
reassembled for new values: after `remake(prob; p = [k => 3.0])` the solve throws an
`ArgumentError` naming `k`; build a new problem with `DAEProblem(pdesys, disc)` instead.
"""
struct FEMDiscretization{T} <: PDEBase.AbstractMeshDiscretization
    domain::PDEBase.MeshDomain
    time::T
    spaces::Dict{Any, Ferrite.Interpolation}
    regions::Dict{Any, Any}
    quadrature::Union{Nothing, Int}
    test_space::AbstractTestSpace
    ibp::Dict{Any, StrongForm.IBPPolicy}
    pairing::Dict{Equation, Any}
    codegen::Symbol
    dirichlet::Symbol
    algebraic_abstol::Union{Symbol, Nothing, Float64}
    kwargs::NamedTuple
end

# Keywords of `numeric_dae_problem` that the front end sets itself.
const _RESERVED_KWARGS = (:iv, :unknowns, :du0, :coordinate, :name, :parameters)

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
        domain::PDEBase.MeshDomain, time = nothing; spaces = Dict(), regions = Dict(), quadrature = nothing,
        test_space = Galerkin(), ibp = Dict(), pairing = Dict(), codegen::Symbol = :numeric,
        dirichlet::Symbol = :algebraic, algebraic_abstol = :auto, kwargs...
    )
    domain.mesh isa Ferrite.Grid ||
        throw(ArgumentError("FEMDiscretization needs a MeshDomain of a Ferrite.Grid, got a mesh of type $(typeof(domain.mesh))"))
    codegen === :mtk && throw(ArgumentError("codegen = :mtk is not supported; use the default codegen = :numeric"))
    codegen === :numeric || throw(ArgumentError("codegen must be :numeric, got $(repr(codegen))"))
    dirichlet in (:algebraic, :eliminate) ||
        throw(ArgumentError("dirichlet must be :algebraic or :eliminate, got $(repr(dirichlet))"))
    test_space isa Union{Galerkin, SUPG} ||
        throw(ArgumentError("test_space must be Galerkin() or SUPG(τ; b = nothing), got $(repr(test_space))"))
    for (dv, region) in regions
        region isa PDEBase.Region && region.parent === domain ||
            throw(ArgumentError("the region of $(_field_op(dv)) must be a Region of the FEMDiscretization's MeshDomain, as Region(Ω, name); got $(repr(region))"))
    end
    algebraic_abstol === :auto || algebraic_abstol === nothing || algebraic_abstol isa Real && algebraic_abstol > 0 ||
        throw(ArgumentError("algebraic_abstol must be :auto, nothing or a positive number, got $(repr(algebraic_abstol))"))
    time === nothing && algebraic_abstol isa Real && throw(
        ArgumentError(
            "algebraic_abstol applies to time-dependent problems (the DAE path); a steady FEMDiscretization has no time stepper"
        )
    )
    _check_reserved(kwargs)
    return FEMDiscretization{typeof(time)}(
        domain, time, Dict{Any, Ferrite.Interpolation}(_field_op(k) => ip for (k, ip) in spaces),
        Dict{Any, Any}(_field_op(k) => r for (k, r) in regions),
        quadrature, test_space, Dict{Any, StrongForm.IBPPolicy}(ibp), Dict{Equation, Any}(pairing),
        codegen, dirichlet, algebraic_abstol isa Real ? Float64(algebraic_abstol) : algebraic_abstol, NamedTuple(kwargs)
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
# it on the Discretization page and docs/src/boundary_conditions.md keeps a static copy;
# test/Frontend/documentation.jl keeps the two equal.
PDEBase.supported_conditions(::FEMDiscretization) = PDEBase.SupportedConditions(;
    geometries = (:mesh,), dirichlet = true, neumann = true, robin = true, periodic = false,
    interface = false, time_dependent = true,
    notes = """
    Restrictions of this release:

      - Geometries are `Ferrite.Grid` meshes (1D, 2D and 3D cell shapes) wrapped as `MeshDomain(grid)`, with boundary pieces named by the grid's facet sets. Only 1D and 2D are evidenced; meshes with mixed cell types are rejected until M4.
      - Coordinates are Cartesian and named by the `(x, y) ∈ Ω` pairing. Time is the first or last argument of every dependent variable, never between spatial arguments.
      - Dirichlet data may depend on space, time and parameters but not on another field. Neumann and Robin conditions are written with `NormalDerivative(Γ)`, with any conormal coefficient written explicitly. Neumann and Robin data may depend on time; the Robin coefficient multiplying `u(t, Γ)` may depend only on space and parameters until M3.
      - Periodic conditions `u(t, Γa) ~ u(t, Γb)` are rejected until M4. Interface conditions between regions have no syntax; every field lives on the whole mesh.
      - Boundary pieces must be disjoint. A piece with no condition gets the natural zero-flux condition, with a warning when the operator has a flux.
      - Where two Dirichlet pieces of one field meet, the condition listed last in the `PDESystem` wins.
      - Every field with a time derivative needs an initial condition, linear in that one field and finite at every node.
      - Time-dependent mass or stiffness coefficients and `codegen = :mtk` are rejected until M3.
    """
)
