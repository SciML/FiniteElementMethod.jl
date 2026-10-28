"""
    FiniteElementMethod

A SciML discretizer that takes a ModelingToolkit `PDESystem` in strong form on a mesh,
derives the Galerkin weak form, assembles it with Ferrite and lowers it to a `DAEProblem`.

The entry point is `FEMDiscretization`; `DAEProblem(pdesys, disc)` lowers a `PDESystem`.
"""
module FiniteElementMethod

using ADTypes: AutoForwardDiff
using DiffEqBase: BrownFullBasicInit
using DomainSets: DomainSets
using Ferrite: Ferrite, Lagrange, RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron
using ForwardDiff: ForwardDiff
using LinearAlgebra: dot, mul!
using ModelingToolkit: ModelingToolkit, @parameters, @variables, Differential, MTKParameters,
    ProblemTypeCtx, System, complete, getmetadata
using NonlinearSolveFirstOrder: NewtonRaphson
using PDEBase: PDEBase
using SciMLBase: SciMLBase, DAEFunction, DAEProblem
using SciMLBase.SymbolicIndexingInterface: getp, variable_index
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics, Equation, Num, @register_array_symbolic, substitute, unwrap

include("weakform.jl")
include("kernels.jl")
include("assembly.jl")
include("operators.jl")
include("numeric_dae.jl")

include("mesh.jl")
include("space.jl")
include("solution.jl")
export FieldInfo, FEMMetadata, piece_nodes, normal_flux, write_vtk
# Re-exported so that the user API of a FEM model needs no `using PDEBase, Ferrite`.
export Lagrange, RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron

# A mesh that exists only to ask PDEBase how it numbers a MeshDomain's coordinates.
struct _AxisOrderProbe end
@static if isdefined(PDEBase, :meshdim)
    PDEBase.meshdim(::_AxisOrderProbe) = 2
end

# Whether PDEBase numbers the coordinates of `(y, x) ∈ MeshDomain(mesh)` by tuple position
# (PDEBase#158) rather than by the dependent variables' argument order; the latter silently binds
# `x` of `u(x, y)` to mesh coordinate 1. No name marks the change, so this asks VariableMap.
function _orders_mesh_axes_by_pairing()
    isdefined(PDEBase, :meshdim) && isdefined(PDEBase, :MeshDomain) || return false
    @variables x y u(..)
    Ω = PDEBase.MeshDomain(_AxisOrderProbe())
    sys = ModelingToolkit.PDESystem([u(x, y) ~ 0], Equation[], [(y, x) ∈ Ω], [x, y], [u(x, y)]; name = :probe)
    return isequal(collect(PDEBase.VariableMap(sys).x̄), [unwrap(y), unwrap(x)])
end

# Whether the installed PDEBase has the mesh-discretization layer the PDESystem front end is
# built on (spec §4), including PDEBase#158's pairing-ordered mesh coordinates. Without it the
# package still loads, with the Ferrite backend only.
# Tracking: delete this gate once a PDEBase release with the M0 layer (forge/m0-all) is the
# compat floor.
const MESH_PDEBASE = isdefined(PDEBase, :AbstractMeshDiscretization) &&
    isdefined(PDEBase, :classify_condition) && isdefined(PDEBase, :substitute_traces) &&
    isdefined(PDEBase, :PieceBoundary) && isdefined(PDEBase, :StrongForm) && all(
    name -> isdefined(getfield(PDEBase, :StrongForm), name),
    (:analyse, :declared_fields, :boundary_conditions, :pair_equations)
) && _orders_mesh_axes_by_pairing()

@static if MESH_PDEBASE
    using PDEBase: StrongForm, MeshDomain, boundarypiece, boundarypieces, NormalDerivative
    include("discretization.jl")
    include("frontend.jl")
    _is_normal_derivative(op) = op isa NormalDerivative
    export FEMDiscretization
    export MeshDomain, boundarypiece, boundarypieces, NormalDerivative
else
    _is_normal_derivative(op) = false
end

end # module
