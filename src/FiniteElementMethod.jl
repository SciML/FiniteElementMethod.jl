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

include("numeric_dae.jl")

include("mesh.jl")
include("space.jl")
include("solution.jl")
include("weakform.jl")
include("kernels.jl")
include("assembly.jl")
export FieldInfo, FEMMetadata, piece_nodes, normal_flux, write_vtk
# Re-exported so that the user API of a FEM model needs no `using PDEBase, Ferrite`.
export Lagrange, RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron

# Whether the installed PDEBase has the mesh-discretization layer the PDESystem front end is
# built on (spec §4). Without it the package still loads, with the Ferrite backend only.
# Tracking: delete this gate once a PDEBase release with the mesh classifier and
# StrongForm boundary routing is the compat floor.
const MESH_PDEBASE = isdefined(PDEBase, :AbstractMeshDiscretization) &&
    isdefined(PDEBase, :classify_condition) && isdefined(PDEBase, :substitute_traces) &&
    isdefined(PDEBase, :PieceBoundary) && isdefined(PDEBase, :StrongForm) && all(
    name -> isdefined(getfield(PDEBase, :StrongForm), name),
    (:analyse, :declared_fields, :route_natural, :essential_condition, :pair_equations)
)

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
