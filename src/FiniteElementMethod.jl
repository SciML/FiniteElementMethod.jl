"""
    FiniteElementMethod

A SciML discretizer that takes a ModelingToolkit `PDESystem` in strong form on a mesh,
derives the Galerkin weak form, assembles it with Ferrite and lowers it to a `DAEProblem`.

The entry point is `FEMDiscretization`; `discretize(pdesys, disc)` lowers a `PDESystem`.
"""
module FiniteElementMethod

using ADTypes: AutoForwardDiff
using DiffEqBase: BrownFullBasicInit
using DomainSets: DomainSets
using Ferrite: Ferrite, Lagrange, RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron
using ForwardDiff: ForwardDiff
using LinearAlgebra: cross, dot, lu, lu!, mul!, norm
using ModelingToolkit: ModelingToolkit, @parameters, @variables, Differential, MTKParameters,
    ProblemTypeCtx, System, complete, getmetadata
using NonlinearSolveFirstOrder: NewtonRaphson
using PDEBase: PDEBase
using PrecompileTools: PrecompileTools
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions
using SciMLBase: SciMLBase, DAEFunction, DAEProblem
using SciMLBase.SymbolicIndexingInterface: SymbolicIndexingInterface, getp, parameter_index, variable_index
using SparseArrays: SparseMatrixCSC, dropzeros!, nnz, nonzeros, nzrange, rowvals
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics, Equation, Num, @register_array_symbolic, substitute, unwrap

# Generated kernels are RuntimeGeneratedFunctions owned by this module, so they can be built in a precompile workload.
RuntimeGeneratedFunctions.init(@__MODULE__)

include("numeric_dae.jl")

include("mesh.jl")
include("space.jl")
include("solution.jl")
include("weakform.jl")
include("kernels.jl")
include("assembly.jl")
include("operators.jl")
export FieldInfo, FEMMetadata, piece_nodes, normal_flux, write_vtk, facet_dofs, dof_coordinates, cell_dofs
# Re-exported so that the user API of a FEM model needs no `using PDEBase, Ferrite`.
export Lagrange, RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron

using PDEBase: StrongForm, MeshDomain, Region, boundarypiece, boundarypieces, NormalDerivative
using PDEBase.StrongForm: Weighted
include("discretization.jl")
include("frontend.jl")
_is_normal_derivative(op) = op isa NormalDerivative
export FEMDiscretization, Galerkin, SUPG, add_boundary_pieces!, ferrite_grid
export MeshDomain, Region, boundarypiece, boundarypieces, NormalDerivative
# The integration-by-parts policy `ibp = Dict(eq => Weighted(w))` of `FEMDiscretization`.
export Weighted

# Frequency-domain Maxwell: a separate entry point assembled directly on Ferrite, not through the PDESystem front end.
include("maxwell/ports.jl")
include("maxwell/model.jl")
include("maxwell/driven.jl")
include("maxwell/sweep.jl")
include("maxwell/adjoint.jl")
export MaxwellMaterial, MaxwellModel, MaxwellTerm, maxwell_operator, maxwell_operator!
export MaxwellSolution, driven_problem, solve_driven, field_values
export WavePort, MaxwellSweep, solve_sweep
export MaxwellSensitivities, s_parameter_sensitivities, adjoint_gradient

include("precompile.jl")

end # module
