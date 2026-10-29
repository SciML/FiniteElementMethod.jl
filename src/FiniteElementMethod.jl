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

using PDEBase: StrongForm, MeshDomain, boundarypiece, boundarypieces, NormalDerivative
include("discretization.jl")
include("frontend.jl")
_is_normal_derivative(op) = op isa NormalDerivative
export FEMDiscretization
export MeshDomain, boundarypiece, boundarypieces, NormalDerivative

end # module
