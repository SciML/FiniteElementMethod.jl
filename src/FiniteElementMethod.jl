"""
    FiniteElementMethod

A SciML discretizer that takes a ModelingToolkit `PDESystem` in strong form on a mesh,
derives the Galerkin weak form, assembles it with Ferrite and lowers it to a `DAEProblem`.

This package is under construction and does not yet export any functionality.
"""
module FiniteElementMethod

using ADTypes: AutoForwardDiff
using DiffEqBase: BrownFullBasicInit
using DomainSets: DomainSets
using Ferrite: Ferrite
using ForwardDiff: ForwardDiff
using LinearAlgebra: mul!
using ModelingToolkit: ModelingToolkit, @parameters, @variables, Differential, MTKParameters,
    ProblemTypeCtx, System, complete, getmetadata
using NonlinearSolveFirstOrder: NewtonRaphson
using PDEBase: PDEBase
using PrecompileTools: PrecompileTools
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions
using SciMLBase: SciMLBase, DAEFunction, DAEProblem
using SciMLBase.SymbolicIndexingInterface: getp, variable_index
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals
using Symbolics: Symbolics, Num, @register_array_symbolic

# Generated kernels are RuntimeGeneratedFunctions owned by this module, so they can be built in a precompile workload.
RuntimeGeneratedFunctions.init(@__MODULE__)

include("weakform.jl")
include("kernels.jl")
include("assembly.jl")
include("operators.jl")
include("precompile.jl")

include("mesh.jl")
include("space.jl")
include("numeric_dae.jl")

end # module
