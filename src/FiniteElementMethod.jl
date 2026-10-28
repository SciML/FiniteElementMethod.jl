"""
    FiniteElementMethod

A SciML discretizer that takes a ModelingToolkit `PDESystem` in strong form on a mesh,
derives the Galerkin weak form, assembles it with Ferrite and lowers it to a `DAEProblem`.

This package is under construction and does not yet export any functionality.
"""
module FiniteElementMethod

using DomainSets: DomainSets
using Ferrite: Ferrite
using ModelingToolkit: ModelingToolkit
using PDEBase: PDEBase
using SciMLBase: SciMLBase
using Symbolics: Symbolics, Num

include("weakform.jl")
include("kernels.jl")
include("assembly.jl")

include("mesh.jl")
include("space.jl")

end # module
