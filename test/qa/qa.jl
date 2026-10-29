using SciMLTesting, FiniteElementMethod, Test
using JET

# `Ferrite.DofOrder.FieldWise` is Ferrite's documented renumbering API but is not declared `public`;
# ForwardDiff documents `derivative`, and `Dual`, `Partials`, `value` and `partials` for custom rules, as its API
# but declares none of them `public`;
# `Ferrite.dirichlet_facetdof_indices` (a facet's local nodes) has no public spelling in Ferrite 1.
# The front end uses these PDEBase names, public on the M0 stack but not on PDEBase 0.1.39 (and
# `Base.ispublic` does not exist on Julia 1.10), so each is ignored only where it is not public.
# Tracking: drop NONPUBLIC_PDEBASE once the PDEBase compat floor is the M0 release.
const PDEBase = FiniteElementMethod.PDEBase
ispublic_in(mod, name) = isdefined(Base, :ispublic) && Base.ispublic(mod, name)
const NONPUBLIC_PDEBASE = Tuple(
    name for (mod, name) in (
            (PDEBase, :make_pdesys_compatible), (PDEBase, :parse_bcs),
            (isdefined(PDEBase, :StrongForm) ? PDEBase.StrongForm : PDEBase, :pair_equations),
        ) if !ispublic_in(mod, name)
)
# ModelingToolkit re-exports these ModelingToolkitBase names; on Julia 1.10 `Base.which` reports
# ModelingToolkitBase as their owner, on 1.11+ ModelingToolkit, the module we depend on.
const LTS_MTK_OWNER = VERSION < v"1.11" ? (:ProblemTypeCtx, :t_nounits) : ()

run_qa(
    FiniteElementMethod;
    # The PDEBase mesh-domain interface is extended by methods on the backend's mesh type, by design.
    aqua_kwargs = (; piracies = (; treat_as_own = [FiniteElementMethod.Ferrite.Grid])),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :FieldWise, :derivative, :Dual, :Partials, :value, :partials, :dirichlet_facetdof_indices,
                NONPUBLIC_PDEBASE...,
            ),
        ),
        all_explicit_imports_via_owners = (; ignore = LTS_MTK_OWNER),
        all_qualified_accesses_via_owners = (; ignore = LTS_MTK_OWNER),
    ),
    # FEMDiscretization and Galerkin are rendered on the Discretization page, which docs/make.jl
    # copies into docs/src from docs/discretization.md only when the front end is available.
    api_docs_kwargs = (; rendered_ignore = (:FEMDiscretization, :Galerkin)),
    # The user API of a FEM model: PDEBase's mesh vocabulary and Ferrite's interpolations.
    reexports_allow = (
        :MeshDomain, :Region, :boundarypiece, :boundarypieces, :NormalDerivative, :Lagrange,
        :RefLine, :RefTriangle, :RefQuadrilateral, :RefTetrahedron, :RefHexahedron,
    ),
)
