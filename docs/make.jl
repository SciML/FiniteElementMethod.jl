using Documenter, FiniteElementMethod

include("pages.jl")

# The PDESystem front end exists only when the installed PDEBase has the mesh layer
# (`FiniteElementMethod.MESH_PDEBASE`), and Documenter expands every page under src/, so its
# page is copied in only then.
frontend_page = joinpath(@__DIR__, "src", "discretization.md")
if FiniteElementMethod.MESH_PDEBASE
    cp(joinpath(@__DIR__, "discretization.md"), frontend_page; force = true)
    insert!(pages, length(pages), "Discretization" => "discretization.md")
else
    rm(frontend_page; force = true)
end

makedocs(
    sitename = "FiniteElementMethod.jl",
    authors = "Alex Jones et al.",
    clean = true,
    doctest = true,
    modules = [FiniteElementMethod],
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/FiniteElementMethod/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/FiniteElementMethod.jl"; push_preview = true)
