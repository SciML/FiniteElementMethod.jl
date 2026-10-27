using Documenter, FiniteElementMethod

include("pages.jl")

# The PDESystem front end exists only when the installed PDEBase has the mesh layer
# (`FiniteElementMethod.MESH_PDEBASE`), and Documenter expands every page under src/, so its
# pages are copied in only then.
for (title, file) in ("Discretization" => "discretization.md", "Extension contract" => "extension_contract.md")
    frontend_page = joinpath(@__DIR__, "src", file)
    if FiniteElementMethod.MESH_PDEBASE
        cp(joinpath(@__DIR__, file), frontend_page; force = true)
        insert!(pages, length(pages), title => file)
    else
        rm(frontend_page; force = true)
    end
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
