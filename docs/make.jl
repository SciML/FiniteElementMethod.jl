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

# The README showcase animation, rendered from the README's own code block (examples/showcase.jl)
# and hosted at assets/showcase.gif, the URL the README embeds. It needs the front end as well.
showcase_page = joinpath(@__DIR__, "src", "showcase.md")
if FiniteElementMethod.MESH_PDEBASE
    include(joinpath(@__DIR__, "..", "examples", "showcase.jl"))
    mkpath(joinpath(@__DIR__, "src", "assets"))
    # invokelatest: record_showcase is defined by the include above, inside this same block.
    Base.invokelatest(record_showcase, joinpath(@__DIR__, "src", "assets", "showcase.gif"))
    cp(joinpath(@__DIR__, "showcase.md"), showcase_page; force = true)
    insert!(pages, 2, "Showcase" => "showcase.md")
else
    rm(showcase_page; force = true)
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
