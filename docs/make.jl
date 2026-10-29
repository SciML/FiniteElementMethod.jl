using Documenter, FiniteElementMethod

include("pages.jl")

# The README showcase animation, rendered from the README's own code block (examples/showcase.jl)
# and hosted at assets/showcase.gif, the URL the README embeds.
showcase_page = joinpath(@__DIR__, "src", "showcase.md")
include(joinpath(@__DIR__, "..", "examples", "showcase.jl"))
mkpath(joinpath(@__DIR__, "src", "assets"))
# invokelatest: record_showcase is defined by the include above, inside this same block.
Base.invokelatest(record_showcase, joinpath(@__DIR__, "src", "assets", "showcase.gif"))
cp(joinpath(@__DIR__, "showcase.md"), showcase_page; force = true)
insert!(pages, 2, "Showcase" => "showcase.md")

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
