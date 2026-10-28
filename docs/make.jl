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

# The benchmark page is rendered from benchmarks/results/*.toml at build time (benchmarks/render.jl).
include(joinpath(@__DIR__, "..", "benchmarks", "render.jl"))
write(joinpath(@__DIR__, "src", "benchmarks.md"), render(joinpath(@__DIR__, "..", "benchmarks", "results")))
insert!(pages, length(pages), "Benchmarks" => "benchmarks.md")
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
