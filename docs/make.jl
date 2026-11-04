using Documenter, FiniteElementMethod

include("pages.jl")

# The README showcase animation, rendered from the README's own code block (examples/showcase.jl)
# and hosted at assets/showcase.gif, the URL the README embeds.
showcase_page = joinpath(@__DIR__, "src", "showcase.md")
include(joinpath(@__DIR__, "..", "examples", "showcase.jl"))
mkpath(joinpath(@__DIR__, "src", "assets"))
# invokelatest: record_showcase is defined by the include above.
Base.invokelatest(record_showcase, joinpath(@__DIR__, "src", "assets", "showcase.gif"))
cp(joinpath(@__DIR__, "showcase.md"), showcase_page; force = true)
# The README's Navier–Stokes figure and animation, rendered from its second code block (examples/ns_showcase.jl)
# with one solve, and hosted at assets/ns_showcase.{png,gif}, the URLs the README embeds.
include(joinpath(@__DIR__, "..", "examples", "ns_showcase.jl"))
ns_flow = Base.invokelatest(run_ns_readme)
Base.invokelatest(save_ns_figure, joinpath(@__DIR__, "src", "assets", "ns_showcase.png"); m = ns_flow)
Base.invokelatest(record_ns_showcase, joinpath(@__DIR__, "src", "assets", "ns_showcase.gif"); m = ns_flow)
insert!(pages, 2, "Showcase" => "showcase.md")
# The benchmark page is rendered from benchmarks/results/*.toml at build time (benchmarks/render.jl).
include(joinpath(@__DIR__, "..", "benchmarks", "render.jl"))
write(joinpath(@__DIR__, "src", "benchmarks.md"), render(joinpath(@__DIR__, "..", "benchmarks", "results")))
insert!(pages, length(pages), "Benchmarks" => "benchmarks.md")

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
