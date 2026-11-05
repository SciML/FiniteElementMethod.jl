using Documenter, FiniteElementMethod

include("pages.jl")

# The README showcase animation is rendered from the README's own code block (examples/showcase.jl). The committed
# copy in docs/src/assets is what the README and the Showcase page show; this build re-runs the block into a scratch
# directory to check that it still runs in the documentation environment.
showcase_page = joinpath(@__DIR__, "src", "showcase.md")
include(joinpath(@__DIR__, "..", "examples", "showcase.jl"))
# invokelatest: render_showcase_assets is defined by the include above.
Base.invokelatest(render_showcase_assets, mktempdir())
cp(joinpath(@__DIR__, "showcase.md"), showcase_page; force = true)
# The README's Navier–Stokes figure and animation come from its second and third code blocks (examples/ns_showcase.jl).
# The committed copies in docs/src/assets are what the README and the Navier–Stokes page show; this build re-runs the
# blocks into a scratch directory to check that they still run in the documentation environment.
include(joinpath(@__DIR__, "..", "examples", "ns_showcase.jl"))
Base.invokelatest(render_ns_showcase, mktempdir())
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
