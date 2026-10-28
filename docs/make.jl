using Documenter, FiniteElementMethod

include("pages.jl")

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
