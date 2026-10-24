using Documenter, FiniteElementMethod

include("pages.jl")

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
