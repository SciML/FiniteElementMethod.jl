# Renders the README's Navier–Stokes images: runs the README's second and third code blocks (the simulation and
# the plotting) in one module and writes ns_showcase.png, ns_showcase.gif and the checksum of the two blocks
# (ns_showcase.fnv1a, checked by test/Frontend/readme.jl) to a directory. By default that is docs/src/assets,
# where the committed copies the README links to live. To refresh them after changing the blocks, from the
# repository root, with the docs environment (it has CairoMakie):
#
#     julia --project=docs examples/ns_showcase.jl
#
# The documentation build calls `render_ns_showcase` with a scratch directory, to check that the blocks run.
include(joinpath(@__DIR__, "readme_code.jl"))

const NS_ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")

function render_ns_showcase(directory = NS_ASSETS)
    mkpath(directory)
    mod = Module(:ReadmeNavierStokes)
    Core.eval(mod, :(include(path) = Base.include($mod, path)))
    cd(directory) do
        for block in (2, 3)   # the plotting block uses the simulation block's `sol`, `u`, `v`, `t`, `x`, `y` and `tend`
            Base.include_string(mod, readme_code(; block), "README.md, block $block")
        end
    end
    write(joinpath(directory, "ns_showcase.fnv1a"), blocks_hash(2:3) * "\n")
    return directory
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_ns_showcase(isempty(ARGS) ? NS_ASSETS : ARGS[1])
end
