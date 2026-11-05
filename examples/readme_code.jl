# The README's showcase code block, so that the test that runs it and the documentation build
# that animates it both use exactly what the README shows.

# The `block`-th ```julia block of the README: 1 is the heat showcase, 2 the Navier–Stokes one.
function readme_code(readme = joinpath(@__DIR__, "..", "README.md"); block = 1)
    blocks = collect(eachmatch(r"```julia\n(.*?)```"s, read(readme, String)))
    length(blocks) < block && error("$readme has no ```julia block number $block")
    return blocks[block].captures[1]
end

# Run a README block in a fresh module and return the bindings `names` (by default the heat showcase's
# `sol`, `u`, `x`, `y`, `t` and pieces) as a named tuple. The bindings are read with invokelatest: they
# are newer than the caller's world.
function run_readme(
        readme = joinpath(@__DIR__, "..", "README.md"); block = 1,
        names = (:sol, :u, :x, :y, :t, :Γ_in, :Γ_wall, :Γ_out)
    )
    mod = Module(:ReadmeShowcase)
    Core.eval(mod, :(include(path) = Base.include($mod, path)))
    Base.include_string(mod, readme_code(readme; block), readme)
    return NamedTuple{names}(map(name -> Base.invokelatest(getglobal, mod, name), names))
end

# FNV-1a (64 bit) checksum of the README's code blocks numbered `blocks`, in plain Julia so that it needs no package.
# The images rendered from the blocks are committed with this checksum next to them, and a test fails when the blocks
# change without a re-render: the heat showcase is block 1, the Navier–Stokes simulation and plotting blocks 2 and 3.
function blocks_hash(blocks, readme = joinpath(@__DIR__, "..", "README.md"))
    checksum = 0xcbf29ce484222325
    for byte in codeunits(join(readme_code(readme; block) for block in blocks))
        checksum = (checksum ⊻ byte) * 0x00000100000001b3
    end
    return string(checksum; base = 16, pad = 16)
end
