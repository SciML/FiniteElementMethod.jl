# The README's showcase code block, so that the test that runs it and the documentation build
# that animates it both use exactly what the README shows.

# The first ```julia block of the README.
function readme_code(readme = joinpath(@__DIR__, "..", "README.md"))
    m = match(r"```julia\n(.*?)```"s, read(readme, String))
    m === nothing && error("$readme has no ```julia block")
    return m.captures[1]
end

# Run the README block in a fresh module and return its `sol`, `u`, `x`, `y`, `t` and pieces as a
# named tuple. The bindings are read with invokelatest: they are newer than the caller's world.
function run_readme(readme = joinpath(@__DIR__, "..", "README.md"))
    mod = Module(:ReadmeShowcase)
    Core.eval(mod, :(include(path) = Base.include($mod, path)))
    Base.include_string(mod, readme_code(readme), readme)
    names = (:sol, :u, :x, :y, :t, :Γ_in, :Γ_wall, :Γ_out)
    return NamedTuple{names}(map(name -> Base.invokelatest(getglobal, mod, name), names))
end
