# Renders the README showcase animation: runs the README's code block, resamples the solution on
# a regular grid with `sol(ts, xs, ys; dv)` (NaN in the holes) and records it with Makie. The gif and the checksum
# of the README block (showcase.fnv1a, checked by test/Frontend/readme.jl) go to a directory, by default
# docs/src/assets, where the committed copies the README links to live. To refresh them after changing the block,
# from the repository root, with the docs environment (it has CairoMakie):
#
#     julia --project=docs examples/showcase.jl
#
# The documentation build calls `render_showcase_assets` with a scratch directory, to check that the block runs.
using CairoMakie

include(joinpath(@__DIR__, "readme_code.jl"))

# The README's variables gain call methods while the README runs, so the rendering runs in the latest
# world, after the solve.
record_showcase(path; kwargs...) = Base.invokelatest(render_showcase, run_readme(), path; kwargs...)

const SHOWCASE_ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")

function render_showcase_assets(directory = SHOWCASE_ASSETS)
    mkpath(directory)
    record_showcase(joinpath(directory, "showcase.gif"))
    write(joinpath(directory, "showcase.fnv1a"), blocks_hash(1:1) * "\n")
    return directory
end

function render_showcase(m, path; frames = 72, framerate = 12, resolution = (280, 140), size = (560, 300))
    sol, u, t, x, y = m.sol, m.u, m.t, m.x, m.y
    xs = range(0, 2; length = resolution[1])
    ys = range(0, 1; length = resolution[2])
    ts = range(sol.t[1], sol.t[end]; length = frames)
    field = sol(collect(ts), collect(xs), collect(ys); dv = u(t, x, y))   # frames × nx × ny
    frame = Observable(field[1, :, :])
    label = Observable("t = 0.00")
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = label, xlabel = "x", ylabel = "y")
    hm = heatmap!(ax, xs, ys, frame; colormap = :inferno, colorrange = (0, 1), nan_color = :white)
    Colorbar(fig[1, 2], hm; label = "u")
    record(fig, path, eachindex(ts); framerate, px_per_unit = 1) do k
        frame[] = field[k, :, :]
        label[] = "t = $(round(ts[k]; digits = 2))"
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_showcase_assets(isempty(ARGS) ? SHOWCASE_ASSETS : ARGS[1])
end
