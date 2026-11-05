# Renders the two figures of docs/src/axisymmetric_swirl.md into docs/src/assets (committed; the documentation
# build does not solve anything). From the repository root, with the docs environment (it has CairoMakie):
#
#     julia --project=docs examples/axisym_swirl/figures.jl
#
# One solve at c = 0.1, tau_end = 1e-2 (32k unknowns, about half a minute after compilation). Rendering also
# writes the checksum of the source files that test/NavierStokes/axisym_swirl.jl compares.
using CairoMakie

include(joinpath(@__DIR__, "axisym_swirl.jl"))
include(joinpath(@__DIR__, "checksum.jl"))

const AXISYM_ASSETS = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")

function render_axisym_figures(directory = AXISYM_ASSETS; c = 0.1, tau_end = 1.0e-2)
    mkpath(directory)
    grid = axisym_grid(; c, tau_end)
    nodes = [Ferrite.get_node_coordinate(n).data for n in Ferrite.getnodes(grid)]
    edges = Point2f[]
    for cell in Ferrite.getcells(grid), (a, b) in ((1, 2), (2, 3), (3, 1))
        push!(edges, Point2f(nodes[cell.nodes[a]]), Point2f(nodes[cell.nodes[b]]), Point2f(NaN, NaN))
    end
    fig = Figure(size = (900, 430))
    for (k, (extent, title)) in enumerate(((8.0, "whole domain"), (0.3, "within 0.3 of the origin")))
        ax = Axis(fig[1, k]; title, xlabel = "r", ylabel = "z", aspect = DataAspect(), limits = (0, extent, -extent, extent))
        lines!(ax, edges; linewidth = 0.4, color = :black)
    end
    save(joinpath(directory, "axisym_mesh.png"), fig; px_per_unit = 1)

    report = vcat(0.05:0.05:0.95, 0.96:0.01:0.99)
    run = axisym_solve(; c, tau_end, report)
    fig = Figure(size = (950, 780))
    for (k, tt) in enumerate((0.5, 0.9, 0.99))
        scale = sqrt(1 - tt)
        rs, zs = range(0, 3scale, 120), range(-3scale, 3scale, 240)
        swirl = run.sol(tt, collect(rs), collect(zs); dv = ut(t, r, z))
        ax = Axis(
            fig[1, k]; title = "swirl u_θ at t = $tt", xlabel = "r", ylabel = k == 1 ? "z" : "",
            aspect = DataAspect()
        )
        hm = heatmap!(ax, rs, zs, swirl; colormap = :viridis)
        Colorbar(fig[2, k], hm; vertical = false, flipaxis = false)
    end
    ax = Axis(
        fig[3, 1:3]; xlabel = "t", ylabel = "relative r-weighted L² error", yscale = log10,
        title = "c = $c: the error stays flat while the core collapses"
    )
    scatterlines!(ax, [e[1] for e in run.errors], [e[2] for e in run.errors]; label = "velocity")
    scatterlines!(ax, [e[1] for e in run.errors], [e[3] for e in run.errors]; label = "pressure")
    axislegend(ax; position = :rc)
    save(joinpath(directory, "axisym_core.png"), fig; px_per_unit = 1)
    write(joinpath(directory, "axisym_swirl.fnv1a"), figure_sources_hash() * "\n")
    return directory
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_axisym_figures(isempty(ARGS) ? AXISYM_ASSETS : ARGS[1])
end
