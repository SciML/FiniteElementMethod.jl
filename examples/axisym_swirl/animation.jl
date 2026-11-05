# Renders the animation of docs/src/axisymmetric_swirl.md into docs/src/assets (committed; the documentation build
# does not solve anything). From the repository root, with the docs environment (it has CairoMakie):
#
#     julia --project=docs examples/axisym_swirl/animation.jl
#
# One solve of `axisym_swirl.jl` at c = 0.1, tau_end = 1e-2 (the coarsest of the page's three meshes, 32k unknowns,
# about half a minute after compilation), sampled at frames spaced evenly in log(tau). Rendering also writes the
# checksum of the source files that test/NavierStokes/axisym_swirl.jl compares.
using CairoMakie

include(joinpath(@__DIR__, "axisym_swirl.jl"))
include(joinpath(@__DIR__, "checksum.jl"))

const AXISYM_ASSETS = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")

function render_axisym_animation(directory = AXISYM_ASSETS; c = 0.1, tau_end = 1.0e-2, frames = 72, tau_start = 0.8)
    mkpath(directory)
    grid = axisym_grid(; c, tau_end)
    run = axisym_solve(; c, tau_end, report = ())
    nodes = [Ferrite.get_node_coordinate(n).data for n in Ferrite.getnodes(grid)]
    edges = Point2f[]
    for cell in Ferrite.getcells(grid), (a, b) in ((1, 2), (2, 3), (3, 1))
        push!(edges, Point2f(nodes[cell.nodes[a]]), Point2f(nodes[cell.nodes[b]]), Point2f(NaN, NaN))
    end
    taus = exp.(range(log(tau_start), log(tau_end), frames))
    fixed = 2.0      # the fixed window is r in [0, 2], z in [-2, 2]
    rs_fixed, zs_fixed = range(0, fixed, 100), range(-fixed, fixed, 200)
    rs_unit, zs_unit = range(0, 3, 90), range(-3, 3, 180)   # the following window, in units of the core's scales
    fixed_swirl(tau) = run.sol(1 - tau, collect(rs_fixed), collect(zs_fixed); dv = ut(t, r, z))
    # In the following window r = x sqrt(tau), z = y tau^(1/2 - h) and the fields are multiplied by tau^(1/2 + h),
    # which is the self-similar shape of manufactured.jl: it changes only through the ramp and the mesh.
    function following(tau, field)
        rs, zs = collect(rs_unit .* sqrt(tau)), collect(zs_unit .* tau^EXPONENT_D)
        return run.sol(1 - tau, rs, zs; dv = field) .* tau^EXPONENT_A
    end
    swirl_max = maximum(maximum(fixed_swirl(tau)) for tau in taus)

    frame = Observable(1)
    tau = @lift taus[$frame]
    fig = Figure(size = (1260, 500))
    Label(
        fig[0, 1:6], @lift("t = $(round(1 - $tau; digits = 4)),   τ = 1 - t = $(round($tau; sigdigits = 3))");
        fontsize = 22, tellwidth = false
    )
    ax1 = Axis(
        fig[1, 1]; title = "u_θ in a fixed window", xlabel = "r", ylabel = "z", aspect = DataAspect(),
        limits = (0, fixed, -fixed, fixed)
    )
    hm1 = heatmap!(ax1, rs_fixed, zs_fixed, @lift(fixed_swirl(taus[$frame])); colormap = :viridis, colorrange = (0, swirl_max))
    lines!(ax1, edges; linewidth = 0.3, color = (:white, 0.35))
    Colorbar(fig[1, 2], hm1; label = "u_θ")
    for (k, (field, name)) in enumerate(((ut(t, r, z), "u_θ"), (uz(t, r, z), "u_z")))
        ax = Axis(
            fig[1, 2k + 1]; title = "$name · τ^(1/2+h), in a window that follows the core", xlabel = "r / τ^(1/2)",
            ylabel = k == 1 ? "z / τ^(1/2-h)" : "", aspect = DataAspect(), titlesize = 13
        )
        hm = heatmap!(
            ax, rs_unit, zs_unit, @lift(following(taus[$frame], field));
            colormap = k == 1 ? :viridis : :balance, colorrange = k == 1 ? (0, 3.2) : (-1.4, 1.4)
        )
        Colorbar(fig[1, 2k + 2], hm)
    end
    record(fig, joinpath(directory, "axisym_swirl.gif"), 1:frames; framerate = 12, px_per_unit = 1) do index
        frame[] = index
    end
    write(joinpath(directory, "axisym_swirl_animation.fnv1a"), animation_sources_hash() * "\n")
    return directory
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_axisym_animation(isempty(ARGS) ? AXISYM_ASSETS : ARGS[1])
end
