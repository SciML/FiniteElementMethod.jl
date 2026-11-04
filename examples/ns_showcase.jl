# Renders the README's Navier–Stokes figure and animation: runs the README's second code block, resamples
# the velocity on a regular grid with `sol(ts, xs, ys; dv)` (NaN inside the cylinder) and records the speed
# and the vorticity with Makie. The documentation build calls `record_ns_showcase` and `save_ns_figure`
# (docs/make.jl); to run them by hand, from an environment with FiniteElementMethod, CairoMakie, Ferrite,
# Gmsh, LinearSolve, SciMLBase, OrdinaryDiffEqBDF, ModelingToolkit and DomainSets:
#
#     julia --project=docs examples/ns_showcase.jl ns_showcase.gif ns_showcase.png
using CairoMakie

include(joinpath(@__DIR__, "readme_code.jl"))

run_ns_readme() = run_readme(; block = 2, names = (:sol, :u, :v, :x, :y, :t, :tend))

# Speed |(u, v)| and vorticity ∂ₓv − ∂ᵧu of the velocity sampled at times `ts` on the grid `xs` × `ys`.
# The vorticity is a central difference of the sampled velocity: NaN in the cylinder and on its border.
function sample_flow(m, ts, xs, ys)
    U = m.sol(collect(ts), collect(xs), collect(ys); dv = m.u(m.t, m.x, m.y))   # time × nx × ny
    V = m.sol(collect(ts), collect(xs), collect(ys); dv = m.v(m.t, m.x, m.y))
    speed = hypot.(U, V)
    vorticity = fill(NaN, size(U))
    dx, dy = step(xs), step(ys)
    i, j = 2:(length(xs) - 1), 2:(length(ys) - 1)
    @views vorticity[:, i, j] .= (V[:, i .+ 1, j] .- V[:, i .- 1, j]) ./ 2dx .- (U[:, i, j .+ 1] .- U[:, i, j .- 1]) ./ 2dy
    return speed, vorticity
end

function ns_figure(xs, ys, speed, vorticity, label; size = (720, 350), speed_max = 1.6, vorticity_max = 40)
    fig = Figure(; size)
    options = (; aspect = DataAspect(), xlabel = "x", ylabel = "y")
    top = Axis(fig[1, 1]; title = label, options...)
    bottom = Axis(fig[2, 1]; options...)
    h_speed = heatmap!(top, xs, ys, speed; colormap = :viridis, colorrange = (0, speed_max), nan_color = :white)
    h_vorticity = heatmap!(bottom, xs, ys, vorticity; colormap = :balance, colorrange = (-vorticity_max, vorticity_max), nan_color = :white)
    Colorbar(fig[1, 2], h_speed; label = "speed |u|")
    Colorbar(fig[2, 2], h_vorticity; label = "vorticity")
    return fig
end

const NS_GRID = (xs = range(0, 2.2; length = 440), ys = range(0, 0.41; length = 82))

function record_ns_showcase(path; frames = 90, framerate = 18, m = run_ns_readme())
    return Base.invokelatest(render_ns_animation, m, path; frames, framerate)
end

function render_ns_animation(m, path; frames, framerate)
    (; xs, ys) = NS_GRID
    ts = range(m.sol.t[1], m.sol.t[end]; length = frames)
    speed, vorticity = sample_flow(m, ts, xs, ys)
    frame_speed, frame_vorticity = Observable(speed[1, :, :]), Observable(vorticity[1, :, :])
    label = Observable("t = 0.00")
    fig = ns_figure(xs, ys, frame_speed, frame_vorticity, label)
    record(fig, path, eachindex(ts); framerate, px_per_unit = 1) do k
        frame_speed[], frame_vorticity[] = speed[k, :, :], vorticity[k, :, :]
        label[] = "t = $(round(ts[k]; digits = 2))"
    end
    return path
end

# The figure: the flow at the final time.
save_ns_figure(path; m = run_ns_readme()) = Base.invokelatest(render_ns_figure, m, path)

function render_ns_figure(m, path)
    (; xs, ys) = NS_GRID
    speed, vorticity = sample_flow(m, [m.sol.t[end]], xs, ys)
    label = "t = $(round(m.sol.t[end]; digits = 2))"
    save(path, ns_figure(xs, ys, speed[1, :, :], vorticity[1, :, :], label); px_per_unit = 2)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = run_ns_readme()
    record_ns_showcase(isempty(ARGS) ? "ns_showcase.gif" : ARGS[1]; m)
    save_ns_figure(length(ARGS) < 2 ? "ns_showcase.png" : ARGS[2]; m)
end
