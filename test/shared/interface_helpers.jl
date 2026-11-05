# Helpers shared by the Interface group. They use only the public solution interface, so each case
# reads as a user's script followed by a comparison with the exact solution.
using Test

"""
    sample_points(n; lo = 0.0, hi = 1.0)

The `n` cell midpoints of a uniform partition of `[lo, hi]`, the sample points of [`l2_error`](@ref).
"""
sample_points(n; lo = 0.0, hi = 1.0) = range(lo + (hi - lo) / 2n, hi - (hi - lo) / 2n; length = n)

"""
    l2_error(sol, dv, exact, t, points...)

The L2 distance at time `t` between the field `dv` of the solution `sol` and the function
`exact(x...)`, on the tensor product of the sample `points` (one range per coordinate, as made by
[`sample_points`](@ref)). Each sample carries the volume `prod(step.(points))` of its cell, so the
midpoint rule approximates the integral over the domain the points cover.
"""
function l2_error(sol, dv, exact, t::Real, points...)
    return l2_distance(sol(t, collect.(points)...; dv), exact, points)
end

"""
    l2_error(sol, dv, exact, points...)

The same for a steady solution, which is called without a time: `sol(xs...; dv)`.
"""
l2_error(sol, dv, exact, points::AbstractRange...) = l2_distance(sol(collect.(points)...; dv), exact, points)

function l2_distance(computed, exact, points)
    expected = [exact(x...) for x in Iterators.product(points...)]
    cell_volume = prod(step.(points))
    return sqrt(sum(abs2, computed .- expected) * cell_volume)
end

"""
    observed_rate(errors, hs)

The convergence rates `log(errors[i - 1] / errors[i]) / log(hs[i - 1] / hs[i])` between successive
refinements of mesh size `hs`.
"""
observed_rate(errors, hs) = [log(errors[i - 1] / errors[i]) / log(hs[i - 1] / hs[i]) for i in 2:length(errors)]
