# Showcase

![Heat spreading from a pulsing heated hole through a plate and cooling through its right edge](assets/showcase.gif)

Heat diffuses into a 2 × 1 plate with two holes, meshed by Gmsh from
[`examples/plate.geo`](https://github.com/SciML/FiniteElementMethod.jl/blob/master/examples/plate.geo).
The larger hole is a heated pipe with the pulsing Dirichlet data `u = sin²(t)`, the plate's left,
top and bottom edges and the second hole are insulated (`∂ₙu = 0`), and the right edge cools
convectively (`k ∂ₙu + α u = 0`). The
code is the block in the
[README](https://github.com/SciML/FiniteElementMethod.jl#readme); the documentation build runs it
and renders this animation with
[`examples/showcase.jl`](https://github.com/SciML/FiniteElementMethod.jl/blob/master/examples/showcase.jl),
resampling the solution on a regular grid with `sol(ts, xs, ys; dv = u(t, x, y))`, which returns
`NaN` inside the holes.
