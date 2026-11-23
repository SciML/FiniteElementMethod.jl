# The heat equation u_t = u_xx + u_yy on the unit square, periodic in x and y, against its exact Fourier mode
# exp(-8 pi^2 t) sin(2 pi (x + y)): prints the L2 error at t = 0.05 on 8, 16 and 32 cells per side and the rates.
# From the repository root, with the docs environment (it has the ODE solver):
#
#     julia --project=docs examples/periodic_heat.jl
using FiniteElementMethod, ModelingToolkit, DomainSets, OrdinaryDiffEqBDF
using Ferrite: generate_grid, Triangle, Vec
using ModelingToolkit: t_nounits as t
@variables x y u(..)
Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
exact(x_, y_) = exp(-8pi^2 * 0.05) * sin(2pi * (x_ + y_))
function l2err(n)
    Ω = MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))
    left, right, bottom, top = boundarypiece.(Ω, (:left, :right, :bottom, :top))
    bcs = [u(t, left) ~ u(t, right), u(t, bottom) ~ u(t, top), u(0, x, y) ~ sin(2pi * (x + y))]
    @named pdesys = PDESystem([Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 0.05)], [t, x, y], [u(t, x, y)], [])
    sol = solve(discretize(pdesys, FEMDiscretization(Ω, t)), DFBDF(); abstol = 1.0e-10, reltol = 1.0e-10)
    pts = range(1 / 80, 1 - 1 / 80; length = 40)
    return sqrt(sum(abs2, sol(0.05, collect(pts), collect(pts); dv = u(t, x, y)) .- [exact(a, b) for a in pts, b in pts]) / 40^2)
end
errs = l2err.((8, 16, 32))
println("L2 errors at t = 0.05 on 8, 16, 32 cells per side: ", errs)
println("convergence rates: ", log2.(errs[1:(end - 1)] ./ errs[2:end]))

# Expected output:
#
#     L2 errors at t = 0.05 on 8, 16, 32 cells per side: (0.0031009654308897044, 0.0008420798111805154, 0.0002148976007795359)
#     convergence rates: (1.8806885615256936, 1.9703076005356397)
