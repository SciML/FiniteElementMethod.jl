# Spec §10.4 (the linear M2 cases and the M4 two-field case): each problem is written once on a
# DomainSets Rectangle with trace conditions (spec §3.3), discretized by MethodOfLines'
# MOLFiniteDifference on an n×n grid and by FEMDiscretization on a P1 quadrilateral mesh of the same
# box whose facet sets carry the same piece names, so the two share their nodes. Both are measured against the manufactured solution, and at
# equal n the two nodal fields must agree within the sum of their Richardson error estimates; a
# system of several fields must meet both criteria in every field.
# The round trip of the lowering is checked too: PDEBase.to_box_bcs must give exactly the
# endpoint form MethodOfLines' own tests write by hand.
#
# MethodOfLines leaves the corner nodes shared by two faces at zero (existing behaviour, in the
# endpoint form too), so every MethodOfLines error and every comparison excludes the four corners.
using Test
using FiniteElementMethod: FiniteElementMethod, rectangle_grid
using MethodOfLines: MethodOfLines, MOLFiniteDifference
using PDEBase: PDEBase
using Ferrite: Quadrilateral

include(joinpath(@__DIR__, "..", "frontend.jl"))

using ModelingToolkit: t_nounits as t
using OrdinaryDiffEqBDF: FBDF

@parameters α
@variables x y u(..) w(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const NS = (8, 16, 32)
const TSPAN = (0.0, 0.1)
const TOLS = (; abstol = 1.0e-10, reltol = 1.0e-10)
const MIN_RATE = 1.9

ustar(t_, x_, y_) = exp(-t_) * sin(π * x_) * cos(π * y_ / 2) + x_^2 * y_
const U = ustar(t, x, y)
const Ux, Uy = expand_derivatives(Dx(U)), expand_derivatives(Dy(U))
wstar(t_, x_, y_) = exp(-t_ / 2) * cos(π * x_ / 2) * sin(π * y_) + x_ * y_^2
const W = wstar(t, x, y)
const Wx = expand_derivatives(Dx(W))
kcoef(x_, y_) = 1 + x_^2 + y_ / 2

# `g` on the face where `var` = `val`
on(g, var, val) = substitute(g, Dict(var => val))

const RECT = Rectangle(0.0 .. 1.0, 0.0 .. 1.0)

# Each case: the equations, the fields as (variable, manufactured solution) pairs, the grid sizes,
# the trace conditions on the pieces of a domain `Ω` (a Rectangle or a MeshDomain with the same
# piece names), the endpoint form MethodOfLines' tests would write, and the parameters.
function heat_eq(k)
    flux_div = Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y)))
    source = expand_derivatives(Dt(U) - Dx(k * Dx(U)) - Dy(k * Dy(U)))
    return Dt(u(t, x, y)) ~ flux_div + source
end
const SCALAR = [(u, ustar)]

# Two-field reaction–diffusion with cross-diffusion χ∇w in u's flux and linear reactions coupling
# the fields both ways. Linear because the front end rejects nonlinear terms until M3's numeric path
# lands (spec §11.1); the signs of the coupling are not symmetric, so a swapped sign shows up as a
# lost rate. Each flux term is its own divergence: MethodOfLines cannot discretize
# Dx(Dx(u) + χ * Dx(w)) (it fails building an auxiliary variable), and it cannot discretize the
# cross term next to a conormal condition on u, so u is Dirichlet on every face and only w has a
# Neumann face.
const χ, DW = 0.3, 0.5
flux_u(f, g) = Dx(Dx(f)) + Dx(χ * Dx(g)) + Dy(Dy(f)) + Dy(χ * Dy(g))
flux_w(g) = Dx(DW * Dx(g)) + Dy(DW * Dy(g))
function two_field_eqs()
    uu, ww = u(t, x, y), w(t, x, y)
    return [
        Dt(uu) ~ flux_u(uu, ww) - uu + ww + expand_derivatives(Dt(U) - flux_u(U, W) + U - W),
        Dt(ww) ~ flux_w(ww) + uu - 2ww + expand_derivatives(Dt(W) - flux_w(W) - U + 2W),
    ]
end

const CASES = [
    (
        name = "constant-coefficient heat, Neumann on the right",
        eqs = [heat_eq(1)],
        fields = SCALAR,
        ns = NS,
        traces = Ω -> begin
            Γl, Γr, Γb, Γt = boundarypiece.(Ref(Ω), (:left, :right, :bottom, :top))
            [
                u(t, Γl) ~ on(U, x, 0), NormalDerivative(Γr)(u(t, Γr)) ~ on(Ux, x, 1),
                u(t, Γb) ~ on(U, y, 0), u(t, Γt) ~ on(U, y, 1),
            ]
        end,
        endpoint = [
            u(t, 0, y) ~ on(U, x, 0), Dx(u(t, 1, y)) ~ on(Ux, x, 1),
            u(t, x, 0) ~ on(U, y, 0), u(t, x, 1) ~ on(U, y, 1),
        ],
        ps = [],
    ),
    (
        name = "variable k(x, y), conormal Neumann on the bottom",
        eqs = [heat_eq(kcoef(x, y))],
        fields = SCALAR,
        ns = NS,
        traces = Ω -> begin
            Γl, Γr, Γb, Γt = boundarypiece.(Ref(Ω), (:left, :right, :bottom, :top))
            [
                u(t, Γl) ~ on(U, x, 0), u(t, Γr) ~ on(U, x, 1),
                kcoef(x, 0) * NormalDerivative(Γb)(u(t, Γb)) ~ -kcoef(x, 0) * on(Uy, y, 0),
                u(t, Γt) ~ on(U, y, 1),
            ]
        end,
        endpoint = [
            u(t, 0, y) ~ on(U, x, 0), u(t, 1, y) ~ on(U, x, 1),
            kcoef(x, 0) * -Dy(u(t, x, 0)) ~ -kcoef(x, 0) * on(Uy, y, 0),
            u(t, x, 1) ~ on(U, y, 1),
        ],
        ps = [],
    ),
    (
        name = "Robin on the right",
        eqs = [heat_eq(1)],
        fields = SCALAR,
        ns = NS,
        traces = Ω -> begin
            Γl, Γr, Γb, Γt = boundarypiece.(Ref(Ω), (:left, :right, :bottom, :top))
            [
                u(t, Γl) ~ on(U, x, 0),
                NormalDerivative(Γr)(u(t, Γr)) + α * u(t, Γr) ~ on(Ux + α * U, x, 1),
                u(t, Γb) ~ on(U, y, 0), u(t, Γt) ~ on(U, y, 1),
            ]
        end,
        endpoint = [
            u(t, 0, y) ~ on(U, x, 0), Dx(u(t, 1, y)) + α * u(t, 1, y) ~ on(Ux + α * U, x, 1),
            u(t, x, 0) ~ on(U, y, 0), u(t, x, 1) ~ on(U, y, 1),
        ],
        ps = [α => 2.0],
    ),
    (
        name = "two-field reaction–diffusion with cross-diffusion",
        eqs = two_field_eqs(),
        fields = [(u, ustar), (w, wstar)],
        # MethodOfLines' build time grows steeply with the grid (6 min at n = 16 and 20 at n = 32
        # for two fields), so this case stops at n = 16.
        ns = (4, 8, 16),
        traces = Ω -> begin
            Γl, Γr, Γb, Γt = boundarypiece.(Ref(Ω), (:left, :right, :bottom, :top))
            [
                u(t, Γl) ~ on(U, x, 0), u(t, Γr) ~ on(U, x, 1), u(t, Γb) ~ on(U, y, 0), u(t, Γt) ~ on(U, y, 1),
                w(t, Γl) ~ on(W, x, 0), NormalDerivative(Γr)(w(t, Γr)) ~ on(Wx, x, 1),
                w(t, Γb) ~ on(W, y, 0), w(t, Γt) ~ on(W, y, 1),
            ]
        end,
        endpoint = [
            u(t, 0, y) ~ on(U, x, 0), u(t, 1, y) ~ on(U, x, 1), u(t, x, 0) ~ on(U, y, 0), u(t, x, 1) ~ on(U, y, 1),
            w(t, 0, y) ~ on(W, x, 0), Dx(w(t, 1, y)) ~ on(Wx, x, 1), w(t, x, 0) ~ on(W, y, 0), w(t, x, 1) ~ on(W, y, 1),
        ],
        ps = [],
    ),
]

function system(case, Ω, bcs)
    domains = [(x, y) ∈ Ω, t ∈ Interval(TSPAN...)]
    ics = [f(0, x, y) ~ exact(0, x, y) for (f, exact) in case.fields]
    dvs = [f(t, x, y) for (f, _) in case.fields]
    return PDESystem(case.eqs, [bcs; ics], domains, [t, x, y], dvs, case.ps; name = :mol_comparison)
end

# MethodOfLines lowers trace conditions itself once MOLFiniteDifference is a PDEBase box
# discretization (SciML/MethodOfLines.jl#729); a release without it is given the lowered system.
const MOL_LOWERS = MOLFiniteDifference <: PDEBase.AbstractBoxDiscretization

"MethodOfLines nodal values at `TSPAN[2]` on the (n+1)×(n+1) grid, one matrix indexed [i_x, i_y] per field."
function mol_fields(case, n)
    sys = system(case, RECT, case.traces(RECT))
    MOL_LOWERS || (sys = PDEBase.to_box_bcs(sys))
    prob = discretize(sys, MOLFiniteDifference([x => 1 / n, y => 1 / n], t))
    sol = solve(prob, prob isa SciMLBase.DAEProblem ? EXACT_ROWS : FBDF(); saveat = collect(TSPAN), TOLS...)
    @test SciMLBase.successful_retcode(sol)
    @test sol[x] ≈ range(0, 1; length = n + 1) && sol[y] ≈ range(0, 1; length = n + 1)
    return [sol[f(t, x, y)][end, :, :] for (f, _) in case.fields]
end

"FEM nodal values at `TSPAN[2]`, rearranged onto the same grid as `mol_fields`."
function fem_fields(case, n)
    Ω = MeshDomain(rectangle_grid(Quadrilateral, (n, n)))
    sys = system(case, Ω, case.traces(Ω))
    sol = solve(discretize(sys, FEMDiscretization(Ω, t; saveat = [TSPAN[2]])), EXACT_ROWS; TOLS...)
    @test SciMLBase.successful_retcode(sol)
    return map(case.fields) do (f, _)
        field = fill(NaN, n + 1, n + 1)
        for (v, px, py) in zip(sol[f(t, x, y)][end, :], sol[x], sol[y])
            field[round(Int, px * n) + 1, round(Int, py * n) + 1] = v
        end
        field
    end
end

exact_field(exact, n) = [exact(TSPAN[2], (i - 1) / n, (j - 1) / n) for i in 1:(n + 1), j in 1:(n + 1)]
allnodes(n) = vec(CartesianIndices((n + 1, n + 1)))
noncorner(n) = [I for I in CartesianIndices((n + 1, n + 1)) if !(I[1] in (1, n + 1) && I[2] in (1, n + 1))]
maxdiff(a, b, nodes) = maximum(I -> abs(a[I] - b[I]), nodes)
# The nodes of the n grid inside the 2n grid
coarse(f2n) = f2n[1:2:end, 1:2:end]
rate(errs) = log2(errs[end - 1] / errs[end])

@testset verbose = true "MethodOfLines comparison" begin
    @testset "$(case.name)" for case in CASES
        @testset "to_box_bcs gives the hand-written endpoint form" begin
            lowered = PDEBase.to_box_bcs(system(case, RECT, case.traces(RECT)))
            @test all(isequal.(lowered.bcs, system(case, RECT, case.endpoint).bcs))
        end

        ns = case.ns
        mols = [mol_fields(case, n) for n in ns]
        fems = [fem_fields(case, n) for n in ns]
        @testset "field $k" for (k, (f, exact)) in enumerate(case.fields)
            mol, fem = getindex.(mols, k), getindex.(fems, k)
            mol_err = [maxdiff(m, exact_field(exact, n), noncorner(n)) for (m, n) in zip(mol, ns)]
            fem_err = [maxdiff(m, exact_field(exact, n), allnodes(n)) for (m, n) in zip(fem, ns)]
            # Richardson estimate of the rate-2 error at n from the n and 2n solutions
            estimate(fields, i) = 4 / 3 * maxdiff(fields[i], coarse(fields[i + 1]), noncorner(ns[i]))
            i = length(ns) - 1
            gap = maxdiff(mol[i], fem[i], noncorner(ns[i]))
            bound = estimate(mol, i) + estimate(fem, i)
            @info "MOL comparison: $(case.name), field $k" ns mol_err = string(mol_err) fem_err = string(fem_err) mol_rate = rate(mol_err) fem_rate = rate(fem_err) n = ns[i] gap bound

            @test rate(mol_err) >= MIN_RATE
            @test rate(fem_err) >= MIN_RATE
            @test gap <= bound
        end
    end
end
