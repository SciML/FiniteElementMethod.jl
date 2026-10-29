# Spec §10.4 (linear M2 cases): each problem is written once on a DomainSets Rectangle with trace
# conditions (spec §3.3), discretized by MethodOfLines' MOLFiniteDifference on an n×n grid and by
# FEMDiscretization on a P1 quadrilateral mesh of the same box whose facet sets carry the same piece
# names, so the two share their nodes. Both are measured against the manufactured solution, and at
# equal n the two nodal fields must agree within the sum of their Richardson error estimates.
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
@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
const NS = (8, 16, 32)
const TSPAN = (0.0, 0.1)
const TOLS = (; abstol = 1.0e-10, reltol = 1.0e-10)
const MIN_RATE = 1.9

ustar(t_, x_, y_) = exp(-t_) * sin(π * x_) * cos(π * y_ / 2) + x_^2 * y_
const U = ustar(t, x, y)
const Ux, Uy = expand_derivatives(Dx(U)), expand_derivatives(Dy(U))
kcoef(x_, y_) = 1 + x_^2 + y_ / 2

# `g` on the face where `var` = `val`
on(g, var, val) = substitute(g, Dict(var => val))

const RECT = Rectangle(0.0 .. 1.0, 0.0 .. 1.0)

# Each case: the equation, its trace conditions on the pieces of a domain `Ω` (a Rectangle or a
# MeshDomain with the same piece names), the endpoint form MethodOfLines' tests would write, and
# the parameters.
function heat_eq(k)
    flux_div = Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y)))
    source = expand_derivatives(Dt(U) - Dx(k * Dx(U)) - Dy(k * Dy(U)))
    return Dt(u(t, x, y)) ~ flux_div + source
end

const CASES = [
    (
        name = "constant-coefficient heat, Neumann on the right",
        eq = heat_eq(1),
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
        eq = heat_eq(kcoef(x, y)),
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
        eq = heat_eq(1),
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
]

function system(case, Ω, bcs)
    domains = [(x, y) ∈ Ω, t ∈ Interval(TSPAN...)]
    return PDESystem([case.eq], [bcs; u(0, x, y) ~ ustar(0, x, y)], domains, [t, x, y], [u(t, x, y)], case.ps; name = :mol_comparison)
end

# MethodOfLines lowers trace conditions itself once MOLFiniteDifference is a PDEBase box
# discretization (SciML/MethodOfLines.jl#729); a release without it is given the lowered system.
const MOL_LOWERS = MOLFiniteDifference <: PDEBase.AbstractBoxDiscretization

"MethodOfLines nodal values at `TSPAN[2]` on the (n+1)×(n+1) grid, as a matrix indexed [i_x, i_y]."
function mol_field(case, n)
    sys = system(case, RECT, case.traces(RECT))
    MOL_LOWERS || (sys = PDEBase.to_box_bcs(sys))
    prob = discretize(sys, MOLFiniteDifference([x => 1 / n, y => 1 / n], t))
    sol = solve(prob, prob isa SciMLBase.DAEProblem ? EXACT_ROWS : FBDF(); saveat = collect(TSPAN), TOLS...)
    @test SciMLBase.successful_retcode(sol)
    @test sol[x] ≈ range(0, 1; length = n + 1) && sol[y] ≈ range(0, 1; length = n + 1)
    return sol[u(t, x, y)][end, :, :]
end

"FEM nodal values at `TSPAN[2]`, rearranged onto the same grid as `mol_field`."
function fem_field(case, n)
    Ω = MeshDomain(rectangle_grid(Quadrilateral, (n, n)))
    sys = system(case, Ω, case.traces(Ω))
    sol = solve(discretize(sys, FEMDiscretization(Ω, t; saveat = [TSPAN[2]])), EXACT_ROWS; TOLS...)
    @test SciMLBase.successful_retcode(sol)
    field = fill(NaN, n + 1, n + 1)
    for (v, px, py) in zip(sol[u(t, x, y)][end, :], sol[x], sol[y])
        field[round(Int, px * n) + 1, round(Int, py * n) + 1] = v
    end
    return field
end

exact_field(n) = [ustar(TSPAN[2], (i - 1) / n, (j - 1) / n) for i in 1:(n + 1), j in 1:(n + 1)]
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

        mol = [mol_field(case, n) for n in NS]
        fem = [fem_field(case, n) for n in NS]
        mol_err = [maxdiff(f, exact_field(n), noncorner(n)) for (f, n) in zip(mol, NS)]
        fem_err = [maxdiff(f, exact_field(n), allnodes(n)) for (f, n) in zip(fem, NS)]
        # Richardson estimate of the rate-2 error at n from the n and 2n solutions
        estimate(fields, i) = 4 / 3 * maxdiff(fields[i], coarse(fields[i + 1]), noncorner(NS[i]))
        i = length(NS) - 1
        gap = maxdiff(mol[i], fem[i], noncorner(NS[i]))
        bound = estimate(mol, i) + estimate(fem, i)
        @info "MOL comparison: $(case.name)" NS mol_err = string(mol_err) fem_err = string(fem_err) mol_rate = rate(mol_err) fem_rate = rate(fem_err) n = NS[i] gap bound

        @test rate(mol_err) >= MIN_RATE
        @test rate(fem_err) >= MIN_RATE
        @test gap <= bound
    end
end
