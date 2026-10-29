# Benchmark (c), spec §11.5 and §7.2: time to construct the DAEProblem of a heat equation on P1 triangles at
# N = 1e4 unknowns, on the numeric path (`codegen = :numeric`, the default). The ModelingToolkit codegen path
# (`codegen = :mtk`) is measured only when the installed FiniteElementMethod accepts it; today it is reserved and
# raises, because of the upstream blockers of spec §7.7. No number is recorded for it in that case.
include(joinpath(@__DIR__, "common.jl"))
using FiniteElementMethod, ModelingToolkit, DomainSets, Printf
using SciMLBase: DAEProblem
using ModelingToolkit: t_nounits as t
using Ferrite: Triangle

const SIDES = parse.(Int, split(get(ENV, "BENCHMARK_CONSTRUCTION_SIDES", "50,100"), ","))   # nodes per side
const NRUNS = parse(Int, get(ENV, "BENCHMARK_CONSTRUCTION_RUNS", "3"))

# Upstream blockers gating `codegen = :mtk` as the default (spec §7.7 and Appendix B), with their tracking issues.
const BLOCKERS = [
    "ModelingToolkit #5121", "ModelingToolkit #5312", "ModelingToolkit #5327", "ModelingToolkit #5325",
    "ModelingToolkit #5328", "SciMLBase #1602",
]

@variables x y u(..)
const Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)

function heat_system(side)
    Ω = MeshDomain(FiniteElementMethod.rectangle_grid(Triangle, (side - 1, side - 1)))
    eq = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + (1 + x + 2y)
    bcs = [[u(t, Γ) ~ (1 + x + 2y) * (1 + t) for Γ in boundarypiece.(Ω, (:left, :right, :bottom, :top))]; u(0, x, y) ~ 1 + x + 2y]
    @named pdesys = PDESystem([eq], bcs, [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [u(t, x, y)], [])
    return pdesys, Ω
end

function mtk_status(Ω)
    return try
        FEMDiscretization(Ω, t; codegen = :mtk)
        (; available = true, message = "")
    catch err
        (; available = false, message = sprint(showerror, err))
    end
end

function run()
    rows = Dict{String, Any}[]
    mtk = nothing
    cold = nothing
    for side in SIDES
        pdesys, Ω = heat_system(side)
        disc = FEMDiscretization(Ω, t)
        first_call = @elapsed prob = DAEProblem(pdesys, disc)
        n = length(prob.u0)
        cold === nothing && (cold = Dict("unknowns" => n, "seconds" => first_call))
        warm = minimum(@elapsed(DAEProblem(pdesys, disc)) for _ in 1:NRUNS)
        mtk = mtk_status(Ω)
        row = Dict{String, Any}("unknowns" => n, "numeric_s" => warm)
        if mtk.available
            mdisc = FEMDiscretization(Ω, t; codegen = :mtk)
            row["mtk_s"] = minimum(@elapsed(DAEProblem(pdesys, mdisc)) for _ in 1:NRUNS)
        end
        push!(rows, row)
        @printf("N = %6d  numeric path: %.3f s (minimum of %d, after a first call of %.3f s)\n", n, warm, NRUNS, first_call)
    end
    mtk.available || println("codegen = :mtk not available: ", mtk.message)
    return write_results(
        "construction", Dict(
            "benchmark" => "(c) problem construction time, numeric path against ModelingToolkit codegen",
            "description" => "DAEProblem(pdesys, FEMDiscretization(Ω, t)) for a P1 heat equation with Dirichlet conditions, minimum of $NRUNS calls after a first call",
            "rows" => rows,
            "cold_first_call" => cold,
            "mtk_codegen" => Dict(
                "available" => mtk.available,
                "message" => mtk.message,
                "status" => mtk.available ? "measured" :
                    "not available today: codegen = :mtk is reserved until the upstream blockers of spec §7.7 land",
                "upstream_blockers" => BLOCKERS,
            ),
        )
    )
end

run()
