# Runs every benchmark in its own process, so each starts cold: `julia --project=benchmarks benchmarks/run_all.jl`.
const SCRIPTS = ("accuracy_convdiff.jl", "assembly_cost.jl", "construction.jl")
for script in SCRIPTS
    println("== ", script)
    run(`$(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, script))`)
end
