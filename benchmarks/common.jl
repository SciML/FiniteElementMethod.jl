# Shared by the benchmark scripts: timing, the environment record and the results file.
using Dates, Pkg, TOML

const RESULTS_DIR = get(ENV, "BENCHMARK_RESULTS_DIR", joinpath(@__DIR__, "results"))

"Minimum wall time of `f()` over `n` runs after one warm-up call."
best_time(f; n = 10) = (f(); minimum(@elapsed(f()) for _ in 1:n))

function git_sha()
    sha = try
        readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        "unknown"
    end
    # The package source and the benchmark scripts and environment; not the results this run writes.
    dirty = try
        !isempty(readchomp(`git -C $(@__DIR__) status --porcelain -- ../src . ':(exclude)results'`))
    catch
        false
    end
    return dirty ? sha * "-dirty" : sha
end

# A version string that pins what ran: a branch-tracked package also records its resolved git tree hash, and a
# path package (FiniteElementMethod itself) points at `git_sha`, which carries the dirty flag.
function package_version(info)
    version = string(something(info.version, "unversioned"))
    info.is_tracking_repo && return "$version ($(info.git_revision) at tree $(info.tree_hash))"
    info.is_tracking_path && return "$version (path, see git_sha)"
    return version
end

"Where and with what a result was produced. `BENCHMARK_ORIGIN` is set by the Benchmarks workflow."
function environment()
    packages = Dict(info.name => package_version(info) for info in values(Pkg.dependencies()) if info.is_direct_dep)
    return Dict(
        "origin" => get(ENV, "BENCHMARK_ORIGIN", "local run, not CI"),
        "date" => string(now(UTC)) * "Z",
        "git_sha" => git_sha(),
        "julia" => string(VERSION),
        "cpu" => strip(Sys.cpu_info()[1].model),
        "cpu_threads" => Sys.CPU_THREADS,
        "julia_threads" => Threads.nthreads(),
        "os" => string(Sys.KERNEL, " ", Sys.MACHINE),
        "packages" => packages,
    )
end

"Write `results` (a Dict) with the environment record to `RESULTS_DIR/name.toml`; return the path."
function write_results(name, results)
    mkpath(RESULTS_DIR)
    path = joinpath(RESULTS_DIR, name * ".toml")
    open(path, "w") do io
        TOML.print(io, merge(Dict("environment" => environment()), results); sorted = true)
    end
    println("wrote ", path)
    return path
end
