# Renders the benchmark results files as the Markdown page docs/src/benchmarks.md. Used by docs/make.jl at build
# time and runnable alone: `julia benchmarks/render.jl [results dir] [output file]`. Only the TOML standard library.
using TOML, Printf

fmt(x::AbstractFloat; digits = 3) = isnan(x) ? "–" : @sprintf("%.*g", digits, x)
fmt(x; digits = 3) = string(x)
ms(x) = @sprintf("%.2f", x)

const SHOWN_PACKAGES = ("FiniteElementMethod", "Ferrite", "MethodOfLines", "ModelingToolkit", "PDEBase", "OrdinaryDiffEqBDF")

function provenance(env)
    pkgs = env["packages"]
    shown = join(("$name $(pkgs[name])" for name in sort(collect(keys(pkgs))) if name in SHOWN_PACKAGES), ", ")
    return "*Produced by: $(env["origin"]); $(env["date"]); commit `$(env["git_sha"])`; $(env["cpu"]), " *
        "$(env["cpu_threads"]) hardware threads, Julia $(env["julia"]) with $(env["julia_threads"]) thread(s); $shown.*"
end

function table(header, rows)
    lines = ["| " * join(header, " | ") * " |", "|" * repeat("---|", length(header))]
    append!(lines, ("| " * join(row, " | ") * " |" for row in rows))
    return join(lines, "\n")
end

function render_accuracy(r)
    rows = [
        (
            row["cells"], row["fem_ndofs"], fmt(row["fem_error"]), fmt(row["fem_rate"]),
            row["mol_ndofs"], fmt(row["mol_error"]), fmt(row["mol_rate"]),
        ) for row in r["rows"]
    ]
    return """
    ## (a) Accuracy per degree of freedom: `convdiff2` against MethodOfLines

    $(r["description"]). FEM solver: $(r["fem_solver"]). MethodOfLines solver: $(r["mol_solver"]). The FEM column is Galerkin P1, which is centred and second order;
    MethodOfLines' default upwinds the advection term, which is first order. On these meshes `convdiff2` has a cell
    Péclet number below one (Tier 1 of the supported-classes table); advection-dominated problems stay Tier 2 until
    SUPG.

    $(provenance(r["environment"]))

    $(table(["cells", "FEM dofs", "FEM max error", "FEM rate", "MOL points", "MOL max error", "MOL rate"], rows))

    The same numbers, for plotting error against degrees of freedom, are in `benchmarks/results/accuracy_convdiff.toml`.
    """
end

function render_assembly(r)
    rows = [
        (row["case"], ms(row["hand_ms"]), ms(row["generated_ms"]), @sprintf("%.2f×", row["ratio"]), fmt(row["max_relerr"]; digits = 2))
            for row in r["rows"]
    ]
    found = sort([name for (name, present) in r["features"] if present])
    features = isempty(found) ? "none of the later driver optimisations" : join(found, ", ")
    return """
    ## (b) Assembly and residual cost against hand-written Ferrite

    $(r["description"]). The generated pipeline is `weak_form`, `emit_kernels` and in-place assembly; the
    hand-written loops are the Perf test group's references. Driver features present: $features.

    $(provenance(r["environment"]))

    $(table(["case", "hand-written (ms)", "generated (ms)", "generated / hand-written", "max relative difference"], rows))
    """
end

function render_construction(r)
    m = r["mtk_codegen"]
    rows = [
        (row["unknowns"], @sprintf("%.3f", row["numeric_s"]), m["available"] ? @sprintf("%.3f", row["mtk_s"]) : "not available")
            for row in r["rows"]
    ]
    cold = r["cold_first_call"]
    mtk_text = m["available"] ? "" :
        "The ModelingToolkit codegen path (`codegen = :mtk`) is $(m["status"]); it reports `$(m["message"])`. " *
        "The blockers are tracked upstream as $(join(m["upstream_blockers"], ", ")). This table will show the " *
        "comparison once they land and the switch is enabled; until then no codegen timing is shown."
    return """
    ## (c) Problem construction time

    $(r["description"]). The first call in the process, which includes compilation, took
    $(@sprintf("%.1f", cold["seconds"])) s at N = $(cold["unknowns"]).

    $(provenance(r["environment"]))

    $(table(["unknowns N", "numeric path (s)", "ModelingToolkit codegen (s)"], rows))

    $mtk_text
    """
end

const RENDERERS = [
    ("accuracy_convdiff", render_accuracy), ("assembly_cost", render_assembly), ("construction", render_construction),
]

function render(results_dir)
    parts = String[
        """
        # Benchmarks

        These tables are generated from the files in `benchmarks/results/` when the documentation is built; nothing
        on this page is typed by hand. Each table states where, when and with which package versions it was
        produced. The scripts are in `benchmarks/`, whose README says how to run them.
        """,
    ]
    for (name, renderer) in RENDERERS
        path = joinpath(results_dir, name * ".toml")
        push!(parts, isfile(path) ? renderer(TOML.parsefile(path)) : "## $name\n\nNo results yet: run `benchmarks/$name.jl` or the Benchmarks workflow.\n")
    end
    return join(parts, "\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = get(ARGS, 1, joinpath(@__DIR__, "results"))
    out = get(ARGS, 2, joinpath(@__DIR__, "..", "docs", "src", "benchmarks.md"))
    write(out, render(results_dir))
    println("wrote ", out)
end
