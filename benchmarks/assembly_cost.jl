# Benchmark (b), spec §11.5 and §6.6: assembly and residual cost of the generated pipeline (weak_form,
# emit_kernels, in-place assembly) against hand-written Ferrite loops, on 100×100 P1 triangles, with the Perf
# group's method: both sides in place into preallocated operators, the minimum of N runs after a warm-up.
#
# The driver internals change between releases, so the script uses only what every version since M1 has
# (`weak_form`, `emit_kernels`, `AssemblyContext`, `assemble_linear!`, `assemble_nonlinear!`) and records which
# later features it found. The strong-form analysis is built by hand, as the precompile workload does, so the
# benchmark does not depend on the PDEBase front end.
include(joinpath(@__DIR__, "common.jl"))
using FiniteElementMethod, Ferrite, LinearAlgebra, Printf, Symbolics
using FiniteElementMethod: weak_form, emit_kernels, AssemblyContext, assemble_linear, assemble_linear!, assemble_nonlinear!

const NEL = parse(Int, get(ENV, "BENCHMARK_ASSEMBLY_NEL", "100"))
const NRUNS = parse(Int, get(ENV, "BENCHMARK_RUNS", "10"))
const P_HEAT = [2.0, 3.0, 0.5, 1.5]   # k, α, β, h

variables(name, n) = [Symbolics.variable(name, i) for i in 1:n]
function slots()
    U, G, Ud, X, N, P = variables(:U, 1), variables(:G, 2), variables(:Ud, 1), variables(:X, 2), variables(:N, 2), variables(:P, 4)
    return (; U, G, Ud, H = nothing, X, T = Symbolics.variable(:T), N, P)
end

# Case (i): ∂ₜu = ∇·(k∇u) + xy + 1, ∂ₙu = h on `right` (flux k·h), k∂ₙu + αu = β on `top`.
function heat_kernels(; facets)
    s = slots()
    (; U, G, Ud, X, P) = s
    natural = facets ? [(; piece = "right", eq = 1, Φ = -P[1] * P[4]), (; piece = "top", eq = 1, Φ = P[2] * U[1] - P[3])] : []
    analysis = (;
        slots = s, nfields = 1, dim = 2, mass = [Ud[1]], flux = [[-P[1] * G[1], -P[1] * G[2]]],
        remainder = [-X[1] * X[2] - 1], natural,
    )
    return emit_kernels(weak_form(analysis))
end

# Case (iii): 0 = ∇·((1 + u²)∇u) + xy + 1 with ∂ₙu = y + 1 on `right`.
function nonlinear_kernels()
    s = slots()
    (; U, G, X) = s
    κ = 1 + U[1]^2
    analysis = (;
        slots = s, nfields = 1, dim = 2, mass = [Num(0)], flux = [[-κ * G[1], -κ * G[2]]],
        remainder = [-X[1] * X[2] - 1], natural = [(; piece = "right", eq = 1, Φ = -κ * (X[2] + 1))],
    )
    return emit_kernels(weak_form(analysis))
end

function setup(nel)
    grid = generate_grid(Triangle, (nel, nel))
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    return (; grid, dh, cv, fv, ctx = AssemblyContext(dh, [:u], cv, fv, Dict("right" => "right", "top" => "top")))
end

# Hand-written case (i), ported from the Perf group (test/Perf/perf.jl `heat_hand!`).
function heat_hand!(K, M, F, s, p; facets)
    (; dh, cv, fv, grid) = s
    kk, a, b, hh = p
    n = getnbasefunctions(cv)
    Ke, Me, Fe = zeros(n, n), zeros(n, n), zeros(n)
    aK, aM = start_assemble(K, F), start_assemble(M)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Ke, 0); fill!(Me, 0); fill!(Fe, 0)
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            for i in 1:n
                v, ∇v = shape_value(cv, q, i), shape_gradient(cv, q, i)
                Fe[i] += (X[1] * X[2] + 1) * v * dΩ
                for j in 1:n
                    Ke[i, j] += kk * (∇v ⋅ shape_gradient(cv, q, j)) * dΩ
                    Me[i, j] += v * shape_value(cv, q, j) * dΩ
                end
            end
        end
        assemble!(aK, celldofs(cell), Ke, Fe)
        assemble!(aM, celldofs(cell), Me)
    end
    facets || return nothing
    aR = start_assemble(K, F; fillzero = false)
    for (set, robin, g) in (("right", 0.0, kk * hh), ("top", a, b))
        for fc in FacetIterator(dh, getfacetset(grid, set))
            Ferrite.reinit!(fv, fc)
            fill!(Ke, 0); fill!(Fe, 0)
            for q in 1:getnquadpoints(fv), i in 1:n
                dΓ = getdetJdV(fv, q)
                v = shape_value(fv, q, i)
                Fe[i] += g * v * dΓ
                for j in 1:n
                    Ke[i, j] += robin * v * shape_value(fv, q, j) * dΓ
                end
            end
            assemble!(aR, celldofs(fc), Ke, Fe)
        end
    end
    return nothing
end

# Hand-written case (iii) residual and Jacobian (test/Perf/perf.jl `nonlinear_hand!`).
function nonlinear_hand!(r, J, s, u)
    (; dh, cv, fv, grid) = s
    n = getnbasefunctions(cv)
    Je, re, u_elem = zeros(n, n), zeros(n), zeros(n)
    aJ = start_assemble(J, r)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Je, 0); fill!(re, 0)
        for (a, d) in enumerate(celldofs(cell))
            u_elem[a] = u[d]
        end
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            uq, gq = function_value(cv, q, u_elem), function_gradient(cv, q, u_elem)
            κ, dκ = 1 + uq^2, 2uq
            for i in 1:n
                v, ∇v = shape_value(cv, q, i), shape_gradient(cv, q, i)
                re[i] += (κ * (gq ⋅ ∇v) - (X[1] * X[2] + 1) * v) * dΩ
                for j in 1:n
                    w, ∇w = shape_value(cv, q, j), shape_gradient(cv, q, j)
                    Je[i, j] += (dκ * w * (gq ⋅ ∇v) + κ * (∇w ⋅ ∇v)) * dΩ
                end
            end
        end
        assemble!(aJ, celldofs(cell), Je, re)
    end
    for fc in FacetIterator(dh, getfacetset(grid, "right"))
        Ferrite.reinit!(fv, fc)
        fill!(Je, 0); fill!(re, 0)
        for (a, d) in enumerate(celldofs(fc))
            u_elem[a] = u[d]
        end
        xs = getcoordinates(fc)
        for q in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q)
            g = spatial_coordinate(fv, q, xs)[2] + 1
            uq = function_value(fv, q, u_elem)
            for i in 1:n
                v = shape_value(fv, q, i)
                re[i] -= (1 + uq^2) * g * v * dΓ
                for j in 1:n
                    Je[i, j] -= 2uq * shape_value(fv, q, j) * g * v * dΓ
                end
            end
        end
        assemble!(aJ, celldofs(fc), Je, re)
    end
    return nothing
end

relerr(A, B) = norm(A - B) / norm(B)
operators(s) = (K = allocate_matrix(s.dh); (copy(K), K, zeros(ndofs(s.dh))))

function run()
    s = setup(NEL)
    features = Dict(
        name => isdefined(FiniteElementMethod, name)
            for name in (:NonlinearOperator, :LinearBlock, :StructuralZero, :LinearCache)
    )
    rows = Dict{String, Any}[]
    for facets in (false, true)
        kernels = heat_kernels(; facets)
        Mh, Kh, Fh = operators(s)
        Mg, Kg, Fg = operators(s)
        th = best_time(() -> heat_hand!(Kh, Mh, Fh, s, P_HEAT; facets); n = NRUNS)
        tg = best_time(() -> assemble_linear!(Mg, Kg, Fg, s.ctx, kernels, P_HEAT); n = NRUNS)
        err = maximum(relerr.((Kg, Mg, Fg), (Kh, Mh, Fh)))
        case = facets ? "case (i), M, K, F with Neumann and Robin facets" : "case (i), M, K, F, volume terms"
        push!(rows, Dict("case" => case, "hand_ms" => 1.0e3th, "generated_ms" => 1.0e3tg, "ratio" => tg / th, "max_relerr" => err))
    end
    kernels = nonlinear_kernels()
    L = assemble_linear(s.ctx, kernels, Float64[])
    u = randn(ndofs(s.dh))
    rh, Jh = zeros(ndofs(s.dh)), allocate_matrix(s.dh)
    rg, Jg = zeros(ndofs(s.dh)), allocate_matrix(s.dh)
    # Per Newton step the generated path assembles the nonlinear kernels and adds the constant K and F.
    function generated!()
        assemble_nonlinear!(rg, Jg, s.ctx, kernels, u, Float64[])
        mul!(rg, L.K, u, true, true)
        rg .-= L.F
        Jg.nzval .+= L.K.nzval
        return nothing
    end
    th = best_time(() -> nonlinear_hand!(rh, Jh, s, u); n = NRUNS)
    tg = best_time(generated!; n = NRUNS)
    err = max(relerr(rg, rh), relerr(Jg, Jh))
    push!(rows, Dict("case" => "case (iii), residual and Jacobian", "hand_ms" => 1.0e3th, "generated_ms" => 1.0e3tg, "ratio" => tg / th, "max_relerr" => err))
    for row in rows
        @printf("%-50s hand %7.2f ms  generated %7.2f ms  ratio %.2f  (relerr %.1e)\n", row["case"], row["hand_ms"], row["generated_ms"], row["ratio"], row["max_relerr"])
    end
    return write_results(
        "assembly_cost", Dict(
            "benchmark" => "(b) assembly and residual cost against hand-written Ferrite",
            "description" => "$(NEL)×$(NEL) P1 triangles, in place, minimum of $NRUNS runs after warm-up",
            "features" => features,
            "rows" => rows,
        )
    )
end

run()
