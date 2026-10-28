# Performance and allocation gates of spec §6.6 items 1, 3 and 4 and §10.6: the generated pipeline (seam,
# weak_form, emit_kernels, in-place assembly) against hand-written Ferrite loops on 100×100 P1 triangles.
# Timings are the minimum of several runs after warm-up, and the gates are ratios, so a busy box slows both sides.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))
using AllocCheck: check_allocs
using FiniteElementMethod: NonlinearOperator, residual!, assemble_linear!, assemble_nonlinear!, LinearOp, NonlinearOp, _split, _kernels!,
    _facet_groups!, _start_cell!

const TIME_GATE = 3.0     # spec §6.6 item 1: within 3× of the hand-written loop
const NRUNS = 10
# Coverage instrumentation (Pkg.test(; coverage=true) in CI) slows the generated kernels about 15× and the hand-written
# loop about 6×, so the ratio inflates (1.4 → 3.7 on the same box); the time gates only mean something uninstrumented.
const INSTRUMENTED = Base.JLOptions().code_coverage != 0

best_time(f) = (f(); minimum(@elapsed(f()) for _ in 1:NRUNS))

@variables t x y k α β h
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

# Case (i): ∂ₜu = ∇·(k∇u) + xy + 1, optionally ∂ₙu = h on `right` and k∂ₙu + αu = β on `top`.
const PV = [2.0, 3.0, 0.5, 1.5]
function heat_kernels(; facets)
    bcs = facets ? [BC("right", Dx(U) ~ h, RIGHT), BC("top", k * Dy(U) + α * U ~ β, [0.0, 1.0])] : BC[]
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    return chain([eq], bcs; depvars = [U], t, sivs = [x, y], params = [k, α, β, h]).kernels
end

# Hand-written case (i) into preallocated K, M, F (scratch/dauntless/test_misc.jl `handwritten`, plus the
# Neumann and Robin facet loops of test_linear.jl `reference`).
function heat_hand!(K, M, F, setup, p; facets)
    (; dh, cv, fv, grid) = setup
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

operators(setup) = (K = allocate_matrix(setup.dh); (copy(K), K, zeros(ndofs(setup.dh))))

# Case (iii): 0 = ∇·((1 + u²)∇u) + xy + 1 with ∂ₙu = y + 1 on `right`. Per Newton step the generated path
# assembles the nonlinear kernels and adds the constant K and F; the hand-written loop assembles everything.
nl_source(X) = X[1] * X[2] + 1
nl_flux(X) = X[2] + 1
function nonlinear_kernels()
    eq = 0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + x * y + 1
    return chain([eq], [BC("right", Dx(U) ~ y + 1, RIGHT)]; depvars = [U], t, sivs = [x, y]).kernels
end

function nonlinear_generated!(r, J, setup, kernels, u, K, F)
    assemble_nonlinear!(r, J, setup.ctx, kernels, u, P0)
    mul!(r, K, u, true, true)
    r .-= F
    J.nzval .+= K.nzval     # one shared pattern (allocate_matrix(dh)), checked below
    return nothing
end

function nonlinear_hand!(r, J, setup, u)
    (; dh, cv, fv, grid) = setup
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
                re[i] += (κ * (gq ⋅ ∇v) - nl_source(X) * v) * dΩ
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
            g = nl_flux(spatial_coordinate(fv, q, xs))
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

const SETUP = mesh_setup(Triangle, 1, 100)
const SMALL = mesh_setup(Triangle, 1, 20)

@testset "gate 1: case (i) assembly within $(TIME_GATE)× of hand-written" begin
    for facets in (false, true)
        kernels = heat_kernels(; facets)
        Mh, Kh, Fh = operators(SETUP)
        Mg, Kg, Fg = operators(SETUP)
        hand() = heat_hand!(Kh, Mh, Fh, SETUP, PV; facets)
        generated() = assemble_linear!(Mg, Kg, Fg, SETUP.ctx, kernels, PV)
        th, tg = best_time(hand), best_time(generated)
        @info "case (i) on 100×100 P1" facets hand_ms = 1.0e3th generated_ms = 1.0e3tg ratio = tg / th
        check_operators((; K = Kg, M = Mg, F = Fg), Kh, Mh, Fh)
        @test tg / th ≤ TIME_GATE skip = INSTRUMENTED
    end
end

@testset "gate 3: case (iii) residual and Jacobian within $(TIME_GATE)× of hand-written" begin
    kernels = nonlinear_kernels()
    L = assemble_linear(SETUP.ctx, kernels, P0)
    u = randn(Xoshiro(1), ndofs(SETUP.dh))
    rh, Jh = zeros(ndofs(SETUP.dh)), allocate_matrix(SETUP.dh)
    rg, Jg = zeros(ndofs(SETUP.dh)), allocate_matrix(SETUP.dh)
    hand() = nonlinear_hand!(rh, Jh, SETUP, u)
    generated() = nonlinear_generated!(rg, Jg, SETUP, kernels, u, L.K, L.F)
    @test Jg.rowval == L.K.rowval && Jg.colptr == L.K.colptr
    th, tg = best_time(hand), best_time(generated)
    @info "case (iii) r and J on 100×100 P1" hand_ms = 1.0e3th generated_ms = 1.0e3tg ratio = tg / th
    @test relerr(rg, rh) ≤ 1.0e-13
    @test relerr(Jg, Jh) ≤ 1.0e-13
    @test tg / th ≤ TIME_GATE skip = INSTRUMENTED
end

@testset "gate 2: no per-cell allocation" begin
    linear, nonlinear = heat_kernels(; facets = true), nonlinear_kernels()
    function pass_allocations(setup)
        M, K, F = operators(setup)
        r, J, u = zeros(ndofs(setup.dh)), allocate_matrix(setup.dh), rand(ndofs(setup.dh))
        assemble_linear!(M, K, F, setup.ctx, linear, PV)
        assemble_nonlinear!(r, J, setup.ctx, nonlinear, u, P0)
        return (
            @allocated(assemble_linear!(M, K, F, setup.ctx, linear, PV)),
            @allocated(assemble_nonlinear!(r, J, setup.ctx, nonlinear, u, P0)),
        )
    end
    small, large = pass_allocations(SMALL), pass_allocations(SETUP)
    @info "allocations per pass (bytes)" linear_20 = small[1] linear_100 = large[1] nonlinear_20 = small[2] nonlinear_100 = large[2]
    # 25× the cells, the same per-pass setup cost: kernel splitting, scratch, assemblers, the cell cache
    @test all(large .≤ small .+ 1024)
    @test all(large .≤ 32 * 1024)

    # The per-cell work, statically: element kernels, facet groups and the element gather. The scatter
    # (Ferrite.assemble!) is left to the scaling test above: its sort buffers may `resize!`, which AllocCheck
    # reports although they only grow on the first cell.
    ctx = SETUP.ctx
    M, K, F = operators(SETUP)
    ne = getnbasefunctions(ctx.cellvalues.u)
    xs = getcoordinates(first(CellIterator(SETUP.dh)))
    ranges = (dof_range(SETUP.dh, :u),)
    lop = LinearOp(start_assemble(M), start_assemble(K), F, zeros(ne, ne), zeros(ne, ne), zeros(ne))
    n = ndofs(SETUP.dh)
    nop = NonlinearOp(start_assemble(allocate_matrix(SETUP.dh)), zeros(n), zeros(ne, ne), zeros(ne), rand(n), zeros(ne))
    rop = NonlinearOp(nothing, zeros(n), zeros(ne, ne), zeros(ne), rand(n), zeros(ne))   # residual only
    cases = ((lop, linear, (:mass, :bilinear, :source)), (nop, nonlinear, (:nonlinear,)), (rop, nonlinear, (:nonlinear,)))
    for (op, kernels, kinds) in cases
        volume, groups = _split(ctx, kernels, kinds)
        counts = zeros(Int, length(groups))
        fi = FacetIndex(1, 1)
        cvs, fvs = values(ctx.cellvalues), values(ctx.facetvalues)
        @test isempty(check_allocs(_kernels!, map(typeof, (op, volume, cvs, xs, ranges, PV, 0.0))))
        @test isempty(check_allocs(_facet_groups!, map(typeof, (op, groups, counts, 1, fi, fvs, fvs, xs, ranges, PV, 0.0))))
        @test isempty(check_allocs(_start_cell!, map(typeof, (op, celldofs(first(CellIterator(SETUP.dh)))))))
    end
end

# Spec §6.6 item 2: the coefficient contraction (`coeffs`) must not be slower than the per-shape-function `f` path
# where it is meant to pay, P2 elements, a variable coefficient and several fields. It is fragile in one way: if
# the contraction's static-index helpers stop inlining, it becomes over 10× slower (seen at two fields).
const CONTRACTION_GATE = 1.0
@testset "gate 4: coefficient contraction no slower than the f path at P2" begin
    @variables Du Dw χ a b
    @variables w(..)
    Wf = w(t, x, y)
    axy = 1 + x^2 + y
    iib = weak_form(
        analyse(
            [Dt(U) ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) - 0.7 * Dx(U) - 0.3 * Dy(U) - 2.5 * U + x * y + 1],
            [BC("right", Dx(U) ~ h, RIGHT), BC("top", axy * Dy(U) + α * U ~ β, [0.0, 1.0])];
            depvars = [U], t, sivs = [x, y], params = [k, α, β, h]
        )
    )
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a - U,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U,
    ]
    two = weak_form(analyse(eqs, [BC("right", Dx(Wf) ~ 0.2, RIGHT)]; depvars = [U, Wf], t, sivs = [x, y], params = [Du, Dw, χ, a, b]))
    for (name, wf, fields, p) in (("(ii-b)", iib, [:u], PV), ("two-field", two, [:u, :w], [1.0, 20.0, 0.3, 0.1, 0.9]))
        setup = mesh_setup(Triangle, 2, 30; fields)
        fused, plain = emit_kernels(wf), emit_kernels(wf; coeffs = false)
        Mf, Kf, Ff = operators(setup)
        Mp, Kp, Fp = operators(setup)
        tf = best_time(() -> assemble_linear!(Mf, Kf, Ff, setup.ctx, fused, p))
        tp = best_time(() -> assemble_linear!(Mp, Kp, Fp, setup.ctx, plain, p))
        @info "$name P2 on 30×30" coeffs_ms = 1.0e3tf f_ms = 1.0e3tp ratio = tf / tp
        @test tf / tp ≤ CONTRACTION_GATE
    end
end

# Kernels no other testset generates, so their first calls compile the driver for new kernel types.
function first_call_kernels()
    linear = chain(
        [Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x^2 + 2], BC[]; depvars = [U], t, sivs = [x, y], params = [k, α, β, h]
    ).kernels
    nonlinear = chain(
        [0 ~ Dx((2 + U^2) * Dx(U)) + Dy((2 + U^2) * Dy(U)) + 1], BC[]; depvars = [U], t, sivs = [x, y]
    ).kernels
    return linear, nonlinear
end

@testset "first call against warm time, assemble_linear! and residual!" begin
    linear, nonlinear = first_call_kernels()
    M, K, F = operators(SETUP)
    op = NonlinearOperator(SETUP.ctx, nonlinear)
    u, r = rand(Xoshiro(2), ndofs(SETUP.dh)), zeros(ndofs(SETUP.dh))
    assemble() = assemble_linear!(M, K, F, SETUP.ctx, linear, PV)
    residual() = residual!(r, op, u, P0, 0.0)
    first_linear, first_residual = @elapsed(assemble()), @elapsed(residual())
    warm_linear, warm_residual = best_time(assemble), best_time(residual)
    @info "first call against warm on 100×100 P1 (s)" first_linear warm_linear first_residual warm_residual
    @test first_linear ≤ 60.0 && first_residual ≤ 60.0      # loose: catches a codegen blow-up, not drift
end

# Time to first assembly in a fresh process, after `using FiniteElementMethod`, for kernels the precompile
# workload did not generate (`workload_chain` with another `k` on another grid). Both modes set PrecompileTools'
# `precompile_workload` preference explicitly in a temporary environment placed FIRST on the load path: preferences
# from earlier environments take precedence, so a value in the active project's LocalPreferences.toml cannot
# override either mode. The two values precompile separate caches.
function time_to_first_assembly(; workload::Bool)
    script = """
    using FiniteElementMethod
    print(@elapsed FiniteElementMethod.workload_chain(3.7; nel = 4))
    """
    return mktempdir() do dir
        write(joinpath(dir, "Project.toml"), "[extras]\nFiniteElementMethod = \"e08948db-0ae8-498a-9b98-73fc7f237ce3\"\n")
        write(joinpath(dir, "LocalPreferences.toml"), "[FiniteElementMethod]\nprecompile_workload = $workload\n")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
        return parse(Float64, readchomp(addenv(cmd, "JULIA_LOAD_PATH" => "$dir:@:@stdlib")))
    end
end

@testset "cold latency: time to first assembly with and without the precompile workload" begin
    with, without = time_to_first_assembly(; workload = true), time_to_first_assembly(; workload = false)
    @info "time to first assembly in a fresh process (s)" with without ratio = with / without
    @test with ≤ 0.5 * without        # measured 0.51 s against 14.4 s (0.04); slack for a busy box
    @test with ≤ 10.0
end
