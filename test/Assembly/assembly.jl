# Generic Ferrite assembly driver (spec §6.5, §6.6, §10.1) against hand-written Ferrite loops.
# Kernels are built by hand with the contract signature f(U, G, Ud, V, W, X, T, N, P).
using FiniteElementMethod, Ferrite, LinearAlgebra, Random, Test
using FiniteElementMethod: AssemblyContext, KernelSpec, assemble_linear, assemble_nonlinear

relerr(A, B) = norm(A - B) / norm(B)
# The DofHandler and the one CellValues/FacetValues pair all fields of `ctx` share.
shared_values(ctx) = (; ctx.dh, cv = first(ctx.cellvalues), fv = first(ctx.facetvalues))

function heat_context(n; neumann = "right", robin = "top", nfields = 1)
    grid = generate_grid(Triangle, (n, n))
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    for f in [:u, :w][1:nfields]
        add!(dh, f, ip)
    end
    close!(dh)
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    return AssemblyContext(dh, [:u, :w][1:nfields], cv, fv, Dict(:neumann => neumann, :robin => robin))
end

# Case (i): ∂ₜu = ∇·(k∇u) + xy + 1, k∂ₙu = k h on `right` (user wrote ∂ₙu = h),
# k∂ₙu + αu = β on `top`. P = [k, α, β, h]. Residual convention M u̇ + K u − F = 0.
const P_HEAT = [2.0, 3.0, 0.5, 1.5]
heat_source(X) = X[1] * X[2] + 1
function heat_kernels()
    return (
        KernelSpec(1, :mass, nothing, (U, G, Ud, V, W, X, T, N, P) -> Ud[1] * V, nothing),
        KernelSpec(1, :bilinear, nothing, (U, G, Ud, V, W, X, T, N, P) -> P[1] * (G[1] * W[1] + G[2] * W[2]), nothing),
        KernelSpec(1, :source, nothing, (U, G, Ud, V, W, X, T, N, P) -> -heat_source(X) * V, nothing),
        KernelSpec(1, :source, :neumann, (U, G, Ud, V, W, X, T, N, P) -> -P[1] * P[4] * V, nothing),
        KernelSpec(1, :bilinear, :robin, (U, G, Ud, V, W, X, T, N, P) -> P[2] * U[1] * V, nothing),
        KernelSpec(1, :source, :robin, (U, G, Ud, V, W, X, T, N, P) -> -P[3] * V, nothing),
    )
end

"Hand-written case (i) assembly, ported from scratch/dauntless/test_linear.jl `reference`."
function heat_reference(ctx, p)
    (; dh, cv, fv) = shared_values(ctx)
    grid = dh.grid
    k, α, β, h = p
    K, M = allocate_matrix(dh), allocate_matrix(dh)
    F = zeros(ndofs(dh))
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
                Fe[i] += heat_source(X) * v * dΩ
                for j in 1:n
                    Ke[i, j] += k * (∇v ⋅ shape_gradient(cv, q, j)) * dΩ
                    Me[i, j] += v * shape_value(cv, q, j) * dΩ
                end
            end
        end
        assemble!(aK, celldofs(cell), Ke, Fe)
        assemble!(aM, celldofs(cell), Me)
    end
    for fc in FacetIterator(dh, getfacetset(grid, "right"))
        Ferrite.reinit!(fv, fc)
        fill!(Fe, 0)
        for q in 1:getnquadpoints(fv), i in 1:n
            Fe[i] += k * h * shape_value(fv, q, i) * getdetJdV(fv, q)
        end
        assemble!(F, celldofs(fc), Fe)
    end
    aR = start_assemble(K, F; fillzero = false)
    for fc in FacetIterator(dh, getfacetset(grid, "top"))
        Ferrite.reinit!(fv, fc)
        fill!(Ke, 0); fill!(Fe, 0)
        for q in 1:getnquadpoints(fv), i in 1:n
            dΓ = getdetJdV(fv, q)
            Fe[i] += β * shape_value(fv, q, i) * dΓ
            for j in 1:n
                Ke[i, j] += α * shape_value(fv, q, i) * shape_value(fv, q, j) * dΓ
            end
        end
        assemble!(aR, celldofs(fc), Ke, Fe)
    end
    return K, M, F
end

@testset "case (i): K, M, F vs hand-written Ferrite" begin
    ctx = heat_context(20)
    A = assemble_linear(ctx, heat_kernels(), P_HEAT)
    Kr, Mr, Fr = heat_reference(ctx, P_HEAT)
    @test relerr(A.K, Kr) ≤ 1.0e-14
    @test relerr(A.M, Mr) ≤ 1.0e-14
    @test relerr(A.F, Fr) ≤ 1.0e-14
    @test A.M.rowval == A.K.rowval && A.M.colptr == A.K.colptr
end

@testset "facet terms are non-vanishing: sum(F) shifts by k·h·|Γ|" begin
    ctx = heat_context(20)
    with_h = assemble_linear(ctx, heat_kernels(), P_HEAT).F
    without_h = assemble_linear(ctx, heat_kernels(), [P_HEAT[1:3]; 0.0]).F
    k, h, Γ = P_HEAT[1], P_HEAT[4], 2.0   # `right` of generate_grid's [−1, 1]² has length 2
    @test sum(with_h) - sum(without_h) ≈ k * h * Γ rtol = 1.0e-13
end

@testset "facet-count assertion" begin
    ks = heat_kernels()
    bogus = Set([FacetIndex(1, 1), FacetIndex(10^6, 1)])   # no cell 10^6; triangles have no facet 4
    @test_throws ArgumentError(
        "piece neumann: assembled 1 facets, but its facet set has 2; the set must hold FacetIndex entries of this grid"
    ) assemble_linear(heat_context(4; neumann = bogus), ks, P_HEAT)
    @test_throws ArgumentError(
        "piece robin: assembled 0 facets, but its facet set has 1; the set must hold FacetIndex entries of this grid"
    ) assemble_linear(heat_context(4; robin = Set([FacetIndex(1, 4)])), ks, P_HEAT)
    @test_throws ArgumentError("piece neumann: the grid has no facet set named \"nonexistent\"") heat_context(4; neumann = "nonexistent")
    unknown = (KernelSpec(1, :source, :elsewhere, (U, G, Ud, V, W, X, T, N, P) -> V, nothing),)
    @test_throws ArgumentError("no facet set is given for piece elsewhere: add it to the pieces of the AssemblyContext") assemble_linear(heat_context(4), unknown, P_HEAT)
end

@testset "malformed kernels are rejected with ArgumentErrors" begin
    f(U, G, Ud, V, W, X, T, N, P) = V
    reject(kernel) = assemble_linear(heat_context(4), (kernel,), P_HEAT)
    @test_throws ArgumentError("kernel kind :stiffness is not one of :mass, :bilinear, :source, :nonlinear") reject(KernelSpec(1, :stiffness, nothing, f, nothing))
    @test_throws ArgumentError("kernel equation 2 is outside 1:1, the fields of the AssemblyContext") reject(KernelSpec(2, :source, nothing, f, nothing))
    @test_throws ArgumentError("nonlinear kernel of equation 1 has no tangent") assemble_nonlinear(heat_context(4), (KernelSpec(1, :nonlinear, nothing, f, nothing),), zeros(25), P_HEAT)
end

@testset "facet sets are snapshots taken at construction" begin
    ks = heat_kernels()
    grid = heat_context(4).dh.grid
    right = Set(getfacetset(grid, "right"))
    ctx = heat_context(4; neumann = right)
    before = assemble_linear(ctx, ks, P_HEAT).F
    extra = first(fi for fi in getfacetset(grid, "top") if all(r -> r[1] != fi[1], right))
    push!(right, extra)   # a valid facet whose cell the original set does not touch
    @test assemble_linear(ctx, ks, P_HEAT).F == before
    @test sum(assemble_linear(heat_context(4; neumann = right), ks, P_HEAT).F) - sum(before) ≈
        P_HEAT[1] * P_HEAT[4] * 0.5 rtol = 1.0e-13   # k·h·|facet|, a top facet of the 4×4 grid on [−1, 1]²
end

# Two fields (u, w), P = [Du, Dw, χ, a, b, c]: cross-diffusion χ∇w in eq 1, linear coupling c w in eq 1,
# nonlinear reaction ±u²w, Neumann data on `right` for both equations.
const P_TWO = [1.0, 20.0, 0.3, 0.1, 0.9, 0.7]
dotW(G, W, f) = G[2f - 1] * W[1] + G[2f] * W[2]
function two_field_kernels()
    return (
        KernelSpec(1, :mass, nothing, (U, G, Ud, V, W, X, T, N, P) -> Ud[1] * V, nothing),
        KernelSpec(2, :mass, nothing, (U, G, Ud, V, W, X, T, N, P) -> Ud[2] * V, nothing),
        KernelSpec(
            1, :bilinear, nothing,
            (U, G, Ud, V, W, X, T, N, P) -> P[1] * dotW(G, W, 1) + P[3] * dotW(G, W, 2) + U[1] * V + P[6] * U[2] * V,
            nothing
        ),
        KernelSpec(2, :bilinear, nothing, (U, G, Ud, V, W, X, T, N, P) -> P[2] * dotW(G, W, 2), nothing),
        KernelSpec(1, :source, nothing, (U, G, Ud, V, W, X, T, N, P) -> -P[4] * V, nothing),
        KernelSpec(2, :source, nothing, (U, G, Ud, V, W, X, T, N, P) -> -P[5] * V, nothing),
        KernelSpec(1, :source, :neumann, (U, G, Ud, V, W, X, T, N, P) -> -0.1 * V, nothing),
        KernelSpec(2, :source, :neumann, (U, G, Ud, V, W, X, T, N, P) -> -0.2 * P[2] * V, nothing),
        KernelSpec(
            1, :nonlinear, nothing, (U, G, Ud, V, W, X, T, N, P) -> -U[1]^2 * U[2] * V,
            (U, G, Ud, V, W, X, T, N, P) -> (-2U[1] * U[2] * V, -U[1]^2 * V, 0.0, 0.0, 0.0, 0.0)
        ),
        KernelSpec(
            2, :nonlinear, nothing, (U, G, Ud, V, W, X, T, N, P) -> U[1]^2 * U[2] * V,
            (U, G, Ud, V, W, X, T, N, P) -> (2U[1] * U[2] * V, U[1]^2 * V, 0.0, 0.0, 0.0, 0.0)
        ),
    )
end

"Hand-written two-field blocks: K, M, F and the nonlinear residual r(u) at state `u`."
function two_field_reference(ctx, p, u)
    (; dh, cv, fv) = shared_values(ctx)
    Du, Dw, χ, a, b, c = p
    ru, rw = dof_range(dh, :u), dof_range(dh, :w)
    K, M = allocate_matrix(dh), allocate_matrix(dh)
    F, r = zeros(ndofs(dh)), zeros(ndofs(dh))
    nb = getnbasefunctions(cv)
    Ke, Me, Fe, re = zeros(2nb, 2nb), zeros(2nb, 2nb), zeros(2nb), zeros(2nb)
    aK, aM = start_assemble(K, F), start_assemble(M)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Ke, 0); fill!(Me, 0); fill!(Fe, 0); fill!(re, 0)
        u_elem = u[celldofs(cell)]
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            uq, wq = function_value(cv, q, u_elem, ru), function_value(cv, q, u_elem, rw)
            for i in 1:nb
                φ, ∇φ = shape_value(cv, q, i), shape_gradient(cv, q, i)
                Fe[ru[i]] += a * φ * dΩ
                Fe[rw[i]] += b * φ * dΩ
                re[ru[i]] -= uq^2 * wq * φ * dΩ
                re[rw[i]] += uq^2 * wq * φ * dΩ
                for j in 1:nb
                    ψ, ∇ψ = shape_value(cv, q, j), shape_gradient(cv, q, j)
                    Ke[ru[i], ru[j]] += (Du * (∇φ ⋅ ∇ψ) + ψ * φ) * dΩ
                    Ke[ru[i], rw[j]] += (χ * (∇φ ⋅ ∇ψ) + c * ψ * φ) * dΩ
                    Ke[rw[i], rw[j]] += Dw * (∇φ ⋅ ∇ψ) * dΩ
                    Me[ru[i], ru[j]] += φ * ψ * dΩ
                    Me[rw[i], rw[j]] += φ * ψ * dΩ
                end
            end
        end
        assemble!(aK, celldofs(cell), Ke, Fe)
        assemble!(aM, celldofs(cell), Me)
        assemble!(r, celldofs(cell), re)
    end
    for fc in FacetIterator(dh, getfacetset(dh.grid, "right"))
        Ferrite.reinit!(fv, fc)
        fill!(Fe, 0)
        for q in 1:getnquadpoints(fv), i in 1:nb
            dΓ = getdetJdV(fv, q)
            φ = shape_value(fv, q, i)
            Fe[ru[i]] += 0.1 * φ * dΓ
            Fe[rw[i]] += Dw * 0.2 * φ * dΓ
        end
        assemble!(F, celldofs(fc), Fe)
    end
    return K, M, F, r
end

"Relative error of J δ against a central difference of r along a random direction δ."
function tangent_error(r_of, J, u; ε = 1.0e-6)
    δ = randn(length(u))
    fd = (r_of(u + ε * δ) - r_of(u - ε * δ)) / 2ε
    return norm(J * δ - fd) / norm(fd)
end

@testset "two fields: blocks vs hand-written, tangent vs central differences" begin
    ctx = heat_context(16; robin = Set{FacetIndex}(), nfields = 2)
    ks = two_field_kernels()
    Random.seed!(2)
    u = rand(ndofs(ctx.dh))
    A = assemble_linear(ctx, ks, P_TWO)
    Kr, Mr, Fr, rr = two_field_reference(ctx, P_TWO, u)
    @test relerr(A.K, Kr) ≤ 1.0e-14
    @test relerr(A.M, Mr) ≤ 1.0e-14
    @test relerr(A.F, Fr) ≤ 1.0e-14
    N = assemble_nonlinear(ctx, ks, u, P_TWO)
    @test relerr(N.r, rr) ≤ 1.0e-14
    @test tangent_error(v -> assemble_nonlinear(ctx, ks, v, P_TWO).r, N.J, u) ≤ 1.0e-9
end

# Case (iii): k(u) = 1 + u², −∇·(k(u)∇u) = f, k(u)∂ₙu = k(u) g on `right` (user wrote ∂ₙu = g).
nl_source(X) = sin(X[1]) * X[2] + 1
nl_flux(X) = X[2] + 1
function nonlinear_kernels()
    gw(G, W) = G[1] * W[1] + G[2] * W[2]
    return (
        KernelSpec(
            1, :nonlinear, nothing,
            (U, G, Ud, V, W, X, T, N, P) -> (1 + U[1]^2) * gw(G, W) - nl_source(X) * V,
            (U, G, Ud, V, W, X, T, N, P) -> (2U[1] * gw(G, W), (1 + U[1]^2) * W[1], (1 + U[1]^2) * W[2])
        ),
        KernelSpec(
            1, :nonlinear, :neumann,
            (U, G, Ud, V, W, X, T, N, P) -> -(1 + U[1]^2) * nl_flux(X) * V,
            (U, G, Ud, V, W, X, T, N, P) -> (-2U[1] * nl_flux(X) * V, 0.0, 0.0)
        ),
    )
end

"Hand-written residual r_i = ∫(1+u²)∇u·∇φ − ∫fφ − ∮_right (1+u²) g φ (scratch/dauntless/test_nonlinear.jl)."
function nonlinear_reference(ctx, u)
    (; dh, cv, fv) = shared_values(ctx)
    r = zeros(ndofs(dh))
    n = getnbasefunctions(cv)
    re = zeros(n)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(re, 0)
        u_elem = u[celldofs(cell)]
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            uq, gq = function_value(cv, q, u_elem), function_gradient(cv, q, u_elem)
            for i in 1:n
                re[i] += ((1 + uq^2) * (gq ⋅ shape_gradient(cv, q, i)) - nl_source(X) * shape_value(cv, q, i)) * dΩ
            end
        end
        assemble!(r, celldofs(cell), re)
    end
    for fc in FacetIterator(dh, getfacetset(dh.grid, "right"))
        Ferrite.reinit!(fv, fc)
        fill!(re, 0)
        u_elem = u[celldofs(fc)]
        xs = getcoordinates(fc)
        for q in 1:getnquadpoints(fv)
            X = spatial_coordinate(fv, q, xs)
            uq = function_value(fv, q, u_elem)
            for i in 1:n
                re[i] -= (1 + uq^2) * nl_flux(X) * shape_value(fv, q, i) * getdetJdV(fv, q)
            end
        end
        assemble!(r, celldofs(fc), re)
    end
    return r
end

@testset "case (iii) k(u) = 1 + u²: residual and tangent" begin
    ctx = heat_context(20; robin = Set{FacetIndex}())
    ks = nonlinear_kernels()
    Random.seed!(1)
    u = randn(ndofs(ctx.dh))
    N = assemble_nonlinear(ctx, ks, u, Float64[])
    @test relerr(N.r, nonlinear_reference(ctx, u)) ≤ 1.0e-14
    @test tangent_error(v -> assemble_nonlinear(ctx, ks, v, Float64[]).r, N.J, u) ≤ 1.0e-9
    # linear kinds are ignored by the nonlinear pass and vice versa
    @test iszero(assemble_nonlinear(ctx, heat_kernels(), u, P_HEAT).r)
    @test iszero(assemble_linear(ctx, ks, Float64[]).K)
end

# End to end: seam analysis, weak_form and emit_kernels, driven with the driver's slot types
# (NTuples for U, G, Ud; Vec for W, X, N).
using FiniteElementMethod: weak_form, emit_kernels
using Symbolics
include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse, BC

@testset "end to end: case (i) weak_form -> emit_kernels -> driver" begin
    @variables t x y k α β h
    @variables u(..)
    Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
    U = u(t, x, y)
    eq = Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1
    bcs = [
        BC("left", U ~ 0.1 * y), BC("right", Dx(U) ~ h, [1.0, 0.0]), BC("top", k * Dy(U) + α * U ~ β, [0.0, 1.0]),
    ]
    wf = weak_form(analyse([eq], bcs; depvars = [U], t, sivs = [x, y], params = [k, α, β, h]))
    ks = emit_kernels(wf)
    @test length(ks) == 6
    (; dh, cv, fv) = shared_values(heat_context(20))   # pieces are keyed by the seam's facet-set names here
    ctx = AssemblyContext(dh, [:u], cv, fv, Dict("right" => "right", "top" => "top"))
    A = assemble_linear(ctx, ks, P_HEAT)
    Kr, Mr, Fr = heat_reference(ctx, P_HEAT)
    @test relerr(A.K, Kr) ≤ 1.0e-14
    @test relerr(A.M, Mr) ≤ 1.0e-14
    @test relerr(A.F, Fr) ≤ 1.0e-14
end

# The coefficient contraction against the per-shape-function `f` path (emit_kernels(wf; coeffs = false)) on the
# same weak forms, including coefficients near 1e±8: the two paths order the floating-point operations differently.
near(A, B) = norm(A - B) ≤ 1.0e-14 * norm(B)
function both_paths(wf, fields, p; u = nothing)
    grid = generate_grid(Triangle, (12, 12))
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    foreach(f -> add!(dh, f, ip), fields)
    close!(dh)
    cv, fv = CellValues(QuadratureRule{RefTriangle}(2), ip), FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    ctx = AssemblyContext(dh, fields, cv, fv, Dict(s => s for s in ("left", "right", "bottom", "top")))
    fused, plain = emit_kernels(wf), emit_kernels(wf; coeffs = false)
    A, B = assemble_linear(ctx, fused, p), assemble_linear(ctx, plain, p)
    @test near(A.M, B.M) && near(A.K, B.K) && near(A.F, B.F)
    @test norm(A.K) > 0
    u === nothing && return nothing
    v = u(ndofs(dh))
    @test assemble_nonlinear(ctx, fused, v, p).r == assemble_nonlinear(ctx, plain, v, p).r
    return nothing
end

@testset "coefficient contraction equals the f path" begin
    @variables t x y k α β h Du Dw χ a b
    @variables u(..) w(..)
    Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
    U, Wf = u(t, x, y), w(t, x, y)
    sivs, right, top = [x, y], [1.0, 0.0], [0.0, 1.0]
    scalar(eq, bcs; params = [k, α, β, h]) = weak_form(analyse([eq], bcs; depvars = [U], t, sivs, params))
    heat = scalar(
        Dt(U) ~ Dx(k * Dx(U)) + Dy(k * Dy(U)) + x * y + 1, [BC("right", Dx(U) ~ h, right), BC("top", k * Dy(U) + α * U ~ β, top)]
    )
    for p in (P_HEAT, [1.0e8, 1.0e-8, 1.0e8, 1.0e-8], [1.0e-8, 1.0e8, 1.0e-8, 1.0e8])
        both_paths(heat, [:u], p)
    end
    axy = 1 + x^2 + y
    iib = scalar(
        Dt(U) ~ axy * (Dx(Dx(U)) + Dy(Dy(U))) - 0.7 * Dx(U) - 0.3 * Dy(U) - 2.5 * U + x * y + 1,
        [BC("right", Dx(U) ~ h, right), BC("top", axy * Dy(U) + α * U ~ β, top)]
    )
    both_paths(iib, [:u], P_HEAT)
    iii = scalar(0 ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) + x * y + 1, [BC("right", Dx(U) ~ y + 1, right)]; params = [])
    both_paths(iii, [:u], [0.0]; u = n -> randn(Xoshiro(4), n))
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a - U + U^2 * Wf,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U^2 * Wf,
    ]
    bcs = [BC("right", Du * Dx(U) + χ * Dx(Wf) ~ 0.1, right), BC("right", Dx(Wf) ~ 0.2, right)]
    two = weak_form(analyse(eqs, bcs; depvars = [U, Wf], t, sivs, params = [Du, Dw, χ, a, b]))
    both_paths(two, [:u, :w], [1.0, 20.0, 0.3, 0.1, 0.9]; u = n -> rand(Xoshiro(5), n))
    both_paths(two, [:u, :w], [1.0e8, 1.0e-8, 1.0e8, 1.0e-8, 1.0e8])
end

@testset "a kernel without f needs coeffs" begin
    ks = (KernelSpec(1, :bilinear, nothing, nothing, nothing),)
    @test_throws ArgumentError("bilinear kernel of equation 1 has no f: pass f, or coeffs for a linear kind") assemble_linear(heat_context(4), ks, P_HEAT)
end

@testset "per-field values: NamedTuple, tuple or one shared object" begin
    grid = generate_grid(Triangle, (3, 3))
    ip1, ip2 = Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip2)
    add!(dh, :p, ip1)
    close!(dh)
    qr, fqr = QuadratureRule{RefTriangle}(4), FacetQuadratureRule{RefTriangle}(4)
    cvu, cvp = CellValues(qr, ip2), CellValues(qr, ip1)
    fvu, fvp = FacetValues(fqr, ip2), FacetValues(fqr, ip1)
    byname = AssemblyContext(dh, [:u, :p], (p = cvp, u = cvu), (u = fvu, p = fvp))
    @test byname.cellvalues.u === cvu && byname.cellvalues.p === cvp
    @test AssemblyContext(dh, [:u, :p], (cvu, cvp), (fvu, fvp)).cellvalues == byname.cellvalues
    @test_throws ArgumentError("field p has 3 dofs per cell, but its CellValues have 6 base functions") AssemblyContext(dh, [:u, :p], cvu, fvu)
    @test_throws ArgumentError("no cell values given for field p; the cell values have keys (:u,)") AssemblyContext(dh, [:u, :p], (u = cvu,), (u = fvu, p = fvp))
    @test_throws ArgumentError("1 facet values given for the 2 fields [:u, :p]") AssemblyContext(dh, [:u, :p], (cvu, cvp), (fvu,))
    @test_throws ArgumentError("field q is not a field of the DofHandler, whose fields are [:u, :p]") AssemblyContext(dh, [:u, :q], (cvu, cvp), (fvu, fvp))
    cvp_coarse = CellValues(QuadratureRule{RefTriangle}(2), ip1)
    @test_throws ArgumentError("the CellValues of all fields must share one quadrature rule, but they have 6, 3 quadrature points") AssemblyContext(dh, [:u, :p], (cvu, cvp_coarse), (fvu, fvp))
end
