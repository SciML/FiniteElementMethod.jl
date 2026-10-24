# Generic Ferrite assembly driver (spec §6.5, §6.6, §10.1) against hand-written Ferrite loops.
# Kernels are built by hand with the contract signature f(U, G, Ud, V, W, X, T, N, P).
using FiniteElementMethod, Ferrite, LinearAlgebra, Random, Test
using FiniteElementMethod: AssemblyContext, KernelSpec, assemble_linear, assemble_nonlinear

relerr(A, B) = norm(A - B) / norm(B)

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
    (; dh, cv, fv) = ctx
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
    @test_throws ErrorException assemble_linear(heat_context(4; neumann = bogus), ks, P_HEAT)
    @test_throws ErrorException assemble_linear(heat_context(4; robin = Set([FacetIndex(1, 4)])), ks, P_HEAT)
    @test_throws ErrorException assemble_linear(heat_context(4; neumann = "nonexistent"), ks, P_HEAT)
    unknown = (KernelSpec(1, :source, :elsewhere, (U, G, Ud, V, W, X, T, N, P) -> V, nothing),)
    @test_throws ErrorException assemble_linear(heat_context(4), unknown, P_HEAT)
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
    (; dh, cv, fv) = ctx
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
        ue = u[celldofs(cell)]
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            uq, wq = function_value(cv, q, ue, ru), function_value(cv, q, ue, rw)
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
    (; dh, cv, fv) = ctx
    r = zeros(ndofs(dh))
    n = getnbasefunctions(cv)
    re = zeros(n)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(re, 0)
        ue = u[celldofs(cell)]
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            uq, gq = function_value(cv, q, ue), function_gradient(cv, q, ue)
            for i in 1:n
                re[i] += ((1 + uq^2) * (gq ⋅ shape_gradient(cv, q, i)) - nl_source(X) * shape_value(cv, q, i)) * dΩ
            end
        end
        assemble!(r, celldofs(cell), re)
    end
    for fc in FacetIterator(dh, getfacetset(dh.grid, "right"))
        Ferrite.reinit!(fv, fc)
        fill!(re, 0)
        ue = u[celldofs(fc)]
        xs = getcoordinates(fc)
        for q in 1:getnquadpoints(fv)
            X = spatial_coordinate(fv, q, xs)
            uq = function_value(fv, q, ue)
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
