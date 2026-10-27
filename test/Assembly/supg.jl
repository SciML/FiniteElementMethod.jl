# SUPG test space (spec §1 goal 4, §5.1 stage e, §10.1 SUPG row): K and F of steady advection–diffusion on P2
# triangles against a hand-written SUPG assembly (scratch/dauntless/test_testspace.jl (b)), for a constant τ and
# for τ(hK) = hK / (2|b|) with hK the cell diameter.
using FiniteElementMethod: weak_form, emit_kernels, AssemblyContext, assemble_linear, SUPG, Galerkin, CellCoeffs
using Ferrite, LinearAlgebra, Symbolics, Test

include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse

relerr(A, B) = norm(A - B) / norm(B)

@variables t x y ε
@variables u(..)
Dx, Dy = Differential(x), Differential(y)
U = u(t, x, y)
const B = [1.0, 0.5]
const EPS = 0.01
const P_SUPG = [EPS]

# −εΔu + b·∇u = 1 + x, as the user writes it.
analysis() = analyse(
    [0 ~ ε * (Dx(Dx(U)) + Dy(Dy(U))) - B[1] * Dx(U) - B[2] * Dy(U) + 1 + x], [];
    depvars = [U], t, sivs = [x, y], params = [ε]
)

function p2_context(n)
    grid = generate_grid(Triangle, (n, n))
    ip = Lagrange{RefTriangle, 2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    cv = CellValues(QuadratureRule{RefTriangle}(4), ip; update_hessians = true)
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(4), ip)
    return AssemblyContext(dh, [:u], cv, fv), grid, dh, cv
end

diameter(xs) = maximum(norm(xs[i] - xs[j]) for i in eachindex(xs) for j in eachindex(xs))

function supg_hand(dh, cv, τ)
    K = allocate_matrix(dh)
    F = zeros(ndofs(dh))
    n = getnbasefunctions(cv)
    Ke, Fe = zeros(n, n), zeros(n)
    A = start_assemble(K, F)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(Fe, 0)
        xs = getcoordinates(cell)
        τK = τ(diameter(xs))
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            for i in 1:n
                φ, ∇φ = shape_value(cv, q, i), shape_gradient(cv, q, i)
                Fe[i] += (1 + X[1]) * (φ + τK * (B ⋅ ∇φ)) * dΩ
                for j in 1:n
                    ψ, ∇ψ, Hψ = shape_value(cv, q, j), shape_gradient(cv, q, j), shape_hessian(cv, q, j)
                    Ke[i, j] += (EPS * (∇φ ⋅ ∇ψ) + (B ⋅ ∇ψ) * φ + τK * (B ⋅ ∇φ) * (-EPS * tr(Hψ) + B ⋅ ∇ψ)) * dΩ
                end
            end
        end
        assemble!(A, celldofs(cell), Ke, Fe)
    end
    return K, F
end

@testset "SUPG K and F on P2 advection–diffusion, τ = $name" for (name, τ, τsym) in (
        ("0.05", h -> 0.05, 0.05),
        ("hK / (2|b|)", h -> h / (2 * norm(B)), (hK, b) -> hK / (2 * sqrt(sum(abs2, b)))),
    )
    ctx, grid, dh, cv = p2_context(12)
    wf = weak_form(analysis(); test_space = SUPG(τsym))
    kernels = emit_kernels(wf)
    @test any(k -> k.coeffs isa CellCoeffs, kernels)
    A = assemble_linear(ctx, kernels, P_SUPG)
    Kh, Fh = supg_hand(dh, cv, τ)
    @test relerr(A.K, Kh) ≤ 1.0e-14
    @test relerr(A.F, Fh) ≤ 1.0e-14
    # the stabilisation is not negligible: Galerkin differs
    G = assemble_linear(ctx, emit_kernels(weak_form(analysis())), P_SUPG)
    @test relerr(G.K, Kh) > 1.0e-3
end

@testset "SUPG reads b from the analysis and rejects a state-dependent one" begin
    wf_auto = weak_form(analysis(); test_space = SUPG(0.05))
    wf_given = weak_form(analysis(); test_space = SUPG(0.05; b = B))
    ctx, = p2_context(4)
    @test assemble_linear(ctx, emit_kernels(wf_auto), P_SUPG).K ≈ assemble_linear(ctx, emit_kernels(wf_given), P_SUPG).K
    burgers = analyse([0 ~ ε * Dx(Dx(U)) - U * Dx(U)], []; depvars = [U], t, sivs = [x, y], params = [ε])
    @test_throws ArgumentError weak_form(burgers; test_space = SUPG(0.05))
end

@testset "SUPG kernels need CellValues with Hessians" begin
    grid = generate_grid(Triangle, (2, 2))
    ip = Lagrange{RefTriangle, 2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ctx = AssemblyContext(dh, [:u], CellValues(QuadratureRule{RefTriangle}(4), ip), FacetValues(FacetQuadratureRule{RefTriangle}(4), ip))
    kernels = emit_kernels(weak_form(analysis(); test_space = SUPG(0.05)))
    @test_throws ArgumentError("a stabilised (SUPG) kernel needs the shape-function Hessians of field u: build its CellValues with update_hessians = true") assemble_linear(ctx, kernels, P_SUPG)
end

@testset "per-field values: only the fields whose Hessians SUPG reads need update_hessians" begin
    @variables w(..)
    Wf = w(t, x, y)
    an = analyse(
        [
            0 ~ ε * (Dx(Dx(U)) + Dy(Dy(U))) - B[1] * Dx(U) - B[2] * Dy(U) + 1 + x,
            0 ~ Dx(Dx(Wf)) + Dy(Dy(Wf)) + U,
        ], [];
        depvars = [U, Wf], t, sivs = [x, y], params = [ε]
    )
    kernels = emit_kernels(weak_form(an; test_space = SUPG(0.05)))
    @test [k.coeffs.fields for k in kernels if k.coeffs isa CellCoeffs && k.kind === :bilinear] == [(1,)]
    grid = generate_grid(Triangle, (6, 6))
    ip = Lagrange{RefTriangle, 2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :w, ip)
    close!(dh)
    qr, fqr = QuadratureRule{RefTriangle}(4), FacetQuadratureRule{RefTriangle}(4)
    with_h, without_h = CellValues(qr, ip; update_hessians = true), CellValues(qr, ip)
    fv = FacetValues(fqr, ip)
    both = AssemblyContext(dh, [:u, :w], (u = with_h, w = CellValues(qr, ip; update_hessians = true)), fv)
    only_u = AssemblyContext(dh, [:u, :w], (u = with_h, w = without_h), fv)
    @test assemble_linear(only_u, kernels, P_SUPG).K == assemble_linear(both, kernels, P_SUPG).K
    only_w = AssemblyContext(dh, [:u, :w], (u = without_h, w = with_h), fv)
    @test_throws ArgumentError("a stabilised (SUPG) kernel needs the shape-function Hessians of field u: build its CellValues with update_hessians = true") assemble_linear(only_w, kernels, P_SUPG)
end
