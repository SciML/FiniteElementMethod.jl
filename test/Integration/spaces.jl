# End to end, per-field spaces (spec §6.2, §6.3, M4): fields of different order with their own CellValues, coupled
# through bilinear and nonlinear terms, and a field that lives on half of the domain through a SubDofHandler. Each
# case is assembled through the chain with the coefficient form and with the per-shape-function `f` path, against
# a hand-written Ferrite loop.
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y
@variables u1(..) u2(..) pr(..) u(..) w(..) c(..)
Dx, Dy = Differential(x), Differential(y)

const SHAPE_OF = Dict(Triangle => RefTriangle, Quadrilateral => RefQuadrilateral)

# A DofHandler with one field per `orders` entry (name => order), and per-field values on one quadrature rule.
function spaces_setup(celltype, orders; nel = 6)
    grid = generate_grid(celltype, (nel, nel))
    shape = SHAPE_OF[celltype]
    dh = DofHandler(grid)
    for (f, order) in orders
        add!(dh, f, Lagrange{shape, order}())
    end
    close!(dh)
    q = 2 * maximum(last, orders)
    cellvalues = NamedTuple(f => CellValues(QuadratureRule{shape}(q), Lagrange{shape, order}()) for (f, order) in orders)
    facetvalues = NamedTuple(f => FacetValues(FacetQuadratureRule{shape}(q), Lagrange{shape, order}()) for (f, order) in orders)
    fields = first.(orders)
    return (; grid, dh, cellvalues, facetvalues, fields, ctx = AssemblyContext(dh, fields, cellvalues, facetvalues))
end

both_paths(wf) = (emit_kernels(wf), emit_kernels(wf; coeffs = false))

@testset "Stokes blocks, $(celltype) P2/P1" for celltype in (Triangle, Quadrilateral)
    # −Δu + ∇p = f, ∇·u = 0, velocity components u1, u2 of order 2 and pressure of order 1. The seam keeps ∂p as a
    # remainder and does not integrate the divergence by parts, so K[u_d, p] = ∫ ∂_d ψ φ and K[p, u_d] = −∫ ∂_d φ ψ.
    U1, U2, Pr = u1(t, x, y), u2(t, x, y), pr(t, x, y)
    eqs = [
        0 ~ Dx(Dx(U1)) + Dy(Dy(U1)) - Dx(Pr) + x * y,
        0 ~ Dx(Dx(U2)) + Dy(Dy(U2)) - Dy(Pr) + 1,
        0 ~ Dx(U1) + Dy(U2),
    ]
    ch = chain(eqs, BC[]; depvars = [U1, U2, Pr], t, sivs = [x, y])
    s = spaces_setup(celltype, [:u1 => 2, :u2 => 2, :p => 1])
    (; dh, cellvalues) = s
    cvu, cvp = cellvalues.u1, cellvalues.p
    K, F = allocate_matrix(dh), zeros(ndofs(dh))
    r1, r2, rp = dof_range(dh, :u1), dof_range(dh, :u2), dof_range(dh, :p)
    n = ndofs_per_cell(dh)
    Ke, Fe = zeros(n, n), zeros(n)
    assembler = start_assemble(K, F)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cvu, cell)
        Ferrite.reinit!(cvp, cell)
        fill!(Ke, 0); fill!(Fe, 0)
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, q)
            X = spatial_coordinate(cvu, q, xs)
            for i in 1:getnbasefunctions(cvu)
                φi, ∇φi = shape_value(cvu, q, i), shape_gradient(cvu, q, i)
                Fe[r1[i]] += X[1] * X[2] * φi * dΩ
                Fe[r2[i]] += φi * dΩ
                for j in 1:getnbasefunctions(cvu)
                    a = (∇φi ⋅ shape_gradient(cvu, q, j)) * dΩ
                    Ke[r1[i], r1[j]] += a
                    Ke[r2[i], r2[j]] += a
                end
                for j in 1:getnbasefunctions(cvp)
                    ∇ψj = shape_gradient(cvp, q, j)
                    Ke[r1[i], rp[j]] += ∇ψj[1] * φi * dΩ
                    Ke[r2[i], rp[j]] += ∇ψj[2] * φi * dΩ
                end
            end
            for i in 1:getnbasefunctions(cvp), j in 1:getnbasefunctions(cvu)
                ψi, ∇φj = shape_value(cvp, q, i), shape_gradient(cvu, q, j)
                Ke[rp[i], r1[j]] -= ∇φj[1] * ψi * dΩ
                Ke[rp[i], r2[j]] -= ∇φj[2] * ψi * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Ke, Fe)
    end
    p_dofs = field_dofs(s, :p)
    for kernels in both_paths(ch.wf)
        A = assemble_linear(s.ctx, kernels, P0)
        @test relerr(A.K, K) ≤ 1.0e-14
        @test relerr(A.F, F) ≤ 1.0e-14
        @test iszero(A.K[p_dofs, p_dofs])
        @test norm(A.K[p_dofs, field_dofs(s, :u1)]) > 0
    end
end

@testset "unequal orders with a nonlinear coupling: P2 u, P1 w" begin
    # 0 = Δu − u²w + xy + 1 and 0 = ∇·((1 + u²)∇w) + 1.
    U, Wf = u(t, x, y), w(t, x, y)
    eqs = [0 ~ Dx(Dx(U)) + Dy(Dy(U)) - U^2 * Wf + x * y + 1, 0 ~ Dx((1 + U^2) * Dx(Wf)) + Dy((1 + U^2) * Dy(Wf)) + 1]
    ch = chain(eqs, BC[]; depvars = [U, Wf], t, sivs = [x, y])
    s = spaces_setup(Triangle, [:u => 2, :w => 1])
    (; dh, cellvalues) = s
    cvu, cvw = cellvalues.u, cellvalues.w
    ru, rw = dof_range(dh, :u), dof_range(dh, :w)
    function residual_ref(v)
        r = zeros(ndofs(dh))
        re = zeros(ndofs_per_cell(dh))
        for cell in CellIterator(dh)
            Ferrite.reinit!(cvu, cell)
            Ferrite.reinit!(cvw, cell)
            fill!(re, 0)
            v_elem = v[celldofs(cell)]
            xs = getcoordinates(cell)
            for q in 1:getnquadpoints(cvu)
                dΩ = getdetJdV(cvu, q)
                X = spatial_coordinate(cvu, q, xs)
                uq, ∇uq = function_value(cvu, q, v_elem, ru), function_gradient(cvu, q, v_elem, ru)
                wq, ∇wq = function_value(cvw, q, v_elem, rw), function_gradient(cvw, q, v_elem, rw)
                for i in 1:getnbasefunctions(cvu)
                    φ, ∇φ = shape_value(cvu, q, i), shape_gradient(cvu, q, i)
                    re[ru[i]] += (∇uq ⋅ ∇φ + uq^2 * wq * φ - (X[1] * X[2] + 1) * φ) * dΩ
                end
                for i in 1:getnbasefunctions(cvw)
                    ψ, ∇ψ = shape_value(cvw, q, i), shape_gradient(cvw, q, i)
                    re[rw[i]] += ((1 + uq^2) * (∇wq ⋅ ∇ψ) - ψ) * dΩ
                end
            end
            assemble!(r, celldofs(cell), re)
        end
        return r
    end
    v = randn(Xoshiro(7), ndofs(dh))
    for kernels in both_paths(ch.wf)
        R(z) = chain_residual(s, kernels, z, P0)[1]
        r, J = chain_residual(s, kernels, v, P0)
        @test relerr(r, residual_ref(v)) ≤ 1.0e-14
        @test tangent_error(R, J, v) ≤ 1.0e-9
    end
end

@testset "a field on half of the domain (SubDofHandler)" begin
    # u everywhere, c only on the cells left of x = 0: −Δu + u − c = xy + 1 and −Δc + 2c = 1. Where c has no dofs
    # its equation is not assembled and it enters u's equation as zero.
    U, C = u(t, x, y), c(t, x, y)
    eqs = [0 ~ Dx(Dx(U)) + Dy(Dy(U)) - U + C + x * y + 1, 0 ~ Dx(Dx(C)) + Dy(Dy(C)) - 2C + 1]
    ch = chain(eqs, BC[]; depvars = [U, C], t, sivs = [x, y])
    grid = generate_grid(Triangle, (8, 8))
    left = Set(i for i in 1:getncells(grid) if sum(n -> grid.nodes[n].x[1], grid.cells[i].nodes) / 3 < 0)
    right = setdiff(Set(1:getncells(grid)), left)
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    sdh_left = SubDofHandler(dh, left)
    add!(sdh_left, :u, ip)
    add!(sdh_left, :c, ip)
    sdh_right = SubDofHandler(dh, right)
    add!(sdh_right, :u, ip)
    close!(dh)
    cv, fv = CellValues(QuadratureRule{RefTriangle}(2), ip), FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    ctx = AssemblyContext(dh, [:u, :c], cv, fv)
    K, F, M = allocate_matrix(dh), zeros(ndofs(dh)), allocate_matrix(dh)
    assembler = start_assemble(K, F)
    for sdh in dh.subdofhandlers
        has_c = :c in sdh.field_names
        ru = dof_range(sdh, :u)
        rc = has_c ? dof_range(sdh, :c) : (1:0)
        n = ndofs_per_cell(sdh)
        Ke, Fe = zeros(n, n), zeros(n)
        for cell in CellIterator(sdh)
            Ferrite.reinit!(cv, cell)
            fill!(Ke, 0); fill!(Fe, 0)
            xs = getcoordinates(cell)
            for q in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, q)
                X = spatial_coordinate(cv, q, xs)
                for i in 1:getnbasefunctions(cv)
                    φi, ∇φi = shape_value(cv, q, i), shape_gradient(cv, q, i)
                    Fe[ru[i]] += (X[1] * X[2] + 1) * φi * dΩ
                    has_c && (Fe[rc[i]] += φi * dΩ)
                    for j in 1:getnbasefunctions(cv)
                        φj, ∇φj = shape_value(cv, q, j), shape_gradient(cv, q, j)
                        Ke[ru[i], ru[j]] += (∇φi ⋅ ∇φj + φi * φj) * dΩ
                        has_c || continue
                        Ke[ru[i], rc[j]] -= φi * φj * dΩ
                        Ke[rc[i], rc[j]] += (∇φi ⋅ ∇φj + 2 * φi * φj) * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Ke, Fe)
        end
    end
    @test ndofs(dh) < 2 * getnnodes(grid)
    for kernels in both_paths(ch.wf)
        A = assemble_linear(ctx, kernels, P0)
        @test relerr(A.K, K) ≤ 1.0e-14
        @test relerr(A.F, F) ≤ 1.0e-14
    end
end
