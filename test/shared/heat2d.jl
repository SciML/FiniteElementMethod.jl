# Raw-Ferrite operators for the numeric DAE tests: no FEMSpace, no PDESystem.
using Ferrite, SparseArrays, LinearAlgebra
using SciMLBase: DAEFunction, DAEProblem

# Lagrange P1 dof coordinates (dof i of a cell sits on node i of that cell).
function dof_coordinates(dh, field)
    coords = Vector{NTuple{2, Float64}}(undef, ndofs(dh))
    for cell in CellIterator(dh)
        for (x, d) in zip(getcoordinates(cell), celldofs(cell)[dof_range(dh, field)])
            coords[d] = Tuple(x)
        end
    end
    return coords
end

# u_t = Δu + 1 on the unit square, P1 triangles, n × n cells, so (n + 1)^2 dofs; Dirichlet on
# "left", homogeneous Neumann elsewhere. M and K share the pattern of allocate_matrix(dh, ch).
function heat2d(n)
    grid = generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0))
    ip = Lagrange{RefTriangle, 1}()
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    close!(ch)
    K, M = allocate_matrix(dh, ch), allocate_matrix(dh, ch)
    F = zeros(ndofs(dh))
    nb = getnbasefunctions(cv)
    Ke, Me, Fe = zeros(nb, nb), zeros(nb, nb), zeros(nb)
    Kasm, Masm = start_assemble(K, F), start_assemble(M)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0); fill!(Me, 0); fill!(Fe, 0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:nb
                v, ∇v = shape_value(cv, q, i), shape_gradient(cv, q, i)
                Fe[i] += v * dΩ
                for j in 1:nb
                    Ke[i, j] += (∇v ⋅ shape_gradient(cv, q, j)) * dΩ
                    Me[i, j] += v * shape_value(cv, q, j) * dΩ
                end
            end
        end
        assemble!(Kasm, celldofs(cell), Ke, Fe)
        assemble!(Masm, celldofs(cell), Me)
    end
    return (; M, K, F, prescribed = copy(ch.prescribed_dofs), coords = dof_coordinates(dh, :u))
end

# Two fields, numbered field-wise: u_t = Δu + v + 1 (u = 0 on "left"), and the algebraic
# 0 = Δv + e^{-t} sin(πx) (v = 0 on the whole boundary). The v rows of M are all zero.
function coupled2d(n)
    grid = generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0))
    ip = Lagrange{RefTriangle, 1}()
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :v, ip)
    close!(dh)
    renumber!(dh, DofOrder.FieldWise())
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    boundary = union((getfacetset(grid, s) for s in ("left", "right", "top", "bottom"))...)
    add!(ch, Dirichlet(:v, boundary, (x, t) -> 0.0))
    close!(ch)
    K, M = allocate_matrix(dh, ch), allocate_matrix(dh, ch)
    F0, Fs = zeros(ndofs(dh)), zeros(ndofs(dh))
    nb = getnbasefunctions(cv)
    nd = ndofs_per_cell(dh)
    ru, rv = dof_range(dh, :u), dof_range(dh, :v)
    Ke, Me, F0e, Fse = zeros(nd, nd), zeros(nd, nd), zeros(nd), zeros(nd)
    Kasm, Masm = start_assemble(K, F0), start_assemble(M)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0); fill!(Me, 0); fill!(F0e, 0); fill!(Fse, 0)
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            x = spatial_coordinate(cv, q, xs)
            for i in 1:nb
                φi, ∇φi = shape_value(cv, q, i), shape_gradient(cv, q, i)
                F0e[ru[i]] += φi * dΩ
                Fse[rv[i]] += sin(π * x[1]) * φi * dΩ
                for j in 1:nb
                    φj, ∇φj = shape_value(cv, q, j), shape_gradient(cv, q, j)
                    Me[ru[i], ru[j]] += φi * φj * dΩ
                    Ke[ru[i], ru[j]] += (∇φi ⋅ ∇φj) * dΩ
                    Ke[ru[i], rv[j]] -= φi * φj * dΩ
                    Ke[rv[i], rv[j]] += (∇φi ⋅ ∇φj) * dΩ
                end
            end
        end
        assemble!(Kasm, celldofs(cell), Ke, F0e)
        assemble!(Masm, celldofs(cell), Me)
        assemble!(Fs, celldofs(cell), Fse)
    end
    nu = ndofs(dh) ÷ 2
    fields = [:u => 1:nu, :v => (nu + 1):ndofs(dh)]
    load! = (out, t) -> (out .= F0 .+ exp(-t) .* Fs; out)
    return (; M, K, load!, fields, prescribed = copy(ch.prescribed_dofs))
end

# Independent reference: dense-indexed row replacement and a hand-written DAEProblem.
function hand_rowreplace(M, K, F, prescribed, g = 0.0)
    Mz, Kz, Fz = copy(M), copy(K), copy(F)
    for d in prescribed
        Mz[d, :] .= 0; Kz[d, :] .= 0; Kz[d, d] = 1; Fz[d] = g
    end
    return dropzeros!(Mz), dropzeros!(Kz), Fz
end

# F is a vector or (out, t) -> out; g(t) is the Dirichlet value; a row is algebraic iff its M
# row is zero. The algebraic entries of u0 are replaced by consistent ones.
function hand_daeproblem(M, K, F, prescribed, u0, tspan; g = t -> 0.0)
    n = length(u0)
    Mz, Kz, _ = hand_rowreplace(M, K, zeros(n), prescribed)
    Ft = t -> begin
        Fv = F isa AbstractVector ? copy(F) : F(zeros(n), t)
        Fv[prescribed] .= g(t)
        Fv
    end
    dv = vec(sum(abs, Mz; dims = 2)) .> 0
    # Consistent algebraic values, so the reference needs no initialisation.
    al = .!dv
    u0 = copy(u0)
    u0[al] = Kz[al, al] \ (Ft(first(tspan))[al] - Kz[al, dv] * u0[dv])
    A = copy(Mz)
    for i in findall(.!dv)
        A[i, i] = 1
    end
    rhs = Ft(first(tspan)) - Kz * u0
    rhs[.!dv] .= 0
    du0 = A \ rhs
    f = (res, du, u, p, t) -> (mul!(res, Mz, du); mul!(res, Kz, u, 1, 1); res .-= Ft(t); nothing)
    jac = (J, du, u, p, γ, t) -> (J .= Kz .+ γ .* Mz; nothing)
    return DAEProblem(DAEFunction(f; jac, jac_prototype = Kz + Mz), du0, u0, tspan; differential_vars = dv)
end
