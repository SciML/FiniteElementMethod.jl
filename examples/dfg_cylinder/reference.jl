using Ferrite: Ferrite
using LinearAlgebra: dot, norm
using SparseArrays: SparseMatrixCSC

"""
    dfg_reference_steady(grid; ν = 1.0e-3, Umax = 0.3, tol = 1.0e-10, maxiter = 20)

Hand-written Ferrite solution of the steady DFG 2D-1 problem on `grid` (from [`dfg_channel_grid`](@ref)): scalar
velocity components `u`, `v` on `Lagrange{RefQuadrilateral, 2}` and pressure `p` on `Lagrange{RefQuadrilateral, 1}`,
parabolic inflow of peak `Umax`, no slip on the walls and the cylinder, and the do-nothing condition
`ν ∂ₙu − p n = 0` at the outlet, solved by Newton's method from zero. Returns `(dh, x, iterations)`.

This is the oracle the PDESystem path is compared with on the same mesh; it uses no FiniteElementMethod code.
"""
function dfg_reference_steady(grid; ν = 1.0e-3, Umax = 0.3, tol = 1.0e-10, maxiter = 20, pressure = :weak)
    shape = Ferrite.RefQuadrilateral
    ip_u, ip_p = Ferrite.Lagrange{shape, 2}(), Ferrite.Lagrange{shape, 1}()
    ip_geo = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
    qr = Ferrite.QuadratureRule{shape}(4)
    cv_u, cv_p = Ferrite.CellValues(qr, ip_u, ip_geo), Ferrite.CellValues(qr, ip_p, ip_geo)
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip_u)
    Ferrite.add!(dh, :v, ip_u)
    Ferrite.add!(dh, :p, ip_p)
    Ferrite.close!(dh)
    ch = Ferrite.ConstraintHandler(dh)
    noslip = union(Ferrite.getfacetset(grid, "walls"), Ferrite.getfacetset(grid, "cylinder"))
    inflow = Ferrite.getfacetset(grid, "inlet")
    Ferrite.add!(ch, Ferrite.Dirichlet(:u, noslip, Returns(0.0)))
    Ferrite.add!(ch, Ferrite.Dirichlet(:v, noslip, Returns(0.0)))
    Ferrite.add!(ch, Ferrite.Dirichlet(:u, inflow, x -> 4Umax * x[2] * (0.41 - x[2]) / 0.41^2))
    Ferrite.add!(ch, Ferrite.Dirichlet(:v, inflow, Returns(0.0)))
    pressure === :strong && Ferrite.add!(ch, Ferrite.Dirichlet(:p, Ferrite.getfacetset(grid, "outlet"), Returns(0.0)))
    Ferrite.close!(ch)

    ru, rv, rp = Ferrite.dof_range(dh, :u), Ferrite.dof_range(dh, :v), Ferrite.dof_range(dh, :p)
    nu, np = Ferrite.getnbasefunctions(cv_u), Ferrite.getnbasefunctions(cv_p)
    ne = 2nu + np
    J = Ferrite.allocate_matrix(dh)
    r = zeros(Ferrite.ndofs(dh))
    x = zeros(Ferrite.ndofs(dh))
    Ferrite.apply!(x, ch)
    Je, re = zeros(ne, ne), zeros(ne)
    iterations = 0
    for it in 1:maxiter
        assembler = Ferrite.start_assemble(J, r)
        for cell in Ferrite.CellIterator(dh)
            Ferrite.reinit!(cv_u, cell)
            Ferrite.reinit!(cv_p, cell)
            dofs = Ferrite.celldofs(cell)
            u_cell, v_cell, p_cell = x[dofs[ru]], x[dofs[rv]], x[dofs[rp]]
            fill!(Je, 0)
            fill!(re, 0)
            for q in 1:Ferrite.getnquadpoints(cv_u)
                dΩ = Ferrite.getdetJdV(cv_u, q)
                U = (Ferrite.function_value(cv_u, q, u_cell), Ferrite.function_value(cv_u, q, v_cell))
                ∇U = (Ferrite.function_gradient(cv_u, q, u_cell), Ferrite.function_gradient(cv_u, q, v_cell))
                p = Ferrite.function_value(cv_p, q, p_cell)
                ∇p = Ferrite.function_gradient(cv_p, q, p_cell)
                div = ∇U[1][1] + ∇U[2][2]
                blocks = (ru, rv)
                for i in 1:nu
                    φ, ∇φ = Ferrite.shape_value(cv_u, q, i), Ferrite.shape_gradient(cv_u, q, i)
                    for k in 1:2
                        a = blocks[k][i]
                        conv = U[1] * ∇U[k][1] + U[2] * ∇U[k][2]
                        re[a] += (conv * φ + ν * dot(∇U[k], ∇φ) + (pressure === :strong ? ∇p[k] * φ : -p * ∇φ[k])) * dΩ
                        for j in 1:nu
                            ψ, ∇ψ = Ferrite.shape_value(cv_u, q, j), Ferrite.shape_gradient(cv_u, q, j)
                            # d(conv_k)/d(u_l) along ψ: U·∇ψ (l = k) + ψ ∂_l U_k
                            for l in 1:2
                                b = blocks[l][j]
                                dconv = (l == k ? U[1] * ∇ψ[1] + U[2] * ∇ψ[2] : 0.0) + ψ * ∇U[k][l]
                                Je[a, b] += (dconv * φ + (l == k ? ν * dot(∇ψ, ∇φ) : 0.0)) * dΩ
                            end
                        end
                        for j in 1:np
                            Je[a, rp[j]] += (
                                pressure === :strong ? Ferrite.shape_gradient(cv_p, q, j)[k] * φ :
                                    -Ferrite.shape_value(cv_p, q, j) * ∇φ[k]
                            ) * dΩ
                        end
                    end
                end
                for i in 1:np
                    ψ = Ferrite.shape_value(cv_p, q, i)
                    re[rp[i]] += ψ * div * dΩ
                    for j in 1:nu
                        ∇φ = Ferrite.shape_gradient(cv_u, q, j)
                        Je[rp[i], ru[j]] += ψ * ∇φ[1] * dΩ
                        Je[rp[i], rv[j]] += ψ * ∇φ[2] * dΩ
                    end
                end
            end
            Ferrite.assemble!(assembler, dofs, Je, re)
        end
        Ferrite.apply_zero!(J, r, ch)
        resnorm = norm(r)
        iterations = it
        resnorm < tol && break
        x .-= J \ r
    end
    return dh, x, iterations
end
