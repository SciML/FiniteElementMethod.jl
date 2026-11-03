using Ferrite: Ferrite
using LinearAlgebra: dot

"""
    dfg_coefficients(dh, x; ν, Umean, D = 0.1, dxdt = nothing, fields = (:u, :v, :p))

Drag and lift coefficients `(cD, cL)` of the cylinder (facet set `"cylinder"`) and the pressure difference
`Δp = p(0.15, 0.2) − p(0.25, 0.2)` between its front and back, from the dof vector `x` of the `DofHandler` `dh`
with velocity components `fields[1:2]` and pressure `fields[3]`.

The forces are the weighted residuals of the momentum equations (John 2004, Braack & Richter 2006):
with `w` the finite-element function equal to `e_k` on the cylinder and zero at every other node,

    F_k = −∫_Ω (∂ₜu·w + ν ∇u : ∇w + (u·∇)u·w − p ∇·w) dx,

which equals the surface integral `∮ (ν ∂ₙu − p n)·e_k` over the cylinder for the exact solution but converges
at the order of the energy error squared, where the surface integral of the discrete gradient converges at the
order of the energy error. `dxdt` is the time derivative of `x` (omit it for a steady state). The coefficients are
`2F/(Umean² D)` with unit density.
"""
function dfg_coefficients(dh, x; ν, Umean, D = 0.1, dxdt = nothing, fields = (:u, :v, :p))
    fu, fv, fp = fields
    grid = dh.grid
    sdh = only(dh.subdofhandlers)
    ip_u = Ferrite.getfieldinterpolation(sdh, fu)
    ip_p = Ferrite.getfieldinterpolation(sdh, fp)
    ip_geo = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
    shape = Ferrite.getrefshape(ip_u)
    qr = Ferrite.QuadratureRule{shape}(2 * Ferrite.getorder(ip_u) + 1)
    cv_u, cv_p = Ferrite.CellValues(qr, ip_u, ip_geo), Ferrite.CellValues(qr, ip_p, ip_geo)

    # The weight is the nodal interpolant of 1 on the cylinder: the sum of its velocity basis functions.
    w = zeros(Ferrite.ndofs(dh))
    ch = Ferrite.ConstraintHandler(dh)
    Ferrite.add!(ch, Ferrite.Dirichlet(fu, Ferrite.getfacetset(grid, "cylinder"), Returns(1.0)))
    Ferrite.close!(ch)
    w[ch.prescribed_dofs] .= 1

    ru, rv = Ferrite.dof_range(dh, fu), Ferrite.dof_range(dh, fv)
    rp = Ferrite.dof_range(dh, fp)
    F = zeros(2)
    coords = Ferrite.getcoordinates(grid, 1)
    for cell in 1:Ferrite.getncells(grid)
        dofs = Ferrite.celldofs(dh, cell)
        w_cell = w[dofs[ru]]
        any(!iszero, w_cell) || continue
        Ferrite.getcoordinates!(coords, grid, cell)
        Ferrite.reinit!(cv_u, coords)
        Ferrite.reinit!(cv_p, coords)
        u_cell, v_cell, p_cell = x[dofs[ru]], x[dofs[rv]], x[dofs[rp]]
        for q in 1:Ferrite.getnquadpoints(cv_u)
            dΩ = Ferrite.getdetJdV(cv_u, q)
            U = (Ferrite.function_value(cv_u, q, u_cell), Ferrite.function_value(cv_u, q, v_cell))
            ∇U = (Ferrite.function_gradient(cv_u, q, u_cell), Ferrite.function_gradient(cv_u, q, v_cell))
            ∂U = dxdt === nothing ? (0.0, 0.0) :
                (Ferrite.function_value(cv_u, q, dxdt[dofs[ru]]), Ferrite.function_value(cv_u, q, dxdt[dofs[rv]]))
            p = Ferrite.function_value(cv_p, q, p_cell)
            W = Ferrite.function_value(cv_u, q, w_cell)
            ∇W = Ferrite.function_gradient(cv_u, q, w_cell)
            for k in 1:2
                convection = U[1] * ∇U[k][1] + U[2] * ∇U[k][2]
                F[k] -= ((∂U[k] + convection) * W + ν * dot(∇U[k], ∇W) - p * ∇W[k]) * dΩ
            end
        end
    end
    scale = 2 / (Umean^2 * D)
    ph = Ferrite.PointEvalHandler(grid, [Ferrite.Vec(0.15, 0.2), Ferrite.Vec(0.25, 0.2)])
    pfront, pback = Ferrite.evaluate_at_points(ph, dh, x, fp)
    return (cD = scale * F[1], cL = scale * F[2], Δp = pfront - pback)
end
