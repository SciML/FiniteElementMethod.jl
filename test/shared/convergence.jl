# Error norms and observed rates for the convergence tests (spec §10.3–§10.5).
using Ferrite: Ferrite

"""
    l2_error(sol, field::Symbol, exact; k = lastindex(sol.t), qorder = 4)

L2 norm over the mesh of the finite-element field `field` at saved time `sol.t[k]` minus
`exact(t, x...)`, by Gauss quadrature of order `qorder` on every cell.
"""
function l2_error(sol, field::Symbol, exact; k = lastindex(sol.t), qorder = 4)
    dh = sol.disc_data.space.dh
    grid = Ferrite.get_grid(dh)
    sdh = only(dh.subdofhandlers)
    ip = Ferrite.getfieldinterpolation(sdh, field)
    ipg = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.getrefshape(ip)}(qorder), ip, ipg)
    dofs = Ferrite.dof_range(sdh, field)
    u, t = sol.original_sol.u[k], sol.t[k]
    err2 = 0.0
    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        u_elem = u[Ferrite.celldofs(cell)[dofs]]
        coords = Ferrite.getcoordinates(cell)
        for q in 1:Ferrite.getnquadpoints(cv)
            xq = Ferrite.spatial_coordinate(cv, q, coords)
            err2 += (Ferrite.function_value(cv, q, u_elem) - exact(t, xq...))^2 * Ferrite.getdetJdV(cv, q)
        end
    end
    return sqrt(err2)
end

"Observed rates `log(e[i-1] / e[i]) / log(h[i-1] / h[i])` between successive refinements."
rates(errs, hs) = [log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i]) for i in 2:length(errs)]
