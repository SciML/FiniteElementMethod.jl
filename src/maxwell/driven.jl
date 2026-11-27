# Driven single-frequency solve of a MaxwellModel and evaluation of its solution.

# The current density as a function of (x, cell): `J(x)` for a function, for a dictionary the sum of the constants
# of the named cell sets that hold the cell (zero in no set).
_current_density(J, grid) = (x, cell) -> J(x)
function _current_density(J::AbstractDict, grid)
    per_cell = fill(zero(Ferrite.Vec{3, ComplexF64}), Ferrite.getncells(grid))
    for (name, value) in J
        _check_cellset(grid, String(name))
        Jset = Ferrite.Vec{3, ComplexF64}(Tuple(value))
        for cell in Ferrite.getcellset(grid, String(name))
            per_cell[cell] += Jset
        end
    end
    return (x, cell) -> per_cell[cell]
end

"""
    maxwell_rhs(model::MaxwellModel, ω, J)

The load vector `b = -iωμ0 L² ∫ J ⋅ Nᵢ` of [`MaxwellModel`](@ref) for the current density `J` in A/m², without
the PEC constraints. `J` is either a function of the mesh coordinate `x::Vec{3}` returning a 3-vector, or a
dictionary from cell set names to constant 3-vectors (zero outside the named sets); currents of overlapping sets
superpose on their common cells. `J` may itself assemble on `model`: each call has its own `CellValues`.
"""
function maxwell_rhs(model::MaxwellModel, ω::Number, J)
    ω = _frequency(ω)
    current = _current_density(J, model.grid)
    cv = _values_copy(model.cellvalues)
    n = Ferrite.getnbasefunctions(cv)
    b = zeros(ComplexF64, Ferrite.ndofs(model.dh))
    be = zeros(ComplexF64, n)
    scale = -im * ω * model.mu0 * model.length_scale^2
    for cc in Ferrite.CellIterator(model.dh)
        Ferrite.reinit!(cv, cc)
        fill!(be, 0)
        x = Ferrite.getcoordinates(cc)
        for q in 1:Ferrite.getnquadpoints(cv)
            Jq = Ferrite.Vec{3, ComplexF64}(Tuple(current(Ferrite.spatial_coordinate(cv, q, x), Ferrite.cellid(cc))))
            dΩ = Ferrite.getdetJdV(cv, q)
            for i in 1:n
                be[i] += scale * dot(Ferrite.shape_value(cv, q, i), Jq) * dΩ
            end
        end
        Ferrite.assemble!(b, Ferrite.celldofs(cc), be)
    end
    return b
end

"""
    driven_problem(model::MaxwellModel, ω, J)

The SciMLBase `LinearProblem` `A(ω) u = b` of [`MaxwellModel`](@ref) at angular frequency `ω` (rad/s) with
current density `J` (as for [`FiniteElementMethod.maxwell_rhs`](@ref)) and the PEC constraints applied by
Ferrite's `apply!`: the constrained rows and columns are zeroed with a nonzero diagonal. PEC elimination
preserves symmetry when the input operator is symmetric. Its solution `u` are the edge dofs of `E`; wrap them with
`MaxwellSolution(model, ω, u)` to evaluate the field.
"""
function driven_problem(model::MaxwellModel, ω::Real, J)
    ω = _frequency(ω)
    A = maxwell_operator(model, ω)
    b = maxwell_rhs(model, ω, J)
    Ferrite.apply!(A, b, model.ch)
    return SciMLBase.LinearProblem(A, b)
end

"""
    MaxwellSolution(model, omega, u)

The electric field of a [`MaxwellModel`](@ref) at angular frequency `omega`: `u` is the complex vector of
`Nedelec` edge dofs in the order of `model.dh`. Evaluate it at points with [`field_values`](@ref).
"""
struct MaxwellSolution{M <: MaxwellModel}
    model::M
    omega::Float64
    u::Vector{ComplexF64}
end

"""
    solve_driven(model::MaxwellModel, ω, J)

Solve [`driven_problem`](@ref) with a sparse direct factorisation and return the [`MaxwellSolution`](@ref).
"""
function solve_driven(model::MaxwellModel, ω::Real, J)
    ω = _frequency(ω)
    prob = driven_problem(model, ω, J)
    return MaxwellSolution(model, ω, prob.A \ prob.b)
end

"""
    field_values(sol::MaxwellSolution, points)

`(; E, curlE)` at the mesh-coordinate `points` (`Vec{3}`s): vectors of `Vec{3, ComplexF64}`, with `curlE` in
SI units (per metre, not per mesh unit). Each point is evaluated in a tetrahedron that contains it, up to a
barycentric roundoff of `1e-12`; points outside the mesh get `NaN`s. A point on a facet shared by two cells takes
the trace of one of them, so a normal component that jumps there has either one-sided value.
"""
function field_values(sol::MaxwellSolution, points::AbstractVector{<:Ferrite.Vec{3}})
    (; model, u) = sol
    grid = model.grid
    ph = _locate(grid, points)
    for (k, cell) in pairs(ph.cells)
        (cell === nothing || _barycentric_min(ph.local_coords[k]) >= -_TET_ROUNDOFF) && continue
        ph.cells[k], ph.local_coords[k] = _containing_tet(grid, points[k])
    end
    pv = Ferrite.PointValues(_NEDELEC)
    nan = Ferrite.Vec{3, ComplexF64}((NaN, NaN, NaN))
    E = fill(nan, length(points))
    curlE = fill(nan, length(points))
    for (k, cell) in pairs(ph.cells)
        cell === nothing && continue
        Ferrite.reinit!(pv, Ferrite.getcells(grid, cell), Ferrite.getcoordinates(grid, cell), ph.local_coords[k])
        u_cell = u[Ferrite.celldofs(model.dh, cell)]
        E[k] = Ferrite.function_value(pv, 1, u_cell)
        curlE[k] = Ferrite.function_curl(pv, 1, u_cell) / model.length_scale
    end
    return (; E, curlE)
end

# Ferrite's point search accepts a cell that misses the point by up to its own tolerance, which picks the wrong side
# of a normal jump near a facet; the Maxwell point evaluation needs a cell that contains the point.
const _TET_ROUNDOFF = 1.0e-12

# The least barycentric coordinate of the reference point `ξ` of a tetrahedron: negative outside.
_barycentric_min(ξ) = min(1 - sum(ξ), minimum(ξ))

# The cell of the tetrahedral `grid` that contains `p` most deeply (largest least barycentric coordinate) and the
# reference coordinates of `p` in it; `nothing` if every cell misses `p` by more than roundoff.
function _containing_tet(grid, p)
    best, best_ξ, best_λ = nothing, zero(p), -Inf
    for cell in 1:Ferrite.getncells(grid)
        x = Ferrite.getcoordinates(grid, cell)
        J = Ferrite.Tensor{2, 3}((x[2] - x[1]..., x[3] - x[1]..., x[4] - x[1]...))
        ξ = J \ (p - x[1])
        λ = _barycentric_min(ξ)
        λ > best_λ && ((best, best_ξ, best_λ) = (cell, ξ, λ))
    end
    best_λ >= -_TET_ROUNDOFF || return nothing, zero(p)
    return best, best_ξ
end
