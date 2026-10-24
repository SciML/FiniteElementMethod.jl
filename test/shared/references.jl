# Hand-written Ferrite references and end-to-end helpers for the Integration group, ported from the verified
# dauntless prototypes (test_linear.jl `reference`, test_nonlinear.jl and test_multifield.jl `residual_ref`).
# The references never see a symbolic expression: coefficients are plain Julia closures of the quadrature
# point, so agreement with the generated chain is an independent check.
#
# Sign convention, shared with the package (M u̇ + K u − F + r(u) = 0): a single-field problem
# ∂ₜ(m u) = ∇·(κ∇u) − β·∇u − c u + f with natural data κ∇u·n = g and Robin data κ∇u·n + a u = g gives
# K = ∫ κ∇u·∇v + (β·∇u) v + c u v + ∮ a u v,  M = ∫ m u v,  F = ∫ f v + ∮ g v.

using FiniteElementMethod: weak_form, emit_kernels, emit_dirichlet, AssemblyContext, assemble_linear,
    assemble_nonlinear
using Ferrite
using LinearAlgebra
using Random
using Symbolics
using Test

include(joinpath(@__DIR__, "strongform_seam.jl"))
using .StrongFormSeam: analyse, BC, Weighted, ProductRule, PeriodicBC

relerr(A, B) = norm(A - B) / max(norm(B), eps())

const SHAPES = Dict(Line => RefLine, Triangle => RefTriangle, Quadrilateral => RefQuadrilateral)
const FACETSETS = Dict(1 => ("left", "right"), 2 => ("left", "right", "bottom", "top"))

#     mesh_setup(celltype, order, nel; fields = [:u], corners = ())
#
# A Ferrite grid (`generate_grid`, so `[−1, 1]^dim` unless `corners` is given), a `DofHandler` with one Lagrange
# field of `order` per name in `fields`, quadrature of order `2·order` (spec §5.6), and the `AssemblyContext` with
# every facet set of the grid as a piece keyed by its own name.
function mesh_setup(celltype, order, nel; fields = [:u], corners = ())
    dim = celltype === Line ? 1 : 2
    grid = generate_grid(celltype, ntuple(_ -> nel, dim), corners...)
    shape = SHAPES[celltype]
    ip = Lagrange{shape, order}()
    dh = DofHandler(grid)
    for f in fields
        add!(dh, f, ip)
    end
    close!(dh)
    cv = CellValues(QuadratureRule{shape}(2order), ip)
    fv = FacetValues(FacetQuadratureRule{shape}(2order), ip)
    pieces = Dict(s => s for s in FACETSETS[dim])
    ctx = AssemblyContext(dh, fields, cv, fv, pieces)
    return (; grid, dh, cv, fv, ctx, fields, dim, order)
end

#     chain(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict())
#
# The whole M1 chain: `analyse` (the seam), `weak_form`, `emit_kernels` and `emit_dirichlet`.
function chain(eqs, bcs; ibp = Dict(), kwargs...)
    an = analyse(eqs, bcs; ibp, kwargs...)
    wf = weak_form(an)
    return (; an, wf, kernels = emit_kernels(wf), dirichlet = emit_dirichlet(an))
end

# `ConstraintHandler` from the chain's `emit_dirichlet` output, with each piece read as a facet-set name.
function constraints(setup, dirichlet, p; t = 0.0)
    ch = ConstraintHandler(setup.dh)
    for (piece, field, g) in dirichlet
        add!(ch, Dirichlet(setup.fields[field], getfacetset(setup.grid, piece), (X, tt) -> g(X, tt, p)))
    end
    close!(ch)
    update!(ch, t)
    return ch
end

# Hand-written `ConstraintHandler` from `(facet set, field, g(X, t))` triples.
function constraints_reference(setup, data; t = 0.0)
    ch = ConstraintHandler(setup.dh)
    for (set, field, g) in data
        add!(ch, Dirichlet(field, getfacetset(setup.grid, set), g))
    end
    close!(ch)
    update!(ch, t)
    return ch
end

# Prescribed values as a full vector, the public route of spec §6.4.
dirichlet_values(ch) = apply!(zeros(ndofs(ch.dh)), ch)

zero_block(dim) = (; κ = zero(Tensor{2, dim}), β = zero(Vec{dim}), c = 0.0, m = 0.0)

#     linear_reference(setup; block, source, facets = ())
#
# Hand-written `K`, `M`, `F` for `nf` fields. `block(e, b, X)` returns the named tuple `(; κ, β, c, m)` coupling
# equation `e` to field `b` (start from `zero_block(dim)`); `source(e, X)` is `f`. Each facet entry is
# `(set, eq, field, a, g)` adding `∮ a(X) u_field v` to `K` and `∮ g(X) v` to `F`.
function linear_reference(setup; block, source, facets = ())
    (; dh, cv, fv, grid, fields) = setup
    nf = length(fields)
    ranges = [dof_range(dh, f) for f in fields]
    K, M = allocate_matrix(dh), allocate_matrix(dh)
    F = zeros(ndofs(dh))
    nb = getnbasefunctions(cv)
    Ke, Me, Fe = zeros(nf * nb, nf * nb), zeros(nf * nb, nf * nb), zeros(nf * nb)
    aK, aM = start_assemble(K, F), start_assemble(M)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(Ke, 0); fill!(Me, 0); fill!(Fe, 0)
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv), e in 1:nf
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            for i in 1:nb
                v, ∇v = shape_value(cv, q, i), shape_gradient(cv, q, i)
                Fe[ranges[e][i]] += source(e, X) * v * dΩ
                for b in 1:nf
                    (; κ, β, c, m) = block(e, b, X)
                    for j in 1:nb
                        w, ∇w = shape_value(cv, q, j), shape_gradient(cv, q, j)
                        Ke[ranges[e][i], ranges[b][j]] += (∇v ⋅ (κ ⋅ ∇w) + (β ⋅ ∇w) * v + c * w * v) * dΩ
                        Me[ranges[e][i], ranges[b][j]] += m * w * v * dΩ
                    end
                end
            end
        end
        assemble!(aK, celldofs(cell), Ke, Fe)
        assemble!(aM, celldofs(cell), Me)
    end
    for (set, e, b, a, g) in facets
        aR = start_assemble(K, F; fillzero = false)
        for fc in FacetIterator(dh, getfacetset(grid, set))
            Ferrite.reinit!(fv, fc)
            fill!(Ke, 0); fill!(Fe, 0)
            xs = getcoordinates(fc)
            for q in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q)
                X = spatial_coordinate(fv, q, xs)
                for i in 1:nb
                    v = shape_value(fv, q, i)
                    Fe[ranges[e][i]] += g(X) * v * dΓ
                    for j in 1:nb
                        Ke[ranges[e][i], ranges[b][j]] += a(X) * shape_value(fv, q, j) * v * dΓ
                    end
                end
            end
            assemble!(aR, celldofs(fc), Ke, Fe)
        end
    end
    return K, M, F
end

#     residual_reference(setup, u; flux, source, facets = ())
#
# Hand-written steady residual `r_e,i = ∫ flux(e, U, ∇U, X)·∇φ_i + ∫ source(e, U, ∇U, X) φ_i + Σ ∮ g(U, X) φ_i`,
# where `U` and `∇U` are tuples over the fields at the quadrature point and each facet entry is `(set, eq, g)`.
function residual_reference(setup, u; flux, source, facets = ())
    (; dh, cv, fv, grid, fields) = setup
    nf = length(fields)
    ranges = [dof_range(dh, f) for f in fields]
    r = zeros(ndofs(dh))
    nb = getnbasefunctions(cv)
    r_cell = zeros(nf * nb)
    state(vals, q, u_cell) = (
        ntuple(f -> function_value(vals, q, u_cell[ranges[f]]), nf),
        ntuple(f -> function_gradient(vals, q, u_cell[ranges[f]]), nf),
    )
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        fill!(r_cell, 0)
        u_cell = u[celldofs(cell)]
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv), e in 1:nf
            dΩ = getdetJdV(cv, q)
            X = spatial_coordinate(cv, q, xs)
            Uq, ∇Uq = state(cv, q, u_cell)
            for i in 1:nb
                r_cell[ranges[e][i]] += (
                    flux(e, Uq, ∇Uq, X) ⋅ shape_gradient(cv, q, i) +
                        source(e, Uq, ∇Uq, X) * shape_value(cv, q, i)
                ) * dΩ
            end
        end
        assemble!(r, celldofs(cell), r_cell)
    end
    for (set, e, g) in facets
        for fc in FacetIterator(dh, getfacetset(grid, set))
            Ferrite.reinit!(fv, fc)
            fill!(r_cell, 0)
            u_cell = u[celldofs(fc)]
            xs = getcoordinates(fc)
            for q in 1:getnquadpoints(fv)
                X = spatial_coordinate(fv, q, xs)
                Uq, _ = state(fv, q, u_cell)
                for i in 1:nb
                    r_cell[ranges[e][i]] += g(Uq, X) * shape_value(fv, q, i) * getdetJdV(fv, q)
                end
            end
            assemble!(r, celldofs(fc), r_cell)
        end
    end
    return r
end

# Steady residual `K u − F + r(u)` and Jacobian `K + J` of the generated operators.
function chain_residual(setup, kernels, u, p; t = 0.0)
    L = assemble_linear(setup.ctx, kernels, p; t)
    N = assemble_nonlinear(setup.ctx, kernels, u, p; t)
    return L.K * u - L.F + N.r, L.K + N.J
end

# Relative error of the Jacobian `J` of `R` against central differences along a random direction.
function tangent_error(R, J, u; ε = 1.0e-6, rng = Xoshiro(3))
    δ = randn(rng, length(u))
    fd = (R(u + ε * δ) - R(u - ε * δ)) / 2ε
    return norm(J * δ - fd) / norm(fd)
end

# Newton on `residual(u) -> (R, J)` with Dirichlet rows from `ch`; returns `u` and the history of `‖R‖`.
function newton(residual, ch, u; tol = 1.0e-11, maxiter = 25)
    u = copy(u)
    apply!(u, ch)
    history = Float64[]
    for _ in 1:maxiter
        R, J = residual(u)
        apply_zero!(J, R, ch)
        push!(history, norm(R))
        history[end] < tol && break
        u .-= J \ R
    end
    return u, history
end

# Estimated orders `log(r₊/r)/log(r/r₋)` over consecutive Newton residuals above `floor`, the round-off level
# where the order estimate stops being meaningful.
function newton_orders(history; floor = 1.0e-12)
    h = filter(>(floor), history)
    return [log(h[k + 1] / h[k]) / log(h[k] / h[k - 1]) for k in 2:(length(h) - 1)]
end

# Linear steady solve `K u = F` with Dirichlet rows condensed by `ch`.
function linear_solve(setup, kernels, ch, p)
    A = assemble_linear(setup.ctx, kernels, p)
    K, F = copy(A.K), copy(A.F)
    apply!(K, F, ch)
    u = K \ F
    apply!(u, ch)
    return u
end

# Maximum nodal error of field `field` of `u` against `exact(X)`.
function nodal_error(setup, u, exact; field = :u)
    nodal = evaluate_at_grid_nodes(setup.dh, u, field)
    return maximum(abs(nodal[i] - exact(node.x)) for (i, node) in enumerate(setup.grid.nodes))
end

# L2 error of a single-field `u` against `exact(X)` with quadrature of order `2·order + 2`.
function l2_error(setup, u, exact)
    (; dh, order, grid) = setup
    shape = SHAPES[getcelltype(grid)]
    cv = CellValues(QuadratureRule{shape}(2order + 2), Lagrange{shape, order}())
    err = 0.0
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        u_cell = u[celldofs(cell)]
        xs = getcoordinates(cell)
        for q in 1:getnquadpoints(cv)
            err += (function_value(cv, q, u_cell) - exact(spatial_coordinate(cv, q, xs)))^2 * getdetJdV(cv, q)
        end
    end
    return sqrt(err)
end

rates(errs) = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]

# Numeric function of a symbolic expression in the spatial variables `sivs`, called as `f(X)`.
function numeric(ex, sivs)
    f = Symbolics.build_function(ex, sivs...; expression = Val(false))
    return X -> f(ntuple(d -> X[d], length(sivs))...)
end

# `true` when `f()` throws an `ErrorException` whose message contains `needle`.
function raises(f, needle)
    try
        f()
    catch err
        return err isa ErrorException && occursin(needle, err.msg)
    end
    return false
end
