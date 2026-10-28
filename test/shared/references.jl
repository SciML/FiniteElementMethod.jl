# Hand-written Ferrite references and end-to-end helpers for the Integration group, ported from the verified
# dauntless prototypes (test_linear.jl `reference`, test_nonlinear.jl and test_multifield.jl `residual_ref`).
# The references never see a symbolic expression: coefficients are plain Julia closures of the quadrature
# point, so agreement with the generated chain is an independent check.
#
# Sign convention, shared with the package (M u̇ + K u − F + r(u) = 0): a single-field problem
# ∂ₜ(m u) = ∇·(κ∇u) − β·∇u − c u + f with natural data κ∇u·n = g and Robin data κ∇u·n + a u = g gives
# K = ∫ κ∇u·∇v + (β·∇u) v + c u v + ∮ a u v,  M = ∫ m u v,  F = ∫ f v + ∮ g v.

using FiniteElementMethod: weak_form, emit_kernels, emit_dirichlet, AssemblyContext, assemble_linear,
    assemble_nonlinear, Galerkin
using Ferrite
using LinearAlgebra
using Random
using Symbolics
using Test

include(joinpath(@__DIR__, "strongform_seam.jl"))
using .StrongFormSeam: StrongFormSeam, BC
using DomainSets: DomainSets
using ModelingToolkit: ModelingToolkit
using PDEBase: PDEBase
using SciMLBase: SciMLBase

# The analysis switch. With a PDEBase whose StrongForm has `analyse` (SciML/PDEBase.jl#149, 4f536e4, and later) the
# chain's analysis comes from PDEBase: the volume terms, and the boundary conditions on box faces, which the adapter
# below rewrites as traces. PDEBase numbers fields in the declared order and emits conditions in input order.
# With the whole M0 stack (MeshDomain, BoundaryPiece and NormalDerivative slotting, PDEBase#154), conditions on mesh
# pieces go through PDEBase too, written with `NormalDerivative` on a `MeshDomain` (the `Ω` keyword below); without
# it, interior-form conditions on a non-box piece fall back to the seam's stage d against PDEBase's fluxes. Released
# PDEBase has no StrongForm, and the whole analysis falls back to the seam. test/README.md says how to run the
# PDEBase paths locally.
const PDEBASE_PATH = isdefined(PDEBase, :StrongForm) && isdefined(PDEBase.StrongForm, :analyse)
const STRONGFORM = PDEBASE_PATH ? PDEBase.StrongForm : StrongFormSeam
const Weighted, ProductRule, PeriodicBC = STRONGFORM.Weighted, STRONGFORM.ProductRule, STRONGFORM.PeriodicBC

# `@test` on the PDEBase path and `@test_broken` on the seam path, for checks only PDEBase's analyse passes.
macro test_pdebase(ex)
    return esc(:(PDEBASE_PATH ? @test($ex) : @test_broken($ex)))
end

const PDEBASE_MESH = PDEBASE_PATH && isdefined(PDEBase, :MeshDomain) && isdefined(PDEBase, :NormalDerivative)

# `@test` with PDEBase's mesh pieces and `@test_broken` without them.
macro test_pdebase_mesh(ex)
    return esc(:(PDEBASE_MESH ? @test($ex) : @test_broken($ex)))
end

if PDEBASE_PATH
    struct TimeDiscretization <: PDEBase.AbstractEquationSystemDiscretization
        time::Any
    end
    PDEBase.get_time(d::TimeDiscretization) = d.time
end

if PDEBASE_MESH
    # A Ferrite grid's facet sets as the pieces of a PDEBase `MeshDomain`.
    struct FacetSetMesh
        dim::Int
        names::Vector{Symbol}
    end
    PDEBase.meshdim(m::FacetSetMesh) = m.dim
    PDEBase.facetset_names(m::FacetSetMesh) = Tuple(m.names)

    struct MeshDiscretization <: PDEBase.AbstractMeshDiscretization
        time::Any
        captured::Base.RefValue{Any}
    end
    PDEBase.get_time(d::MeshDiscretization) = d.time
    # Without `parsed_boundary_map` (PDEBase#155), stop `symbolic_discretize` once the boundary map is parsed.
    if !isdefined(PDEBase, :parsed_boundary_map)
        struct BoundaryMapReady <: Exception end
        function PDEBase.check_boundarymap(bmap, v::PDEBase.VariableMap, d::MeshDiscretization)
            d.captured[] = (bmap, v)
            throw(BoundaryMapReady())
        end
    end
end

# The parsed boundary map and variable map of a mesh system, with the system analyse should read.
function parsed_mesh_system(sys, disc)
    isdefined(PDEBase, :parsed_boundary_map) && return PDEBase.parsed_boundary_map(sys, disc)
    try
        SciMLBase.symbolic_discretize(sys, disc)
    catch err
        err isa BoundaryMapReady || rethrow()
    end
    boundarymap, v = disc.captured[]
    return (; boundarymap, v, pdesys = sys)
end

# The `MeshDomain` whose pieces are `names` (default: the grid's facet sets), for the `Ω` keyword of `analyse`.
mesh_domain(setup; names = collect(keys(setup.grid.facetsets))) =
    PDEBase.MeshDomain(FacetSetMesh(setup.dim, Symbol.(names)))

# PDEBase's analysis of a system on the mesh domain `Ω`, with `bcs` raw equations on its pieces; boundary data is
# re-keyed from each `BoundaryPiece` to its facet-set name.
function mesh_analyse(eqs, bcs; Ω, depvars, t, sivs, params = [], ibp = Dict())
    SF = PDEBase.StrongForm
    domains = [Tuple(sivs) ∈ Ω, t ∈ DomainSets.ClosedInterval(0.0, 1.0)]
    sys = ModelingToolkit.PDESystem(eqs, bcs, domains, [t; sivs], depvars, params; name = :sys)
    parsed = parsed_mesh_system(sys, MeshDiscretization(t, Ref{Any}(nothing)))
    an = SF.analyse(parsed.pdesys, parsed.v, parsed.boundarymap; ibp)
    name(piece) = string(piece.name)
    return SF.StrongFormAnalysis{Any}(
        an.slots, an.nfields, an.dim, an.strong, an.mass, an.flux, an.remainder, an.conormal,
        SF.EssentialBC{Any}[SF.EssentialBC(name(b.piece), b.field, b.g) for b in an.essential],
        SF.NaturalBC{Any}[SF.NaturalBC(name(b.piece), b.eq, b.Φ) for b in an.natural], SF.PeriodicBC{Any}[]
    )
end

# Box faces of the Ferrite facet sets the tests name, as (spatial variable index, side).
const BOX_FACES = Dict("left" => (1, :lo), "right" => (1, :hi), "bottom" => (2, :lo), "top" => (2, :hi))

# `ex`, a condition written with interior unknowns, as a condition on the box face `(d, side)`: every unknown
# becomes its trace there, derivatives act on the trace, and the face coordinate takes the endpoint value.
function on_face(ex, depvars, sivs, d, endpoint)
    ex = Symbolics.unwrap(ex)
    trace(u) = (
        args = collect(Symbolics.arguments(u)); args[findfirst(a -> isequal(a, sivs[d]), args)] = endpoint;
        Symbolics.operation(u)(args...)
    )
    isequal(ex, Symbolics.unwrap(sivs[d])) && return endpoint
    Symbolics.iscall(ex) || return ex
    op = Symbolics.operation(ex)
    any(u -> isequal(ex, Symbolics.unwrap(u)), depvars) && return Symbolics.unwrap(trace(ex))
    op isa Symbolics.Differential && return Symbolics.unwrap(op(on_face(only(Symbolics.arguments(ex)), depvars, sivs, d, endpoint)))
    return Symbolics.unwrap(op(map(a -> on_face(a, depvars, sivs, d, endpoint), Symbolics.arguments(ex))...))
end

#     analyse(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict(), periodic = [], domain = nothing, Ω = nothing)
#
# The seam's interface over either path. `bcs` holds seam `BC`s and, on the PDEBase path, raw trace `Equation`s
# (periodic pairs). `domain` gives each spatial variable's interval, `[−1, 1]` by default as generate_grid's. With
# a `MeshDomain` `Ω` (PDEBase mesh path only), `bcs` are raw equations on its pieces.
function analyse(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict(), periodic = [], domain = nothing, Ω = nothing)
    Ω === nothing || return mesh_analyse(eqs, bcs; Ω, depvars, t, sivs, params, ibp)
    PDEBASE_PATH || return StrongFormSeam.analyse(eqs, bcs; depvars, t, sivs, params, ibp, periodic)
    SF = PDEBase.StrongForm
    intervals = domain === nothing ? [(-1.0, 1.0) for _ in sivs] : domain
    box = Symbolics.Equation[]
    pieces = Dict{Any, Any}()                       # BoxFace -> facet-set name
    mesh = BC[]
    for bc in bcs
        if bc isa Symbolics.Equation
            push!(box, bc)
        elseif haskey(BOX_FACES, bc.piece) && BOX_FACES[bc.piece][1] <= length(sivs)
            d, side = BOX_FACES[bc.piece]
            endpoint = side === :lo ? intervals[d][1] : intervals[d][2]
            push!(box, on_face(bc.eq.lhs, depvars, sivs, d, endpoint) ~ on_face(bc.eq.rhs, depvars, sivs, d, endpoint))
        else
            push!(mesh, bc)
        end
    end
    for (d, x) in enumerate(sivs), (side, name) in ((:lo, d == 1 ? "left" : "bottom"), (:hi, d == 1 ? "right" : "top"))
        pieces[SF.BoxFace(Symbolics.unwrap(x), side)] = name
    end
    ivs = [t; sivs]
    domains = [t ∈ DomainSets.ClosedInterval(0.0, 1.0); [x ∈ DomainSets.ClosedInterval(I...) for (x, I) in zip(sivs, intervals)]]
    sys = ModelingToolkit.PDESystem(eqs, box, domains, ivs, depvars, params; name = :sys)
    PDEBase.cardinalize_eqs!(sys)
    sys, replaced = PDEBase.make_pdesys_compatible(sys)
    v = PDEBase.VariableMap(sys, TimeDiscretization(t); replaced_vars = replaced)
    orders = Dict(x => PDEBase.d_orders(x, box) for x in PDEBase.all_ivs(v))
    an = SF.analyse(sys, v, PDEBase.parse_bcs(box, v, orders); ibp)
    essential = SF.EssentialBC{Any}[SF.EssentialBC(pieces[b.piece], b.field, b.g) for b in an.essential]
    natural = SF.NaturalBC{Any}[SF.NaturalBC(pieces[b.piece], b.eq, b.Φ) for b in an.natural]
    periodic_out = SF.PeriodicBC{Any}[SF.PeriodicBC(pieces[b.master], pieces[b.mirror], b.field, b.shift) for b in an.periodic]
    append!(periodic_out, periodic)
    ess, nat = StrongFormSeam.boundary_conditions(mesh, an.flux, an.slots; depvars, t, sivs, params)
    append!(essential, [SF.EssentialBC(b.piece, b.field, b.g) for b in ess])
    append!(natural, [SF.NaturalBC(b.piece, b.eq, b.Φ) for b in nat])
    return SF.StrongFormAnalysis{Any}(
        an.slots, an.nfields, an.dim, an.strong, an.mass, an.flux, an.remainder, an.conormal, essential, natural, periodic_out
    )
end

relerr(A, B) = norm(A - B) / max(norm(B), eps())

const P0 = [0.0]            # parameter vector of a problem with no parameters
const RIGHT = [1.0, 0.0]    # outward normal of the `right` facet set in 2D
iso(a) = a * one(Tensor{2, 2})

# Generated K, M, F against the hand-written reference, each to relative error `tol`.
function check_operators(A, Kr, Mr, Fr; tol = 1.0e-14)
    @test relerr(A.K, Kr) ≤ tol
    @test relerr(A.M, Mr) ≤ tol
    @test relerr(A.F, Fr) ≤ tol
    return nothing
end

# Global dofs of field `f`, sorted.
function field_dofs(setup, f)
    return sort(unique(reduce(vcat, [celldofs(c)[dof_range(setup.dh, f)] for c in CellIterator(setup.dh)])))
end

# Conditions of a manufactured solution `ustar` for the unknown `U`: Dirichlet `u*` on every facet set except
# `right`, and `∂ₓu = ∂ₓu*` on `right`, written as the user would (not the conormal), so the chain rescales it.
function manufactured_bcs(U, ustar, sivs)
    dim = length(sivs)
    dirichlet = [BC(set, U ~ ustar) for set in FACETSETS[dim] if set != "right"]
    neumann = BC("right", Differential(sivs[1])(U) ~ expand_derivatives(Differential(sivs[1])(ustar)), dim == 1 ? [1.0] : RIGHT)
    return [dirichlet; neumann]
end

const SHAPES = Dict(
    Line => RefLine, Triangle => RefTriangle, Quadrilateral => RefQuadrilateral,
    QuadraticTriangle => RefTriangle, QuadraticQuadrilateral => RefQuadrilateral,
)
const FACETSETS = Dict(1 => ("left", "right"), 2 => ("left", "right", "bottom", "top"))

#     mesh_setup(celltype, order, nel; fields = [:u], corners = (), hessians = false)
#
# A Ferrite grid (`generate_grid`, so `[−1, 1]^dim` unless `corners` is given), a `DofHandler` with one Lagrange
# field of `order` per name in `fields`, quadrature of order `2·order` (spec §5.6), and the `AssemblyContext` with
# every facet set of the grid as a piece keyed by its own name. `hessians = true` builds the `CellValues` with shape
# Hessians, which SUPG kernels need.
function mesh_setup(celltype, order, nel; fields = [:u], corners = (), hessians = false)
    dim = celltype === Line ? 1 : 2
    grid = generate_grid(celltype, ntuple(_ -> nel, dim), corners...)
    return grid_setup(grid, order, FACETSETS[dim]; fields, hessians)
end

#     grid_setup(grid, order, sets; fields = [:u], hessians = false)
#
# As `mesh_setup` for a given grid, with the facet sets `sets` as pieces. The values use the grid's own geometric
# interpolation (spec §6.2), so a grid of quadratic cells is mapped isoparametrically.
function grid_setup(grid, order, sets; fields = [:u], hessians = false)
    celltype = getcelltype(grid)
    dim = Ferrite.getspatialdim(grid)
    shape = SHAPES[celltype]
    ip, ip_geo = Lagrange{shape, order}(), Ferrite.geometric_interpolation(celltype)
    dh = DofHandler(grid)
    for f in fields
        add!(dh, f, ip)
    end
    close!(dh)
    cv = CellValues(QuadratureRule{shape}(2order), ip, ip_geo; update_hessians = hessians)
    fv = FacetValues(FacetQuadratureRule{shape}(2order), ip, ip_geo)
    ctx = AssemblyContext(dh, fields, cv, fv, Dict(s => s for s in sets))
    return (; grid, dh, cv, fv, ctx, fields, dim, order)
end

#     quarter_annulus(n; quadratic)
#
# The quarter annulus 1 ≤ r ≤ 2, 0 ≤ θ ≤ π/2 meshed by n × n cells of the (r, θ) rectangle, each split into two
# triangles. Every node, mid-side nodes included, sits at its (r, θ) position, so with `quadratic = true`
# (`QuadraticTriangle`) the edges on r = 1 and r = 2 are arcs of the circles; with `quadratic = false` the same
# vertices make straight-sided `Triangle`s. Facet sets: `inner`, `outer`, `bottom` (y = 0) and `left` (x = 0).
function quarter_annulus(n; quadratic)
    m = 2n + 1
    node(i, j) = i + 1 + j * m
    nodes = [Node(Vec((1 + i / 2n) * cos(pi / 2 * j / 2n), (1 + i / 2n) * sin(pi / 2 * j / 2n))) for j in 0:(2n) for i in 0:(2n)]
    cells = quadratic ? QuadraticTriangle[] : Triangle[]
    for a in 0:(n - 1), b in 0:(n - 1)
        i, j = 2a, 2b
        # vertices, then mid-sides of edges 1-2, 2-3, 3-1 (Ferrite's QuadraticTriangle order)
        lower = (node(i, j), node(i + 2, j), node(i + 2, j + 2), node(i + 1, j), node(i + 2, j + 1), node(i + 1, j + 1))
        upper = (node(i, j), node(i + 2, j + 2), node(i, j + 2), node(i + 1, j + 1), node(i + 1, j + 2), node(i, j + 1))
        for t in (lower, upper)
            push!(cells, quadratic ? QuadraticTriangle(t) : Triangle(t[1:3]))
        end
    end
    grid = Grid(cells, nodes)
    addfacetset!(grid, "inner", X -> norm(X) ≈ 1)
    addfacetset!(grid, "outer", X -> norm(X) ≈ 2)
    addfacetset!(grid, "bottom", X -> abs(X[2]) < 1.0e-12)
    addfacetset!(grid, "left", X -> abs(X[1]) < 1.0e-12)
    return grid
end

#     chain(eqs, bcs; depvars, t, sivs, params = [], ibp = Dict(), test_space = Galerkin())
#
# The whole M1 chain: `analyse` (the seam), `weak_form`, `emit_kernels` and `emit_dirichlet`.
function chain(eqs, bcs; ibp = Dict(), test_space = Galerkin(), kwargs...)
    an = analyse(eqs, bcs; ibp, kwargs...)
    wf = weak_form(an; test_space)
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

# Newton converged in at most `maxiter` steps and the last order estimate above round-off is at least `order`.
function newton_converged(history; order = 1.8, maxiter = 8)
    orders = newton_orders(history)
    return history[end] < 1.0e-11 && length(history) ≤ maxiter && length(orders) ≥ 1 && orders[end] ≥ order
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
    celltype = getcelltype(grid)
    shape = SHAPES[celltype]
    cv = CellValues(QuadratureRule{shape}(2order + 2), Lagrange{shape, order}(), Ferrite.geometric_interpolation(celltype))
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

# `true` when `f()` throws any exception whose displayed message contains `needle`, so the checks survive
# PDEBase raising an `ArgumentError` or its own exception type instead of the seam's `ErrorException`.
function raises(f, needle)
    try
        f()
    catch err
        return err isa Exception && occursin(needle, sprint(showerror, err))
    end
    return false
end
