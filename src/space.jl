"""
    FEMSpace(grid, fields; dirichlet = (), quadrature = nothing, t0 = 0.0)

The Ferrite discrete space of a mesh discretization (spec §6.2-6.4). `fields` is an ordered collection of
`name => interpolation` pairs; after `renumber!(dh, DofOrder.FieldWise())` field `k` occupies a contiguous
block of the global dof vector, in declaration order. `dirichlet` is a collection of
`(field, facetset_name, g)` with `g(x, t)`. `quadrature` is the quadrature order shared by every field,
`2p` for the highest Lagrange order `p` by default (spec §5.6). The constraint handler is first
updated at `t0`, the initial time, since Dirichlet data need not be defined before it.

Fields: `grid`, `dh` (the `DofHandler`), `ch` (the closed `ConstraintHandler`), `fields` (names in
declaration order), `cellvalues` and `facetvalues` (`NamedTuple`s keyed by field), `ranges` (global dof
range per field), `pieces` (piece name `=>` facet set name, for every facet set of the grid) and `pattern`
(`allocate_matrix(dh, ch)`, to be copied for every operator).
"""
struct FEMSpace{G, DH, CH, CV <: NamedTuple, FV <: NamedTuple, R <: NamedTuple, A} <:
    PDEBase.AbstractDiscreteSpace
    grid::G
    dh::DH
    ch::CH
    fields::Vector{Symbol}
    cellvalues::CV
    facetvalues::FV
    ranges::R
    pieces::Dict{Symbol, String}
    pattern::A
    gbuf::Vector{Float64}
end

_order(::Ferrite.Interpolation{shape, order}) where {shape, order} = order

function FEMSpace(
        grid::Ferrite.Grid, fields; dirichlet = (), quadrature::Union{Nothing, Int} = nothing, t0::Real = 0.0
    )
    names = Symbol[first(f) for f in fields]
    ips = Tuple(last(f) for f in fields)
    allunique(names) || throw(ArgumentError("field names must be unique; got $(names)"))
    for (field, _, _) in dirichlet
        Symbol(field) in names ||
            throw(ArgumentError("Dirichlet condition on unknown field $(repr(Symbol(field))); fields are $(names)"))
    end
    check_facetsets(grid, [Symbol(set) for (_, set, _) in dirichlet])

    dh = Ferrite.DofHandler(grid)
    for (name, ip) in zip(names, ips)
        Ferrite.add!(dh, name, ip)
    end
    Ferrite.close!(dh)
    Ferrite.renumber!(dh, Ferrite.DofOrder.FieldWise())

    # `close!(ch)` calls `update!(ch, 0.0)` itself; until it returns, time-dependent data are evaluated at
    # `t0` instead. Time-independent `g(x)` is passed through for Ferrite's own arity detection.
    t0 = float(t0)
    closed = Ref(false)
    X = typeof(Ferrite.get_node_coordinate(grid, 1))
    ch = Ferrite.ConstraintHandler(dh)
    for (field, set, g) in dirichlet
        f = hasmethod(g, Tuple{X, typeof(t0)}) ? (x, t) -> g(x, closed[] ? t : t0) : g
        Ferrite.add!(ch, Ferrite.Dirichlet(Symbol(field), Ferrite.getfacetset(grid, String(set)), f))
    end
    Ferrite.close!(ch)
    closed[] = true
    Ferrite.update!(ch, t0)

    shape = Ferrite.getrefshape(first(ips))
    order = something(quadrature, 2 * maximum(_order, ips))
    ip_geo = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
    qr = Ferrite.QuadratureRule{shape}(order)
    fqr = Ferrite.FacetQuadratureRule{shape}(order)
    key = Tuple(names)
    cellvalues = NamedTuple{key}(map(ip -> Ferrite.CellValues(qr, ip, ip_geo), ips))
    facetvalues = NamedTuple{key}(map(ip -> Ferrite.FacetValues(fqr, ip, ip_geo), ips))
    ranges = NamedTuple{key}(_global_ranges(dh, names))
    pieces = Dict(Symbol(name) => name for name in keys(grid.facetsets))
    return FEMSpace(
        grid, dh, ch, names, cellvalues, facetvalues, ranges, pieces,
        Ferrite.allocate_matrix(dh, ch), zeros(Ferrite.ndofs(dh))
    )
end

# FieldWise numbering makes each field one block; the blocks tiling 1:ndofs in order proves contiguity.
function _global_ranges(dh, names)
    ranges = map(Tuple(names)) do name
        local_range = Ferrite.dof_range(dh, name)
        lo, hi = typemax(Int), 0
        for cell in 1:Ferrite.getncells(dh.grid)
            for dof in view(Ferrite.celldofs(dh, cell), local_range)
                lo, hi = min(lo, dof), max(hi, dof)
            end
        end
        lo:hi
    end
    stops = (0, map(last, ranges)...)
    all(first(r) == stops[k] + 1 for (k, r) in enumerate(ranges)) && last(stops) == Ferrite.ndofs(dh) ||
        error("field-wise renumbering did not give contiguous blocks in declaration order: $(ranges)")
    return ranges
end

Ferrite.ndofs(space::FEMSpace) = Ferrite.ndofs(space.dh)

"""
    field_range(space::FEMSpace, name)

Global dof indices of field `name`, a contiguous `UnitRange`.
"""
function field_range(space::FEMSpace, name)
    haskey(space.ranges, Symbol(name)) ||
        throw(ArgumentError("unknown field $(repr(Symbol(name))); fields are $(space.fields)"))
    return space.ranges[Symbol(name)]
end

"""
    prescribed_dofs(space::FEMSpace)

Global indices of the Dirichlet dofs, in the `ConstraintHandler`'s order.
"""
prescribed_dofs(space::FEMSpace) = space.ch.prescribed_dofs

"""
    dirichlet_values!(vals, space::FEMSpace, t)

Fill `vals[i]` with the Dirichlet value at time `t` of dof `prescribed_dofs(space)[i]` and return `vals`.
"""
function dirichlet_values!(vals, space::FEMSpace, t)
    Ferrite.update!(space.ch, t)
    fill!(space.gbuf, 0)
    Ferrite.apply!(space.gbuf, space.ch)
    vals .= view(space.gbuf, prescribed_dofs(space))
    return vals
end
