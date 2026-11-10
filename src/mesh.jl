# Ferrite implementations of the PDEBase mesh-domain interface (spec §3.4, §6.1).

_meshdim(::Ferrite.Grid{dim}) where {dim} = dim

_facetset_names(grid::Ferrite.Grid) = sort!(Symbol.(collect(keys(grid.facetsets))))

_cellset_names(grid::Ferrite.Grid) = sort!(Symbol.(collect(keys(grid.cellsets))))

function _contains_point(grid::Ferrite.Grid{dim}, x) where {dim}
    length(x) == dim ||
        throw(DimensionMismatch("point has $(length(x)) coordinates but the mesh is $(dim)-dimensional"))
    all(isfinite, x) || return false
    point = [Ferrite.Vec{dim, Float64}(Tuple(x))]
    Ferrite.PointEvalHandler(grid, point; warn = false).cells[1] === nothing || return true
    # The default search only visits cells around the 3 nearest nodes, which misses points on graded
    # meshes; retry over every node, after a bounding-box test that settles most outside points.
    all(1:dim) do d
        lo, hi = extrema(node -> node.x[d], grid.nodes)
        lo <= x[d] <= hi
    end || return false
    handler = Ferrite.PointEvalHandler(grid, point; warn = false, search_nneighbors = Ferrite.getnnodes(grid))
    return handler.cells[1] !== nothing
end

# Ferrite's `generate_grid` face names, renamed to the PDEBase `BoundaryPiece` names of a box.
const _BOX_FACE_NAMES = Dict(
    1 => ("left" => "x1_lo", "right" => "x1_hi"),
    2 => ("bottom" => "bottom", "right" => "right", "top" => "top", "left" => "left"),
    3 => (
        "left" => "x1_lo", "right" => "x1_hi", "front" => "x2_lo",
        "back" => "x2_hi", "bottom" => "x3_lo", "top" => "x3_hi",
    ),
)

"""
    rectangle_grid(celltype, cells::NTuple{N, Int}; lo = zeros(N), hi = ones(N))

`Ferrite.generate_grid` on the box `[lo, hi]` (default the unit box, not Ferrite's `[-1, 1]^N`) with the
boundary facet sets named as DomainSets names the box faces: `"bottom"`, `"right"`, `"top"`, `"left"` in 2D
and `"x1_lo"`, `"x1_hi"`, `"x2_lo"`, … in every other dimension. No other facet sets are kept.
"""
function rectangle_grid(
        celltype, cells::NTuple{N, Int};
        lo = ntuple(Returns(0.0), N), hi = ntuple(Returns(1.0), N)
    ) where {N}
    grid = Ferrite.generate_grid(
        celltype, cells, Ferrite.Vec{N, Float64}(Tuple(lo)), Ferrite.Vec{N, Float64}(Tuple(hi))
    )
    named = Dict(new => grid.facetsets[old] for (old, new) in _BOX_FACE_NAMES[N])
    empty!(grid.facetsets)
    merge!(grid.facetsets, named)
    return grid
end

_facet_key(fi::Ferrite.FacetIndex) = (fi[1], fi[2])

"""
    check_facetsets(grid, names; disjoint = true)

Validate the facet sets that boundary conditions refer to (spec §4.6, §6.1): every name must be a facet set
of `grid`, no facet may lie in two of the named sets (unless `disjoint = false`), and a set containing
interior facets (shared by two cells) is an error, since a boundary piece lies on the boundary. Returns
`nothing`.
"""
function check_facetsets(grid::Ferrite.Grid, names; disjoint::Bool = true)
    available = _facetset_names(grid)
    used = unique(Symbol.(names))
    for name in used
        name in available || throw(
            ArgumentError(
                "unknown facet set $(repr(name)); the mesh has facet sets $(join(repr.(available), ", "))"
            )
        )
    end
    owner = Dict{Ferrite.FacetIndex, Symbol}()
    for name in used, fi in (disjoint ? Ferrite.getfacetset(grid, String(name)) : ())
        first_owner = get!(owner, fi, name)
        first_owner === name && continue
        shared = sort!(
            [f for f in Ferrite.getfacetset(grid, String(name)) if get(owner, f, name) === first_owner];
            by = _facet_key
        )
        throw(
            ArgumentError(
                "facet sets $(repr(first_owner)) and $(repr(name)) share $(length(shared)) facets " *
                    "(first: cell $(shared[1][1]), local facet $(shared[1][2])); boundary pieces must be disjoint"
            )
        )
    end
    isempty(used) && return nothing
    topology = Ferrite.ExclusiveTopology(grid)
    for name in used
        interior = sort!(
            [fi for fi in Ferrite.getfacetset(grid, String(name)) if !isempty(Ferrite.getneighborhood(topology, grid, fi))];
            by = _facet_key
        )
        isempty(interior) || throw(
            ArgumentError(
                "facet set $(repr(name)) is not on the boundary: it contains $(length(interior)) interior facets " *
                    "(shared by two cells; first: cell $(interior[1][1]), local facet $(interior[1][2]))"
            )
        )
    end
    return nothing
end

# Sorted corner-node ids of local facet `lf` of `cell`: the identity of a facet of a conforming mesh, whatever the
# cell order and orientation.
_facet_corners(cell, lf) = sort!([Ferrite.vertices(cell)[i] for i in Ferrite.reference_facets(cell)[lf]])

# Every facet of `grid`, keyed by its corner nodes: a boundary facet has one owner, an interior facet two.
function _facet_owners(grid::Ferrite.Grid)
    owners = Dict{Vector{Int}, Vector{Ferrite.FacetIndex}}()
    for (c, cell) in enumerate(Ferrite.getcells(grid)), lf in 1:Ferrite.nfacets(cell)
        push!(get!(owners, _facet_corners(cell, lf), Ferrite.FacetIndex[]), Ferrite.FacetIndex(c, lf))
    end
    return owners
end

_boundary_facets(grid::Ferrite.Grid) = Set(only(fs) for fs in values(_facet_owners(grid)) if length(fs) == 1)

# Map each node of `sub` to the parent node within `atol` of it (the nearest one), through a hash of `atol`-wide
# buckets, so that a point's candidates lie in the 3^dim buckets around its own.
function _match_nodes(grid::Ferrite.Grid{dim}, sub::Ferrite.Grid, atol) where {dim}
    # Relative to the parent's lowest corner, and clamped, so a translated mesh or a tiny `atol` cannot overflow `Int`.
    origin = ntuple(d -> minimum(n -> n.x[d], Ferrite.getnodes(grid)), dim)
    bucket(x) = ntuple(d -> floor(Int, clamp((x[d] - origin[d]) / atol, -1.0e15, 1.0e15)), dim)
    buckets = Dict{NTuple{dim, Int}, Vector{Int}}()
    for (i, node) in enumerate(Ferrite.getnodes(grid))
        push!(get!(buckets, bucket(node.x), Int[]), i)
    end
    offsets = Iterators.product(ntuple(Returns(-1:1), dim)...)
    return map(Ferrite.getnodes(sub)) do node
        b = bucket(node.x)
        best, bestdist = 0, Inf
        for off in offsets, i in get(buckets, b .+ off, ())
            dist = norm(Ferrite.get_node_coordinate(grid, i) - node.x)
            dist <= atol && dist < bestdist && ((best, bestdist) = (i, dist))
        end
        best
    end
end

function _submesh_facets(grid::Ferrite.Grid{dim}, name, sub::Ferrite.Grid, atol, owners) where {dim}
    _meshdim(sub) == dim || throw(
        ArgumentError(
            "boundary piece $(repr(name)): the submesh has $(_meshdim(sub))-dimensional coordinates but the parent mesh " *
                "has $(dim)-dimensional ones; give the submesh its nodes in the coordinates of the parent"
        )
    )
    cells = Ferrite.getcells(sub)
    isempty(cells) && throw(ArgumentError("boundary piece $(repr(name)): the submesh has no cells"))
    all(c -> Ferrite.getrefdim(c) == dim - 1, cells) || throw(
        ArgumentError(
            "boundary piece $(repr(name)): the cells of the submesh must have dimension $(dim - 1), the facets of the " *
                "$(dim)-dimensional parent mesh"
        )
    )
    atol = something(atol, 1.0e-8 * max(1.0, maximum(d -> maximum(n -> n.x[d], Ferrite.getnodes(grid)) - minimum(n -> n.x[d], Ferrite.getnodes(grid)), 1:dim)))
    parent = _match_nodes(grid, sub, atol)
    unmatched = findall(iszero, parent)
    isempty(unmatched) || throw(
        ArgumentError(
            "boundary piece $(repr(name)): $(length(unmatched)) of $(length(parent)) submesh nodes are not within " *
                "atol = $atol of a node of the parent mesh (first: submesh node $(unmatched[1]) at " *
                "$(Tuple(Ferrite.get_node_coordinate(sub, unmatched[1]))))"
        )
    )
    facets = valtype(grid.facetsets)()
    missing_cells, interior_cells = Int[], Int[]
    for (k, cell) in enumerate(cells)
        found = get(owners, sort!([parent[i] for i in Ferrite.vertices(cell)]), nothing)
        found === nothing && (push!(missing_cells, k); continue)
        length(found) == 1 || (push!(interior_cells, k); continue)
        push!(facets, only(found))
    end
    centre(k) = Tuple(sum(Ferrite.get_node_coordinate(sub, i) for i in Ferrite.vertices(cells[k])) / length(Ferrite.vertices(cells[k])))
    isempty(missing_cells) || throw(
        ArgumentError(
            "boundary piece $(repr(name)): $(length(missing_cells)) of $(length(cells)) submesh cells match no facet of " *
                "the parent mesh, whose corner nodes they share (first: submesh cell $(missing_cells[1]) centred at " *
                "$(centre(missing_cells[1])))"
        )
    )
    isempty(interior_cells) || throw(
        ArgumentError(
            "boundary piece $(repr(name)): $(length(interior_cells)) of $(length(cells)) submesh cells match an interior " *
                "facet of the parent mesh (shared by two cells), not a boundary facet (first: submesh cell " *
                "$(interior_cells[1]) centred at $(centre(interior_cells[1])))"
        )
    )
    return facets
end

"""
    ferrite_grid(mesh)

The `Ferrite.Grid` of `mesh`, a mesh from another Julia meshing package, for `MeshDomain` and
[`add_boundary_pieces!`](@ref). A package is supported by an extension that loads with it:

  - Meshes.jl (`using Meshes`): a mesh in Cartesian coordinates whose elements are segments, triangles,
    quadrangles, tetrahedra or hexahedra. The nodes are the vertices of the mesh in its length unit, without
    the unit. A cell with the other orientation, such as a clockwise triangle in 2D, has its nodes reordered,
    since Ferrite needs cells with a positive Jacobian.

A `Ferrite.Grid` is returned as it is.
"""
ferrite_grid(grid::Ferrite.Grid) = grid
function ferrite_grid(mesh)
    throw(ArgumentError("ferrite_grid converts a Ferrite.Grid, or a Meshes.jl mesh after `using Meshes`; got a $(nameof(typeof(mesh)))"))
end

"""
    add_boundary_pieces!(grid, name => submesh, ...; atol = nothing)
    add_boundary_pieces!(grid, pieces::AbstractDict; atol = nothing)

Add a boundary piece to the `Ferrite.Grid` `grid` for each pair `name => submesh`: a facet set called `name`
holding the facets of `grid` that the cells of `submesh` coincide with. `submesh` is a `Ferrite.Grid` of one
dimension less than `grid`, in the coordinates of `grid`: the boundary mesh that Gmsh gives for a physical
group, or any grid whose cells are facets of `grid`. It may also be a mesh that [`ferrite_grid`](@ref)
converts, such as a Meshes.jl mesh. Afterwards `boundarypiece(MeshDomain(grid), name)` is the piece, exactly
as for any other facet set; `grid` is modified, as by `Ferrite.addfacetset!`, and returns.

A submesh cell is matched to a facet of `grid` through the nodes: each node of the submesh is the node of `grid`
within `atol > 0` of it (by coordinates, since the two grids number their nodes independently; the default `atol`
is `1e-8` times the larger of `1` and the largest extent of `grid`), and the cell is the facet whose corner
nodes are those of the cell. Nothing is added, for any of the pieces, if a node has no node of `grid` within
`atol`, a cell matches no facet of `grid`, a cell matches an interior facet (shared by two cells) or `name` is
already a facet set of `grid`; each is an `ArgumentError` that says which.
"""
function add_boundary_pieces!(grid::Ferrite.Grid, pieces::Pair...; atol = nothing)
    atol === nothing || atol > 0 || throw(ArgumentError("atol must be positive, got $atol"))
    names = [String(first(p)) for p in pieces]
    allunique(names) || throw(ArgumentError("boundary piece names must be unique; got $(Symbol.(names))"))
    for name in names
        haskey(grid.facetsets, name) &&
            throw(ArgumentError("the mesh already has a facet set $(repr(Symbol(name))); pick another name for the boundary piece"))
    end
    owners = _facet_owners(grid)
    sets = [_submesh_facets(grid, Symbol(n), ferrite_grid(last(p)), atol, owners) for (n, p) in zip(names, pieces)]
    foreach((n, s) -> Ferrite.addfacetset!(grid, n, s), names, sets)
    return grid
end
add_boundary_pieces!(grid::Ferrite.Grid, pieces::AbstractDict; kwargs...) = add_boundary_pieces!(grid, pieces...; kwargs...)
