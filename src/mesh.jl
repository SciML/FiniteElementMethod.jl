# Ferrite implementations of the PDEBase mesh-domain interface (spec §3.4, §6.1). They act on a plain
# `Ferrite.Grid` so that `PDEBase.meshdim(::MeshDomain)` and friends can forward to them later.

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
    check_facetsets(grid, names)

Validate the facet sets that boundary conditions refer to (spec §4.6, §6.1): every name must be a facet set
of `grid`, no facet may lie in two of the named sets, and a set containing interior facets (shared by two
cells) draws a warning. Returns `nothing`.
"""
function check_facetsets(grid::Ferrite.Grid, names)
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
    for name in used, fi in Ferrite.getfacetset(grid, String(name))
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
        isempty(interior) || @warn "facet set $(repr(name)) contains $(length(interior)) interior facets " *
            "(shared by two cells; first: cell $(interior[1][1]), local facet $(interior[1][2]))"
    end
    return nothing
end
