# Ferrite implementations of the PDEBase mesh-domain interface (spec §3.4, §6.1). They act on a plain
# `Ferrite.Grid` so that `PDEBase.meshdim(::MeshDomain)` and friends can forward to them later.

_meshdim(::Ferrite.Grid{dim}) where {dim} = dim

_facetset_names(grid::Ferrite.Grid) = sort!(Symbol.(collect(keys(grid.facetsets))))

_cellset_names(grid::Ferrite.Grid) = sort!(Symbol.(collect(keys(grid.cellsets))))

function _contains_point(grid::Ferrite.Grid{dim}, x) where {dim}
    length(x) == dim ||
        throw(DimensionMismatch("point has $(length(x)) coordinates but the mesh is $(dim)-dimensional"))
    all(isfinite, x) || return false
    handler = Ferrite.PointEvalHandler(grid, [Ferrite.Vec{dim, Float64}(Tuple(x))]; warn = false)
    return handler.cells[1] !== nothing
end

# Ferrite's `generate_grid` face names, renamed to the DomainSets order of spec §3.2.
const _BOX_FACE_NAMES = Dict(
    1 => ("left" => "x_lo", "right" => "x_hi"),
    2 => ("bottom" => "bottom", "right" => "right", "top" => "top", "left" => "left"),
    3 => (
        "left" => "x_lo", "right" => "x_hi", "front" => "y_lo",
        "back" => "y_hi", "bottom" => "z_lo", "top" => "z_hi",
    ),
)

"""
    rectangle_grid(celltype, cells::NTuple{N, Int}; lo = zeros(N), hi = ones(N))

`Ferrite.generate_grid` on the box `[lo, hi]` (default the unit box, not Ferrite's `[-1, 1]^N`) with the
boundary facet sets named as DomainSets names the box faces: `"bottom"`, `"right"`, `"top"`, `"left"` in 2D
and `"x_lo"`, `"x_hi"`, `"y_lo"`, … in 1D and 3D. No other facet sets are kept.
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
