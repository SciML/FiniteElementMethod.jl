# Periodic identification of boundary pieces. The dofs of one field on two pieces that are translates of each other
# are matched by coordinate; every connected class of matched dofs (a corner of a fully periodic box joins four
# of them) keeps one master and ties the others to it, `u_mirror = u_master`. The assembled operators keep all dofs
# as unknowns: `apply_ties!` adds each tied row to its master's row and replaces the tied row by `u_mirror - u_master`.

# Coordinates of the dofs of `field` on the facet set `set`.
function _piece_dofs(dh, field, grid, set)
    ch = Ferrite.ConstraintHandler(dh)
    Ferrite.add!(ch, Ferrite.Dirichlet(field, Ferrite.getfacetset(grid, String(set)), (x, t) -> 0.0))
    Ferrite.close!(ch)
    return sort(ch.prescribed_dofs)
end

# Points are matched in coordinates scaled by a per-axis tolerance, so a match is a distance of at most 1, on a grid of
# cells of side 1 looking in the neighbouring cells too. The scaled coordinates are relative to the corner of the pair's cells,
# from whose extent the tolerance comes, so they are at most about 1e6 in size; the clamp keeps any other point
# (a facet point of a degenerate mesh) from overflowing the integer keys, and such a point matches nothing.
_cell_key(point) = Tuple(floor.(Int, clamp.(point, -1.0e15, 1.0e15)))

# Nearest points of `to` to `from[d] + shift` (in scaled coordinates), as `from_dof => to_dof`, and the `from` dofs with no
# point of `to` within distance 1.
function _match_translate(scaled, from, to, shift)
    index = Dict{Any, Vector{Int}}()
    for d in to
        push!(get!(index, _cell_key(scaled[d]), Int[]), d)
    end
    dim = length(shift)
    matched = Pair{Int, Int}[]
    unmatched = Int[]
    for d in from
        target = scaled[d] + shift
        key = _cell_key(target)
        best, bestdist = 0, Inf
        for offset in Iterators.product(ntuple(_ -> -1:1, dim)...), e in get(index, key .+ offset, Int[])
            dist = norm(scaled[e] - target)
            dist < bestdist && ((best, bestdist) = (e, dist))
        end
        bestdist <= 1 ? push!(matched, d => best) : push!(unmatched, d)
    end
    return matched, unmatched
end

# Why the matching of `da` onto `db` is not a one-to-one correspondence, or `nothing`: every node of Γ_a needs a node of
# Γ_b within the tolerance, no node of Γ_b may be the target of two, and the nearest node of Γ_a to each node of Γ_b must be
# its own preimage.
function _matching_problem(coords, scaled, da, db, matched, unmatched, shift, sshift, a, b)
    isempty(unmatched) || return "the node at $(_tuple_str(coords[first(unmatched)])) on Γ_$a has no node of Γ_$b at " *
        "$(_tuple_str(coords[first(unmatched)] + shift)) ($(length(unmatched)) such nodes of Γ_$a, $(length(da)) on Γ_$a and $(length(db)) on Γ_$b)"
    owner = Dict{Int, Int}()
    for (x, y) in matched
        haskey(owner, y) && return "the nodes at $(_tuple_str(coords[owner[y]])) and $(_tuple_str(coords[x])) on Γ_$a both match the node at " *
            "$(_tuple_str(coords[y])) on Γ_$b, at distances $(round(norm(coords[owner[y]] + shift - coords[y]); sigdigits = 3)) and " *
            "$(round(norm(coords[x] + shift - coords[y]); sigdigits = 3))"
        owner[y] = x
    end
    missed = [y for y in db if !haskey(owner, y)]
    isempty(missed) || return "the node at $(_tuple_str(coords[first(missed)])) on Γ_$b has no node of Γ_$a ($(length(da)) on Γ_$a and $(length(db)) on Γ_$b)"
    back, _ = _match_translate(scaled, db, da, -sshift)
    for (y, x) in back
        owner[y] == x && continue
        return "the node at $(_tuple_str(coords[y])) on Γ_$b is nearest to the node at $(_tuple_str(coords[x])) on Γ_$a, not to its match at " *
            "$(_tuple_str(coords[owner[y]]))"
    end
    return nothing
end

#     _pair_frame(grid, dh, field, ip, cells) -> (lo, tolerance, scaled, local_grid)
#
# The frame in which a periodic pair is matched: `cells` are the cells of `field` (interpolation `ip`) with a facet on either
# piece, `lo` the corner of their bounding box and `tolerance` 1e-6 times its extent along each axis, so a thin mesh keeps its
# translations. `local_grid` is those cells alone, in their order, moved by `-lo`, and `scaled[d]` the position on it of each
# dof `d` of `field` on the cells, divided by `tolerance`: interpolated there, neither the distance of the cells from the origin
# nor any other cell of the mesh rounds a position.
function _pair_frame(grid, dh, field, ip, cells)
    ids = unique(n for c in cells for n in Ferrite.getcells(grid, c).nodes)
    x = [Ferrite.get_node_coordinate(grid, n) for n in ids]
    lo = Ferrite.Vec(ntuple(k -> minimum(p -> p[k], x), length(first(x))))
    hi = Ferrite.Vec(ntuple(k -> maximum(p -> p[k], x), length(first(x))))
    tolerance = max.(1.0e-6 .* Vector(hi - lo), floatmin(Float64))
    local_id = Dict(n => k for (k, n) in enumerate(ids))
    local_cells = [(cell = Ferrite.getcells(grid, c); typeof(cell)(map(n -> local_id[n], cell.nodes))) for c in cells]
    local_grid = Ferrite.Grid(local_cells, [Ferrite.Node(p - lo) for p in x])
    local_dh = Ferrite.DofHandler(local_grid)
    Ferrite.add!(local_dh, field, ip)
    Ferrite.close!(local_dh)
    positions = [Ferrite.apply_analytical!(zeros(Ferrite.ndofs(local_dh)), local_dh, field, p -> p[k] / tolerance[k]) for k in eachindex(tolerance)]
    scaled = Dict{Int, Vector{Float64}}()
    for sdh in dh.subdofhandlers, (k, c) in enumerate(cells)
        field in sdh.field_names && c in sdh.cellset || continue
        for (l, d) in zip(Ferrite.celldofs(local_dh, k), view(Ferrite.celldofs(dh, c), Ferrite.dof_range(sdh, field)))
            scaled[d] = getindex.(positions, l)
        end
    end
    return Vector(lo), tolerance, scaled, local_grid
end

# The points of each of `facets` of `grid` at a quadrature rule of the mesh geometry, scaled by `tolerance`. The rule is exact
# to twice the geometric order with positive weights, so two facets whose maps agree at its points are the same facet, and it is
# symmetric, so the set of points does not depend on the orientation of the facet in its cell.
function _facet_points(grid, facets, tolerance)
    ip = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{Ferrite.getrefshape(ip)}(2 * _order(ip)), ip, ip)
    return map(facets) do (cell, facet)
        x = Ferrite.getcoordinates(grid, cell)
        Ferrite.reinit!(fv, Ferrite.getcells(grid, cell), x, facet)
        [Vector(Ferrite.spatial_coordinate(fv, q, x)) ./ tolerance for q in 1:Ferrite.getnquadpoints(fv)]
    end
end

# `nothing` if the facets of Γ_a (the scaled points `pa` of each), translated by `sshift`, are the facets of Γ_b one to one,
# else why not. Facets are paired by their mean points, and a pair must agree at every point.
function _facet_problem(pa, pb, sshift, origin, tolerance, a, b)
    centres = [sum(p) / length(p) for p in [pa; pb]]
    place(f) = _tuple_str(origin + centres[f] .* tolerance)
    na = length(pa)
    matched, unmatched = _match_translate(centres, 1:na, na .+ eachindex(pb), sshift)
    isempty(unmatched) || return "the facet of Γ_$a centred at $(place(first(unmatched))) has no facet of Γ_$b at the translated place: " *
        "the two pieces are triangulated differently, or a facet is curved differently"
    hits = zeros(Int, length(pb))
    for (f, g) in matched
        hits[g - na] += 1
        all(p -> any(q -> norm(p + sshift - q) <= 1, pb[g - na]), pa[f]) || return "the facet of Γ_$a centred at $(place(f)) and the facet of " *
            "Γ_$b centred at $(place(g)) differ in shape: the two pieces are curved differently"
    end
    g = findfirst(!=(1), hits)
    g === nothing || return "the facet of Γ_$b centred at $(place(na + g)) is the image of $(hits[g]) facets of Γ_$a: the two pieces are triangulated differently"
    return nothing
end

function _tuple_str(x)
    return "(" * join(round.(x; sigdigits = 6), ", ") * ")"
end

#     periodic_ties(grid, dh, ips, nodes, periodic, prescribed) -> (ties, classes)
#
# `ips[field]` is the interpolation of each field and `nodes` are the dof coordinates, used for the Dirichlet data and the
# messages; each pair is matched in its own frame (`_pair_frame`), on the facets of its pieces where the field lives.
# `(ties, classes)`: `mirror => master` dof pairs (and the prescribed dofs of each class, for `check_agreement`) that make the dofs matched by `periodic`, a collection of
# `(field, set_a, set_b)`, equal. The pieces of a pair must be translates of each other: the translation is the
# difference of their centroids and each node of one must have a node of the other at the translated place, or
# an `ArgumentError` names both pieces. A master is a `prescribed` (Dirichlet) dof of its class if it has one, else
# its lowest dof; prescribed dofs are never mirrors.
function periodic_ties(grid, dh, ips, nodes, periodic, prescribed; dirichlet = Dict{Int, Any}(), times = (0.0,))
    coords = [collect(Float64, Tuple(x)) for x in nodes]
    parent = collect(1:Ferrite.ndofs(dh))
    root(d) = parent[d] == d ? d : (parent[d] = root(parent[d]))
    for (field, a, b) in periodic
        da, db = _piece_dofs(dh, field, grid, a), _piece_dofs(dh, field, grid, b)
        carrying = Set(_field_cells(dh, field))
        fa, fb = ([f for f in Ferrite.getfacetset(grid, String(set)) if f[1] in carrying] for set in (a, b))
        cells = unique([f[1] for f in [fa; fb]])
        lo, tolerance, scaled, local_grid = _pair_frame(grid, dh, field, ips[field], cells)
        local_cell = Dict(c => k for (k, c) in enumerate(cells))
        local_facets(fs) = [Ferrite.FacetIndex(local_cell[c], f) for (c, f) in fs]
        mean_coord(ds) = sum(scaled[d] for d in ds) / length(ds)
        sshift = mean_coord(db) - mean_coord(da)
        shift = sshift .* tolerance
        names = "Γ_$a and Γ_$b of $field"
        norm(sshift) <= 1 && throw(ArgumentError("periodic pieces $names coincide: their centroids are the same point, so there is no translation between them"))
        matched, unmatched = _match_translate(scaled, da, db, sshift)
        problem = _matching_problem(coords, scaled, da, db, matched, unmatched, shift, sshift, a, b)
        problem === nothing && (problem = _facet_problem(_facet_points(local_grid, local_facets(fa), tolerance), _facet_points(local_grid, local_facets(fb), tolerance), sshift, lo, tolerance, a, b))
        problem === nothing || throw(
            ArgumentError(
                "periodic pieces $names do not match under a translation: $problem; the translation between their centroids is " *
                    "$(_tuple_str(shift)) and the tolerance per axis is $(_tuple_str(tolerance)). Periodic pieces must be translates of each other, " *
                    "with a one-to-one correspondence of their nodes and facets"
            )
        )
        for (x, y) in matched
            parent[root(x)] = root(y)
        end
    end
    classes = Dict{Int, Vector{Int}}()
    for d in eachindex(parent)
        push!(get!(classes, root(d), Int[]), d)
    end
    isprescribed = falses(length(parent))
    isprescribed[prescribed] .= true
    ties = Pair{Int, Int}[]
    groups = Vector{Int}[]
    for members in values(classes)
        length(members) > 1 || continue
        master = members[something(findfirst(d -> isprescribed[d], members), 1)]
        fixed = filter(d -> isprescribed[d], members)
        length(fixed) > 1 && push!(groups, fixed)
        for d in members
            d == master || isprescribed[d] || push!(ties, d => master)
        end
    end
    classes = DirichletClasses(
        groups, Dict(d => collect(Float64, Tuple(nodes[d])) for g in groups for d in g),
        Dict(d => String(first(dirichlet[d])) for g in groups for d in g), collect(Any, periodic)
    )
    for t in times
        check_agreement(classes, d -> (g = last(dirichlet[d]); hasmethod(g, Tuple{typeof(nodes[d]), Float64}) ? g(nodes[d], t) : g(nodes[d])), t)
    end
    return sort!(ties), classes
end

# The prescribed dofs of each periodic class that holds more than one, with what an error needs to name them.
struct DirichletClasses
    groups::Vector{Vector{Int}}
    nodes::Dict{Int, Vector{Float64}}
    pieces::Dict{Int, String}
    periodic::Vector{Any}
end

#     check_agreement(classes::DirichletClasses, value, t)
#
# Throw an `ArgumentError` when `value(d)`, the prescribed value of dof `d` at time `t`, differs between two prescribed dofs
# of one periodic class by more than `1e-8` relative to the data. Called at every evaluation of the Dirichlet data.
function check_agreement(classes::DirichletClasses, value, t)
    for group in classes.groups
        a = ForwardDiff.value(value(first(group)))
        for d in view(group, 2:length(group))
            b = ForwardDiff.value(value(d))
            abs(a - b) <= 1.0e-8 * max(1.0, abs(a), abs(b)) && continue
            pairs = join(("Γ_$pa ~ Γ_$pb" for (_, pa, pb) in classes.periodic), ", ")
            throw(
                ArgumentError(
                    "the Dirichlet data of periodic pieces ($pairs) disagree: the nodes at $(_tuple_str(classes.nodes[first(group)])) " *
                        "and $(_tuple_str(classes.nodes[d])) are the same node of the periodic map, but Γ_$(classes.pieces[first(group)]) " *
                        "prescribes $a and Γ_$(classes.pieces[d]) prescribes $b at t = $(ForwardDiff.value(t))"
                )
            )
        end
    end
    return nothing
end

# Every `member => master` of the periodic classes: the solver's `ties` plus the prescribed members of a class beyond its
# master, whose Dirichlet rows already enforce them (the master is the first of each group).
class_ties(ties, classes::DirichletClasses) = sort!([ties; [d => first(g) for g in classes.groups for d in view(g, 2:length(g))]])

#     periodic_projection(n, ties) -> SparseMatrixCSC{Float64, Int}
#
# The `n × n_independent` matrix `P` of the identification `ties`: one column per dof that is not a mirror, in dof order,
# and a 1 in each row for the column of that dof's master. `P' * M * P` and `P' * F` are the condensation `apply_ties!`
# writes, up to its identification rows.
function periodic_projection(n, ties)
    master = collect(1:n)
    foreach(((mirror, m),) -> master[mirror] = m, ties)
    column = cumsum(master .== 1:n)
    return sparse(1:n, column[master], 1.0, n, last(column))
end

# The tied rows whose residual is summed into the master's row: those whose master is not prescribed.
_summed(ties, isprescribed) = [m => s for (m, s) in ties if !isprescribed[s]]

#     extend_pattern(pattern, ties, prescribed) -> SparseMatrixCSC
#
# `pattern` with the entries `apply_ties!` writes: the master row of every summed tie gains the columns of the tied
# row, and each tied row gains its diagonal and its master's column.
function extend_pattern(pattern::SparseMatrixCSC, ties, prescribed)
    isempty(ties) && return pattern
    isprescribed = isprescribed_mask(size(pattern, 1), prescribed)
    rows = rowvals(pattern)
    I, J = findnz(pattern)
    by_row = Dict{Int, Vector{Int}}()
    for j in axes(pattern, 2), k in nzrange(pattern, j)
        push!(get!(by_row, rows[k], Int[]), j)
    end
    for (m, s) in ties
        append!(I, (m, m)); append!(J, (m, s))
        for (mm, ss) in _summed((m => s,), isprescribed), j in by_row[mm]
            push!(I, ss); push!(J, j)
        end
    end
    return sparse(I, J, zeros(length(I)), size(pattern)...)
end

# Position in `A`'s stored values of entry `(i, j)`, which the pattern must hold.
function _position(A, i, j)
    r = nzrange(A, j)
    k = searchsortedfirst(view(rowvals(A), r), i)
    k <= length(r) && rowvals(A)[r[k]] == i || throw(ArgumentError("the sparsity pattern has no entry ($i, $j) for a periodic tie"))
    return r[k]
end

#     apply_ties!(M, K, ties, prescribed)
#
# For each `mirror => master` of `ties`, add the mirror's row of `M` and `K` to the master's row (unless the master
# is prescribed) and replace the mirror's rows: `M` row zero, `K` row `e_mirror - e_master`. `M` and `K` must have the
# pattern of `extend_pattern`.
function apply_ties!(M, K, ties, prescribed)
    isempty(ties) && return M, K
    n = size(K, 1)
    isprescribed = isprescribed_mask(n, prescribed)
    master = zeros(Int, n)
    foreach(((m, s),) -> master[m] = s, ties)
    rows = rowvals(K)
    for A in (M, K), j in axes(A, 2), k in nzrange(A, j)
        s = master[rows[k]]
        s == 0 || isprescribed[s] || (nonzeros(A)[_position(A, s, j)] += nonzeros(A)[k])
    end
    for j in axes(K, 2), k in nzrange(K, j)
        i = rows[k]
        master[i] == 0 && continue
        nonzeros(M)[k] = 0
        nonzeros(K)[k] = i == j ? 1 : j == master[i] ? -1 : 0
    end
    return M, K
end

# Rows of the load vector summed and cleared like the rows of `apply_ties!`.
apply_ties!(F::AbstractVector, ties, prescribed::AbstractVector{<:Integer}) = apply_ties!(F, ties, isprescribed_mask(length(F), prescribed))
function apply_ties!(F::AbstractVector, ties, isprescribed::BitVector)
    for (m, s) in ties
        isprescribed[s] || (F[s] += F[m])
        F[m] = 0
    end
    return F
end

# What `FEMNonlinear` needs to add a mirror row of the nonlinear residual and Jacobian to its master's row.
function _tie_positions(J, ties, prescribed)
    isempty(ties) && return nothing
    summed = _summed(ties, isprescribed_mask(size(J, 1), prescribed))
    master = zeros(Int, size(J, 1))
    foreach(((m, s),) -> master[m] = s, summed)
    from, to = Int[], Int[]
    rows = rowvals(J)
    for j in axes(J, 2), k in nzrange(J, j)
        s = master[rows[k]]
        s == 0 && continue
        push!(from, k)
        push!(to, _position(J, s, j))
    end
    return (summed, from, to)
end
