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

# Coordinates are matched on a grid of cell `tol`, looking in the neighbouring cells too.
function _match_translate(coords, from, to, shift, tol)
    index = Dict{Any, Vector{Int}}()
    for d in to
        push!(get!(index, Tuple(round.(Int, coords[d] / tol)), Int[]), d)
    end
    dim = length(first(coords))
    matched = Pair{Int, Int}[]
    unmatched = Int[]
    for d in from
        target = coords[d] + shift
        key = round.(Int, target / tol)
        best, bestdist = 0, Inf
        for offset in Iterators.product(ntuple(_ -> -1:1, dim)...), e in get(index, Tuple(key .+ offset), Int[])
            dist = norm(coords[e] - target)
            dist < bestdist && ((best, bestdist) = (e, dist))
        end
        bestdist <= tol ? push!(matched, d => best) : push!(unmatched, d)
    end
    return matched, unmatched
end

function _tuple_str(x)
    return "(" * join(round.(x; sigdigits = 6), ", ") * ")"
end

"""
    periodic_ties(grid, dh, coords, periodic, prescribed) -> Vector{Pair{Int, Int}}

`mirror => master` dof pairs that make the dofs matched by `periodic`, a collection of
`(field, set_a, set_b)`, equal. The pieces of a pair must be translates of each other: the translation is the
difference of their centroids and each node of one must have a node of the other at the translated place, or
an `ArgumentError` names both pieces. A master is a `prescribed` (Dirichlet) dof of its class if it has one, else
its lowest dof; prescribed dofs are never mirrors.
"""
function periodic_ties(grid, dh, nodes, periodic, prescribed)
    coords = [collect(Float64, Tuple(x)) for x in nodes]
    lo = reduce((a, b) -> min.(a, b), coords)
    hi = reduce((a, b) -> max.(a, b), coords)
    tol = 1.0e-6 * max(norm(hi - lo), eps())
    parent = collect(1:Ferrite.ndofs(dh))
    root(d) = parent[d] == d ? d : (parent[d] = root(parent[d]))
    for (field, a, b) in periodic
        da, db = _piece_dofs(dh, field, grid, a), _piece_dofs(dh, field, grid, b)
        mean_coord(ds) = sum(coords[d] for d in ds) / length(ds)
        shift = mean_coord(db) - mean_coord(da)
        names = "Γ_$a and Γ_$b of $field"
        norm(shift) <= tol && throw(ArgumentError("periodic pieces $names coincide: their centroids are the same point, so there is no translation between them"))
        matched, unmatched = _match_translate(coords, da, db, shift, tol)
        if !isempty(unmatched) || length(da) != length(db)
            first_miss = isempty(unmatched) ? "" : "; the node at $(_tuple_str(coords[first(unmatched)])) on Γ_$a has no node of Γ_$b at $(_tuple_str(coords[first(unmatched)] + shift))"
            throw(
                ArgumentError(
                    "periodic pieces $names do not match under a translation: Γ_$a has $(length(da)) nodes and Γ_$b has $(length(db)); " *
                        "the translation between their centroids is $(_tuple_str(shift)) and $(length(unmatched)) nodes of Γ_$a " *
                        "have no counterpart$first_miss. Periodic pieces must be translates of each other, with matching nodes"
                )
            )
        end
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
    for members in values(classes)
        length(members) > 1 || continue
        master = members[something(findfirst(d -> isprescribed[d], members), 1)]
        for d in members
            d == master || isprescribed[d] || push!(ties, d => master)
        end
    end
    return sort!(ties)
end

# The tied rows whose residual is summed into the master's row: those whose master is not prescribed.
_summed(ties, isprescribed) = [m => s for (m, s) in ties if !isprescribed[s]]

"""
    extend_pattern(pattern, ties, prescribed) -> SparseMatrixCSC

`pattern` with the entries `apply_ties!` writes: the master row of every summed tie gains the columns of the tied
row, and each tied row gains its diagonal and its master's column.
"""
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

"""
    apply_ties!(M, K, ties, prescribed)

For each `mirror => master` of `ties`, add the mirror's row of `M` and `K` to the master's row (unless the master
is prescribed) and replace the mirror's rows: `M` row zero, `K` row `e_mirror - e_master`. `M` and `K` must have the
pattern of `extend_pattern`.
"""
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

"Rows of the load vector summed and cleared like the rows of `apply_ties!`."
function apply_ties!(F::AbstractVector, ties, prescribed)
    isprescribed = isprescribed_mask(length(F), prescribed)
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
