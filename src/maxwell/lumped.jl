# Lumped ports of a MaxwellModel: a uniform sheet current across a planar gap, loaded by its reference
# impedance and driven by a Norton current source.

"""
    LumpedPort(facetset; direction, Z0 = 50.0)

A lumped port on the planar facet set `facetset` of a [`MaxwellModel`](@ref): a rectangular sheet of length
`l` along `direction` (in the sheet plane, from one conductor to the other) and width `w` across it,
bridging a gap. The sheet may lie on the boundary or inside the mesh; a facet listed from both of its cells
counts once.

The port carries a uniform sheet current `I / w` along `direction`, so the gap voltage is the sheet average
of the line integral of `E` along `direction`, `V = (1/w) ∮ E⋅l̂ dS` (positive when the start of `direction`
is the + terminal). Every port is loaded by its real reference impedance `Z0` (ohm) and the driven port adds a
1 A current source in parallel (see [`FiniteElementMethod.port_excitations`](@ref)). The port couples all of
its edge dofs to each other, so keep the sheet a few elements across.
"""
struct LumpedPort
    facetset::String
    direction::Ferrite.Vec{3, Float64}
    Z0::Float64
end
function LumpedPort(facetset; direction, Z0::Real = 50.0)
    Z0 > 0 || throw(ArgumentError("the reference impedance Z0 of a lumped port must be positive, got $Z0"))
    return LumpedPort(String(facetset), Ferrite.Vec{3, Float64}(Tuple(direction)), float(Z0))
end

"""
    LumpedPortData

A [`LumpedPort`](@ref) assembled on a [`MaxwellModel`](@ref), in `model.ports`. Fields: `port`, `profile`
(`gᵢ = ∮ l̂⋅Nᵢ dS` over the sheet, so that `V = L gᵀu / w`), `dofs` (the edge dofs of the sheet), `block`
(`g gᵀ` on the model's sparsity pattern), `normal`, `length` and `width` (`l` and `w`, mesh units).
"""
struct LumpedPortData
    port::LumpedPort
    profile::Vector{Float64}
    dofs::Vector{Int}
    block::SparseMatrixCSC{Float64, Int}
    normal::Ferrite.Vec{3, Float64}
    length::Float64
    width::Float64
end

# The facets of a set with each geometric facet once, whichever of its two cells lists it.
function _unique_facets(grid, facets)
    seen = Set{Vector{Int}}()
    unique_facets = Ferrite.FacetIndex[]
    for facet in facets
        cell, local_facet = facet[1], facet[2]
        cell_nodes = Ferrite.getcells(grid, cell).nodes
        nodes = sort!([cell_nodes[i] for i in _TET_FACET_NODES[local_facet]])
        nodes in seen && continue
        push!(seen, nodes)
        push!(unique_facets, Ferrite.FacetIndex(cell, local_facet))
    end
    return unique_facets
end

# Profile, dofs and geometry of a lumped port; the block needs the model's pattern, which needs the dofs.
function _lumped_port_geometry(port::LumpedPort, dh)
    grid = dh.grid
    _check_facetset(grid, port.facetset)
    facets = _unique_facets(grid, Ferrite.getfacetset(grid, port.facetset))
    isempty(facets) && throw(ArgumentError("port facet set $(repr(port.facetset)) is empty"))
    fv = _facet_values()
    local_dofs = _TET_FACET_EDGE_DOFS
    profile = zeros(Ferrite.ndofs(dh))
    normal = nothing
    nodes = Ferrite.Vec{3, Float64}[]
    area = 0.0
    l̂ = port.direction / norm(port.direction)
    for (cell, local_facet) in facets
        x = Ferrite.getcoordinates(grid, cell)
        Ferrite.reinit!(fv, Ferrite.getcells(grid, cell), x, local_facet)
        nq = Ferrite.getnormal(fv, 1)
        normal === nothing && (normal = nq)
        abs(dot(nq, normal)) >= 1 - 1.0e-8 || throw(ArgumentError("port facet set $(repr(port.facetset)) is not planar"))
        dofs = Ferrite.celldofs(dh, cell)
        for q in 1:Ferrite.getnquadpoints(fv)
            dS = Ferrite.getdetJdV(fv, q)
            area += dS
            # l̂ is tangential, so only the edges of this facet contribute.
            for i in local_dofs[local_facet]
                profile[dofs[i]] += dot(l̂, Ferrite.shape_value(fv, q, i)) * dS
            end
        end
        append!(nodes, x[i] for i in _TET_FACET_NODES[local_facet])
    end
    abs(dot(l̂, normal)) <= 1.0e-8 || throw(ArgumentError("direction of port $(repr(port.facetset)) is not in the port plane"))
    t̂ = cross(normal, l̂)
    (s0, s1), (t0, t1) = extrema(dot(x, l̂) for x in nodes), extrema(dot(x, t̂) for x in nodes)
    len, width = s1 - s0, t1 - t0
    heights = [dot(x, normal) for x in nodes]
    maximum(heights) - minimum(heights) <= 1.0e-8 * max(len, width) ||
        throw(ArgumentError("port facet set $(repr(port.facetset)) is not planar"))
    abs(area - len * width) <= 1.0e-6 * len * width ||
        throw(ArgumentError("port facet set $(repr(port.facetset)) is not a rectangle with a side along direction"))
    dofs = sort!(unique(Ferrite.celldofs(dh, f[1])[i] for f in facets for i in local_dofs[f[2]]))
    return (; profile, dofs, normal, len, width)
end

# The cell pattern of `dh` plus every pair of edge dofs within each lumped port.
function _sparsity_pattern(dh, port_dofs)
    pattern = Ferrite.allocate_matrix(dh)
    isempty(port_dofs) && return pattern
    rows, cols, _ = findnz(pattern)
    for dofs in port_dofs, i in dofs, j in dofs
        push!(rows, i)
        push!(cols, j)
    end
    n = Ferrite.ndofs(dh)
    pattern = sparse(rows, cols, ones(length(rows)), n, n)
    fill!(nonzeros(pattern), 0.0)
    return pattern
end

function _lumped_port(port::LumpedPort, geometry, pattern)
    (; profile, dofs, normal, len, width) = geometry
    block = copy(pattern)
    Ferrite.assemble!(Ferrite.start_assemble(block), dofs, profile[dofs] * profile[dofs]')
    return LumpedPortData(port, profile, dofs, block, normal, len, width)
end

# The sheet current I = V/Z0 = L gᵀu / (w Z0) gives +iωμ0 L / (w² Z0) g gᵀ on the operator.
function _port_term(data::LumpedPortData, mu0, eps0, L)
    scale = mu0 * L / (data.width^2 * data.port.Z0)
    return MaxwellTerm(data.block, ω -> im * ω * scale)
end

function _lumped(model, port::Integer)
    data = model.ports[port]
    data isa LumpedPortData || throw(ArgumentError("port $port of the model is not a lumped port"))
    return data
end

"""
    port_voltage(model::MaxwellModel, port, u)

The gap voltage `V = (1/w) ∮ E⋅l̂ dS` (volt) of lumped port number `port` of `model` in the edge-dof vector `u`.
"""
port_voltage(model, port::Integer, u::AbstractVector) =
    (data = _lumped(model, port); model.length_scale * dot(data.profile, u) / data.width)

"""
    port_current(model::MaxwellModel, port, u; source = 0)

The current (ampere) that lumped port number `port` delivers into the structure in the solution `u`,
`source - V / Z0`, with `source` the current of its Norton source: 1 for the driven port of a
[`solve_sweep`](@ref) column, 0 for the others.
"""
port_current(model, port::Integer, u::AbstractVector; source::Number = 0) =
    source - port_voltage(model, port, u) / _lumped(model, port).port.Z0

"""
    z_parameters(model::MaxwellModel, S)

The impedance matrix (ohm) `Z = G^½ (I + S)(I - S)⁻¹ G^½`, `G = diag(Z0)`, of a scattering matrix `S` of a
model whose ports are all lumped ports, inverting `Sᵢⱼ = 2 Vᵢ / sqrt(Z0ᵢ Z0ⱼ) - δᵢⱼ` for a 1 A source at port `j`.
"""
function z_parameters(model, S::AbstractMatrix)
    sqrtZ0 = Diagonal([sqrt(_lumped(model, i).port.Z0) for i in eachindex(model.ports)])
    return sqrtZ0 * ((I + S) / (I - S)) * sqrtZ0
end
