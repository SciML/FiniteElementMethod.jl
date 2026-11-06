# Near-to-far-field transform of a MaxwellSolution: equivalent currents on a closed facet surface (the
# Huygens surface) radiating into vacuum.

# E and H (SI units) at the facet quadrature points of a Huygens surface, with the outward normal and the
# area weights; positions in metres.
struct _HuygensSurface
    x::Vector{Ferrite.Vec{3, Float64}}
    normal::Vector{Ferrite.Vec{3, Float64}}
    E::Vector{Ferrite.Vec{3, ComplexF64}}
    H::Vector{Ferrite.Vec{3, ComplexF64}}
    dS::Vector{Float64}
end

_facet_nodes(grid, cell, local_facet) =
    sort!([Ferrite.getcells(grid, cell).nodes[i] for i in Ferrite.reference_facets(RefTetrahedron)[local_facet]])

_centroid(grid, nodes) = sum(Ferrite.get_node_coordinate(grid, n) for n in nodes) / length(nodes)

function _area(grid, (a, b, c))
    xa, xb, xc = (Ferrite.get_node_coordinate(grid, n) for n in (a, b, c))
    return norm(cross(xb - xa, xc - xa)) / 2
end

# Every cell next to each geometric facet of the set, whichever of them the set lists.
function _facet_cells(grid, name)
    _check_facetset(grid, name)
    triangles = Dict{Vector{Int}, Vector{Ferrite.FacetIndex}}()
    for (cell, local_facet) in Ferrite.getfacetset(grid, name)
        triangles[_facet_nodes(grid, cell, local_facet)] = Ferrite.FacetIndex[]
    end
    isempty(triangles) && throw(ArgumentError("facet set $(repr(name)) is empty"))
    for cell in 1:Ferrite.getncells(grid), local_facet in 1:4
        cells = get(triangles, _facet_nodes(grid, cell, local_facet), nothing)
        cells === nothing || push!(cells, Ferrite.FacetIndex(cell, local_facet))
    end
    return triangles
end

# A closed triangulated surface uses every edge an even number of times (twice when it is a manifold).
function _check_closed(triangles, name)
    edges = Dict{Tuple{Int, Int}, Int}()
    for (a, b, c) in keys(triangles), edge in ((a, b), (a, c), (b, c))
        edges[edge] = get(edges, edge, 0) + 1
    end
    open_edges = count(isodd, values(edges))
    open_edges == 0 ||
        throw(ArgumentError("facet set $(repr(name)) is not a closed surface: $open_edges of its edges bound an odd number of its facets"))
    return nothing
end

# The normals chosen facet by facet are one consistent outward orientation when each edge is traversed once in
# each direction by the triangles oriented by them, and the enclosed volume ∮ x⋅n dS / 3 is positive.
function _check_orientation(grid, oriented, name, center)
    edges = Dict{Tuple{Int, Int}, Int}()
    volume = 0.0
    for (a, b, c) in oriented
        for (i, j) in ((a, b), (b, c), (c, a))
            edges[(i, j)] = get(edges, (i, j), 0) + 1
            edges[(j, i)] = get(edges, (j, i), 0) - 1
        end
        xa, xb, xc = (Ferrite.get_node_coordinate(grid, n) for n in (a, b, c))
        volume += dot(xa, cross(xb - xa, xc - xa)) / 6
    end
    (all(iszero, values(edges)) && volume > 0) ||
        throw(ArgumentError("the normals of the Huygens surface $(repr(name)) pointing away from the centre $center are not one outward orientation: the surface must be star-shaped about the centre"))
    return nothing
end

function _huygens_surface(sol::MaxwellSolution, name::AbstractString, center)
    (; model, omega, u) = sol
    grid, dh, L = model.grid, model.dh, model.length_scale
    name = String(name)
    triangles = _facet_cells(grid, name)
    _check_closed(triangles, name)
    if center === nothing
        areas = Dict(nodes => _area(grid, nodes) for nodes in keys(triangles))
        center = sum(_centroid(grid, nodes) * a for (nodes, a) in areas) / sum(values(areas))
    end
    center = Ferrite.Vec{3, Float64}(Tuple(center))
    fv = _facet_values()
    local_dofs = Ferrite.dirichlet_facetdof_indices(_NEDELEC)
    prescribed = Set(model.ch.prescribed_dofs)
    surface = _HuygensSurface(Ferrite.Vec{3, Float64}[], Ferrite.Vec{3, Float64}[], Ferrite.Vec{3, ComplexF64}[], Ferrite.Vec{3, ComplexF64}[], Float64[])
    oriented = NTuple{3, Int}[]
    for (nodes, facets) in triangles
        xf = _centroid(grid, nodes)
        inner = nothing
        curl_sum = zero(Ferrite.Vec{3, ComplexF64})
        for facet in facets
            cell, local_facet = facet[1], facet[2]
            model.materials[cell] == MaxwellMaterial() ||
                throw(ArgumentError("the Huygens surface $(repr(name)) must lie in vacuum; cell $cell next to it is not vacuum"))
            dofs = Ferrite.celldofs(dh, cell)
            any(i -> dofs[i] in prescribed, local_dofs[local_facet]) &&
                throw(ArgumentError("the Huygens surface $(repr(name)) touches a PEC facet; it must enclose every conductor"))
            Ferrite.reinit!(fv, Ferrite.getcells(grid, cell), Ferrite.getcoordinates(grid, cell), local_facet)
            # The curl of a lowest-order Nédélec field is constant in each tetrahedron.
            curl_sum += Ferrite.function_curl(fv, 1, u[dofs])
            dot(Ferrite.getnormal(fv, 1), xf - center) > 0 && (inner = facet)
        end
        inner === nothing &&
            throw(ArgumentError("facet $nodes of the Huygens surface $(repr(name)) has no cell on the side of the centre $center: the surface must be star-shaped about it"))
        # H = (i / ωμ0) ∇ × E, averaged over the cells on both sides, which halves the O(h) bias of one side.
        H = im / (omega * model.mu0 * L) * curl_sum / length(facets)
        cell, local_facet = inner[1], inner[2]
        x = Ferrite.getcoordinates(grid, cell)
        Ferrite.reinit!(fv, Ferrite.getcells(grid, cell), x, local_facet)
        u_cell = u[Ferrite.celldofs(dh, cell)]
        a, b, c = nodes
        xa, xb, xc = (Ferrite.get_node_coordinate(grid, n) for n in nodes)
        push!(oriented, dot(cross(xb - xa, xc - xa), Ferrite.getnormal(fv, 1)) > 0 ? (a, b, c) : (a, c, b))
        for q in 1:Ferrite.getnquadpoints(fv)
            push!(surface.x, Ferrite.spatial_coordinate(fv, q, x) * L)
            push!(surface.normal, Ferrite.getnormal(fv, q))
            push!(surface.E, Ferrite.function_value(fv, q, u_cell))
            push!(surface.H, H)
            push!(surface.dS, Ferrite.getdetJdV(fv, q) * L^2)
        end
    end
    _check_orientation(grid, oriented, name, center)
    return surface
end

function _unit_direction(d)
    r = Ferrite.Vec{3, Float64}(Tuple(d))
    nr = norm(r)
    nr > 0 || throw(ArgumentError("a far-field direction must be a nonzero vector"))
    return r / nr
end

# θ̂ and φ̂ at the unit vector r̂; φ = 0 on the z axis.
function _spherical_basis(r̂)
    θ, φ = acos(clamp(r̂[3], -1.0, 1.0)), atan(r̂[2], r̂[1])
    θ̂ = Ferrite.Vec{3, Float64}((cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ)))
    φ̂ = Ferrite.Vec{3, Float64}((-sin(φ), cos(φ), 0.0))
    return θ̂, φ̂
end

function _far_field(surface::_HuygensSurface, model, omega, directions)
    k = omega * sqrt(model.mu0 * model.eps0)
    η = sqrt(model.mu0 / model.eps0)
    # Love's equivalent currents on the outward normal n: J = n × H, M = -n × E.
    J = cross.(surface.normal, surface.H) .* surface.dS
    M = cross.(surface.E, surface.normal) .* surface.dS
    E_theta, E_phi = ComplexF64[], ComplexF64[]
    for d in directions
        r̂ = _unit_direction(d)
        N, Lm = zero(Ferrite.Vec{3, ComplexF64}), zero(Ferrite.Vec{3, ComplexF64})
        for (i, x) in pairs(surface.x)
            phase = cis(k * dot(r̂, x))
            N += J[i] * phase
            Lm += M[i] * phase
        end
        θ̂, φ̂ = _spherical_basis(r̂)
        push!(E_theta, -im * k / 4π * (dot(Lm, φ̂) + η * dot(N, θ̂)))
        push!(E_phi, im * k / 4π * (dot(Lm, θ̂) - η * dot(N, φ̂)))
    end
    return (; E_theta, E_phi)
end

"""
    far_field(sol::MaxwellSolution, surface, directions; center = nothing)

The far field of `sol` in the unit `directions` (3-vectors, normalised here) as `(; E_theta, E_phi)`, vectors
of the spherical components of `r e^{ikr} E(r r̂)` in volt as `r → ∞` (with `exp(+iωt)` the field is
`E ≈ F e^{-ikr}/r`; `k = ω sqrt(μ0 ε0)`).

`surface` names a closed facet set (the Huygens surface) inside the mesh, in vacuum, that encloses every
source, port and conductor and touches no PEC facet; outside it the fields must be outgoing (an absorbing
boundary or PML further out). Its equivalent currents `J = n × H` and `M = -n × E` on the outward normal
`n` radiate into vacuum. The set may list each facet from either cell or from both. The outward side is the
one away from `center` (default: the area-weighted centroid of the surface), so the surface must be
star-shaped about `center`, such as a sphere or a box. `E` is taken from the inner cell, and
`H = (i/ωμ0) ∇ × E`, constant in each tetrahedron, is the average of the cells on both sides. At the poles
`θ̂` and `φ̂` are those of `φ = 0`.
"""
function far_field(sol::MaxwellSolution, surface::AbstractString, directions; center = nothing)
    return _far_field(_huygens_surface(sol, surface, center), sol.model, sol.omega, directions)
end

# Directions at the midpoints of an ntheta × nphi grid in (θ, φ) and their solid angles; the midpoint rule is
# spectrally accurate for smooth functions on the sphere.
function _sphere_grid(ntheta, nphi)
    θs = (0.5:ntheta) .* (π / ntheta)
    φs = (0:(nphi - 1)) .* (2π / nphi)
    directions = [Ferrite.Vec{3, Float64}((sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))) for φ in φs for θ in θs]
    weights = [sin(θ) * (π / ntheta) * (2π / nphi) for φ in φs for θ in θs]
    return directions, weights
end

_intensity(model, ff) = (abs2.(ff.E_theta) .+ abs2.(ff.E_phi)) ./ (2 * sqrt(model.mu0 / model.eps0))

function _far_field_power(surface, model, omega, ntheta, nphi)
    directions, weights = _sphere_grid(ntheta, nphi)
    return dot(weights, _intensity(model, _far_field(surface, model, omega, directions)))
end

"""
    radiated_power(sol::MaxwellSolution, surface; center = nothing, method = :flux, ntheta = 36, nphi = 72)

The time-averaged power (watt) radiated through the Huygens surface `surface` (see [`far_field`](@ref)).
`method = :flux` integrates the Poynting vector `½ Re(E × H̄)⋅n` over the surface; `method = :far_field`
integrates the radiation intensity `U = (|E_θ|² + |E_φ|²)/2η0` of [`far_field`](@ref) over the sphere on an
`ntheta × nphi` midpoint grid. The far-field integral counts only the outgoing field, the flux the net power,
so they differ by the power the outer boundary reflects back in, besides discretisation error.
"""
function radiated_power(
        sol::MaxwellSolution, surface::AbstractString; center = nothing, method::Symbol = :flux,
        ntheta::Integer = 36, nphi::Integer = 72
    )
    hs = _huygens_surface(sol, surface, center)
    method === :far_field && return _far_field_power(hs, sol.model, sol.omega, ntheta, nphi)
    method === :flux || throw(ArgumentError("method must be :flux or :far_field, got $(repr(method))"))
    return sum(real(dot(cross(E, conj(H)), n)) * dS for (E, H, n, dS) in zip(hs.E, hs.H, hs.normal, hs.dS)) / 2
end

"""
    directivity(sol::MaxwellSolution, surface, directions; center = nothing, ntheta = 36, nphi = 72)

The directivity `D = 4π U / P` (linear, not dB) in the unit `directions`, with `U` the radiation intensity of
[`far_field`](@ref) and `P` its integral over the sphere (`radiated_power(...; method = :far_field)`), so that
`D` averages to 1 over the sphere whatever the error of the overall amplitude.
"""
function directivity(
        sol::MaxwellSolution, surface::AbstractString, directions; center = nothing,
        ntheta::Integer = 36, nphi::Integer = 72
    )
    hs = _huygens_surface(sol, surface, center)
    P = _far_field_power(hs, sol.model, sol.omega, ntheta, nphi)
    P > 0 || throw(ArgumentError("the solution radiates no power through $(repr(surface))"))
    return 4π .* _intensity(sol.model, _far_field(hs, sol.model, sol.omega, directions)) ./ P
end
