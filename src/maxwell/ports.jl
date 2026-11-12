# Boundary terms of a MaxwellModel: tangential facet mass blocks, the first-order absorbing boundary and
# rectangular-waveguide TE10 wave ports.

"""
    WavePort(facetset; a_direction, e_direction = nothing, deembed = 0.0)

A TE10 wave port of a rectangular waveguide on the planar boundary facet set `facetset` of a
[`MaxwellModel`](@ref). The port face must be the full guide cross-section, a rectangle with PEC on its rim,
filled with one material of real permittivity and zero conductivity: a complex `eps_r` (a lossy dielectric) or a
nonzero conductivity is rejected.

`a_direction` is a vector along the broad wall (the side of length `a`); the mode is
`e = ê sin(π ξ / a)`, with `ξ` the distance from the edge of the rectangle along `a_direction` and `ê` the
in-plane unit vector perpendicular to it. By default `ê` is `n × â` with its sign chosen so that its largest
Cartesian component is positive, which gives both ends of a straight guide the same polarisation; pass
`e_direction` to fix it explicitly. `deembed` moves the reference plane of the port a distance (in mesh units)
from the port face into the device.

Above cutoff the wave coefficients are power waves: the incident mode carries real axial power, and a lossless
structure with only propagating ports has a unitary `S`. Below cutoff (`β = -iα`) the mode is evanescent and
carries no real axial power, so the coefficients are generalized traveling-wave coefficients, normalised the same
way and supported, but without an individual incident or outgoing power; `S` is then not power conserving.
"""
struct WavePort
    facetset::String
    a_direction::Ferrite.Vec{3, Float64}
    e_direction::Union{Nothing, Ferrite.Vec{3, Float64}}
    deembed::Float64
end
function WavePort(facetset; a_direction, e_direction = nothing, deembed::Real = 0.0)
    e = e_direction === nothing ? nothing : Ferrite.Vec{3, Float64}(Tuple(e_direction))
    return WavePort(String(facetset), Ferrite.Vec{3, Float64}(Tuple(a_direction)), e, float(deembed))
end

"""
    WavePortData

A [`WavePort`](@ref) assembled on a [`MaxwellModel`](@ref), in `model.ports`. Fields: `port`, `block` (the
tangential mass `∮ (n × Nᵢ)⋅(n × Nⱼ) dS` of the port face), `profile` (`pᵢ = ∮ e⋅Nᵢ dS`), `norm`
(`pᵀ B⁻¹ p` over the unconstrained edge dofs of the face: `∮ e_h⋅e_h dS` of the L² projection `e_h` of the
mode onto the discrete tangential traces, so that a port-face field `c e_h` gives exactly `pᵀu / norm = c`),
`normal` (outward), `a`, `b` (mesh units), `kc` (`π / (a L)`, rad/m),
`eps_r` and `mu_r` of the port material (lossless: complex `eps_r` is rejected).
"""
struct WavePortData
    port::WavePort
    block::SparseMatrixCSC{Float64, Int}
    profile::Vector{Float64}
    norm::Float64
    normal::Ferrite.Vec{3, Float64}
    a::Float64
    b::Float64
    kc::Float64
    eps_r::ComplexF64
    mu_r::Float64
end

function _check_facetset(grid, name)
    return haskey(grid.facetsets, name) ||
        throw(ArgumentError("no facet set $(repr(name)) in the grid; it has $(sort!(collect(keys(grid.facetsets))))"))
end

_facet_values() = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{RefTetrahedron}(2), _NEDELEC)

# ∮ w (n × Nᵢ)⋅(n × Nⱼ) dS over the named facet sets, with a real weight per cell, on the pattern of `pattern`.
# Every edge dof of a facet belongs to the facet's cell, so the cell-based pattern holds every entry.
function _tangential_mass(dh, pattern, names, weight)
    block = copy(pattern)
    asm = Ferrite.start_assemble(block)
    fv = _facet_values()
    n = Ferrite.getnbasefunctions(fv)
    Be = zeros(n, n)
    for name in names, fc in Ferrite.FacetIterator(dh, Ferrite.getfacetset(dh.grid, name))
        Ferrite.reinit!(fv, fc)
        fill!(Be, 0.0)
        for q in 1:Ferrite.getnquadpoints(fv)
            nq = Ferrite.getnormal(fv, q)
            dS = Ferrite.getdetJdV(fv, q)
            for i in 1:n, j in 1:n
                Ni, Nj = Ferrite.shape_value(fv, q, i), Ferrite.shape_value(fv, q, j)
                Be[i, j] += (dot(Ni, Nj) - dot(nq, Ni) * dot(nq, Nj)) * dS
            end
        end
        Ferrite.assemble!(asm, Ferrite.celldofs(fc), weight(Ferrite.cellid(fc)) .* Be)
    end
    return block
end

"""
    tangential_mass(model::MaxwellModel, facetsets; weight = cell -> 1.0)

The real block `∮ w (n × Nᵢ)⋅(n × Nⱼ) dS` over the named facet sets of `model.grid` (in mesh units), with the
weight `w = weight(cellid)` of the cell each facet belongs to, on the sparsity pattern of `model.blocks`, ready
for [`FiniteElementMethod.add_term!`](@ref).
"""
function tangential_mass(model, facetsets; weight = Returns(1.0))
    names = String.(collect(facetsets))
    foreach(name -> _check_facetset(model.grid, name), names)
    return _tangential_mass(model.dh, model.blocks.curlcurl, names, weight)
end

# First-order absorbing boundary n × (μr⁻¹ ∇ × E) = -i k0 ηr⁻¹ n × (n × E), with ηr⁻¹ = sqrt(εc / μr) and
# εc = εr - iσ/(ωε0) of the adjacent cell: the term +i k0 L ηr⁻¹ ∮ (n × E)⋅(n × v) dS, one block per material.
function _absorbing_terms(dh, pattern, names, cell_materials, mu0, eps0, L)
    materials = unique(cell_materials[facet[1]] for name in names for facet in Ferrite.getfacetset(dh.grid, name))
    return map(materials) do m
        block = _tangential_mass(dh, pattern, names, cell -> Float64(cell_materials[cell] == m))
        # The coefficient is ω √εc with εc ~ 1/ω for a conductor, so it tends to 0 as ω → 0.
        MaxwellTerm(
            block,
            ω -> iszero(ω) ? 0.0im : im * ω * sqrt(mu0 * eps0) * L * sqrt((m.eps_r - im * m.sigma / (ω * eps0)) / m.mu_r)
        )
    end
end

function _port_frame(port, normal)
    a_dir = port.a_direction
    abs(dot(a_dir, normal)) <= 1.0e-8 * norm(a_dir) ||
        throw(ArgumentError("a_direction of port $(repr(port.facetset)) is not in the port plane"))
    â = a_dir / norm(a_dir)
    if port.e_direction === nothing
        ê = cross(normal, â)
        ê = ê[argmax(abs.(Tuple(ê)))] < 0 ? -ê : ê
    else
        ê = port.e_direction / norm(port.e_direction)
        abs(dot(ê, normal)) + abs(dot(ê, â)) <= 1.0e-8 ||
            throw(ArgumentError("e_direction of port $(repr(port.facetset)) must be in the port plane and normal to a_direction"))
    end
    return â, ê
end

function _wave_port(port::WavePort, dh, pattern, cell_materials, length_scale, constrained)
    grid = dh.grid
    _check_facetset(grid, port.facetset)
    facets = Ferrite.getfacetset(grid, port.facetset)
    isempty(facets) && throw(ArgumentError("port facet set $(repr(port.facetset)) is empty"))
    fv = _facet_values()
    # Geometry: one outward normal, the facet nodes and the area.
    normal = nothing
    nodes = Ferrite.Vec{3, Float64}[]
    area = 0.0
    for fc in Ferrite.FacetIterator(dh, facets)
        Ferrite.reinit!(fv, fc)
        nq = Ferrite.getnormal(fv, 1)
        normal === nothing && (normal = nq)
        norm(nq - normal) <= 1.0e-8 || throw(ArgumentError("port facet set $(repr(port.facetset)) is not planar"))
        area += sum(q -> Ferrite.getdetJdV(fv, q), 1:Ferrite.getnquadpoints(fv))
    end
    for (cell, local_facet) in facets
        x = Ferrite.getcoordinates(grid, cell)
        append!(nodes, x[i] for i in _TET_FACET_NODES[local_facet])
    end
    â, ê = _port_frame(port, normal)
    ξ, η, ζ = [dot(x, â) for x in nodes], [dot(x, ê) for x in nodes], [dot(x, normal) for x in nodes]
    (ξ0, ξ1), (η0, η1) = extrema(ξ), extrema(η)
    a, b = ξ1 - ξ0, η1 - η0
    if maximum(ζ) - minimum(ζ) > 1.0e-8 * a || abs(area - a * b) > 1.0e-6 * a * b
        throw(ArgumentError("port facet set $(repr(port.facetset)) is not a planar rectangle with a side along a_direction"))
    end
    materials = unique(cell_materials[f[1]] for f in facets)
    length(materials) == 1 || throw(ArgumentError("port $(repr(port.facetset)) touches more than one material"))
    material = only(materials)
    iszero(material.sigma) || throw(ArgumentError("port $(repr(port.facetset)) is in a conducting material"))
    iszero(imag(material.eps_r)) ||
        throw(ArgumentError("port $(repr(port.facetset)) is in a lossy material (complex eps_r); the S-parameter power normalisation of a wave port assumes a lossless one"))

    e(x) = ê * sin(π * (dot(x, â) - ξ0) / a)
    profile = zeros(Ferrite.ndofs(dh))
    n = Ferrite.getnbasefunctions(fv)
    pe = zeros(n)
    for fc in Ferrite.FacetIterator(dh, facets)
        Ferrite.reinit!(fv, fc)
        x = Ferrite.getcoordinates(fc)
        fill!(pe, 0.0)
        for q in 1:Ferrite.getnquadpoints(fv)
            eq = e(Ferrite.spatial_coordinate(fv, q, x))
            dS = Ferrite.getdetJdV(fv, q)
            for i in 1:n
                pe[i] += dot(eq, Ferrite.shape_value(fv, q, i)) * dS
            end
        end
        Ferrite.assemble!(profile, Ferrite.celldofs(fc), pe)
    end
    block = _tangential_mass(dh, pattern, [port.facetset], Returns(1.0))
    # The TE10 mode needs a PEC rim: every edge dof on just one port facet must be constrained.
    count = Dict{Int, Int}()
    for (cell, local_facet) in facets
        cell_dofs = Ferrite.celldofs(dh, cell)
        for i in _TET_FACET_EDGE_DOFS[local_facet]
            count[cell_dofs[i]] = get(count, cell_dofs[i], 0) + 1
        end
    end
    all(dof -> count[dof] > 1 || dof in constrained, keys(count)) ||
        throw(ArgumentError("the rim of port $(repr(port.facetset)) is not PEC; name the guide walls in `pec`"))
    free = setdiff(_pec_dofs(dh, grid, [port.facetset]), constrained)
    isempty(free) && throw(ArgumentError("every edge dof of port $(repr(port.facetset)) is PEC-constrained; a port face cannot also be in `pec`"))
    # An interior edge of the face in `pec` blocks part of the aperture, which the full-rectangle TE10 mode ignores.
    any(dof -> count[dof] > 1 && dof in constrained, keys(count)) && throw(
        ArgumentError("part of the face of port $(repr(port.facetset)) is in `pec`; only its rim may be PEC")
    )
    norm_h = dot(profile[free], block[free, free] \ profile[free])
    norm_h > 0 || throw(ArgumentError("the TE10 mode of port $(repr(port.facetset)) has no projection onto its free edge dofs"))
    return WavePortData(port, block, profile, norm_h, normal, a, b, π / (a * length_scale), material.eps_r, material.mu_r)
end

# β = sqrt(k² - kc²) on the branch Im β ≤ 0, so that the outgoing exp(-iβζ) decays below cutoff.
function _propagation_constant(data::WavePortData, mu0eps0, ω)
    β = sqrt(complex(ω^2 * mu0eps0 * data.eps_r * data.mu_r - data.kc^2))
    return imag(β) > 0 ? -β : β
end

function _wave(model, port::Integer)
    data = model.ports[port]
    data isa WavePortData || throw(ArgumentError("port $port of the model is not a wave port"))
    return data
end

"""
    propagation_constant(model::MaxwellModel, port, ω)

The TE10 propagation constant `β` (rad/m) of wave port number `port` of `model` at angular frequency `ω`.
Below cutoff `β = -iα` with `α > 0`, so the mode `exp(-iβζ)` decays away from the port.
"""
propagation_constant(model, port::Integer, ω::Real) =
    _propagation_constant(_wave(model, port), model.mu0 * model.eps0, float(ω))

"""
    cutoff_frequency(model::MaxwellModel, port)

The TE10 cutoff frequency in Hz, `kc / (2π sqrt(μ0 ε0 Re(εr) μr))`, of wave port number `port` of `model`.
"""
function cutoff_frequency(model, port::Integer)
    data = _wave(model, port)
    return data.kc / (2π * sqrt(model.mu0 * model.eps0 * real(data.eps_r) * data.mu_r))
end

# n × μr⁻¹∇ × E = -(iβ/μr) n × (n × E) - 2(iβ/μr) E0 e on the port gives +(iβ/μr) L B on the operator and
# +2(iβ/μr) L E0 p on the load, with E0 = 1.
function _port_term(data::WavePortData, mu0, eps0, L)
    return MaxwellTerm(data.block, ω -> im * _propagation_constant(data, mu0 * eps0, ω) * L / data.mu_r)
end

_excitation(model, data::WavePortData, ω) =
    (2im * _propagation_constant(data, model.mu0 * model.eps0, ω) * model.length_scale / data.mu_r) .* data.profile
# The S-parameter normalisation scales a wave port by sqrt(β/(ωμ)). At exact cutoff β = 0 the wave impedance ωμ/β is
# infinite and the scale is 0/0; at ω = 0, β = -i kc and Z = ωμ/β is 0, and the scale diverges. Both are an error
# rather than a NaN matrix.
function _check_not_at_cutoff(model, ω)
    for (j, data) in pairs(model.ports)
        data isa WavePortData && iszero(ω) && throw(
            ArgumentError("ω = 0 is not supported for wave port $j: the amplitude factor sqrt(β/(ωμ)) of its S-parameter normalisation diverges there; use a positive frequency")
        )
        data isa WavePortData && iszero(_propagation_constant(data, model.mu0 * model.eps0, float(ω))) && throw(
            ArgumentError("ω = $ω is exactly the TE10 cutoff of wave port $j, where its wave impedance is infinite and its S-parameters are undefined; use a nearby frequency")
        )
    end
    return nothing
end
# A 1 A Norton source, -(1/w) l̂ sheet current, gives +iωμ0 L² g / (w L) after division by L.
_excitation(model, data::LumpedPortData, ω) = (im * ω * model.mu0 / data.width) .* data.profile

"""
    port_excitations(model::MaxwellModel, ω)

The load vectors of the ports of `model` at `ω`, one column per port, without the PEC constraints: for a
wave port the TE10 mode of unit amplitude (1 V/m) incident on it, for a lumped port a 1 A current source in
parallel with its `Z0`.
"""
function port_excitations(model, ω::Real)
    B = zeros(ComplexF64, Ferrite.ndofs(model.dh), length(model.ports))
    _check_not_at_cutoff(model, ω)
    for (j, data) in pairs(model.ports)
        B[:, j] .= _excitation(model, data, float(ω))
    end
    return B
end

# Wave coefficients of a port for the excitation of `port_excitations`, power waves (peak phasors, |a|²/2 watts) for a
# lumped port and a propagating wave port, generalized traveling-wave coefficients below cutoff: the incident
# wave `a` when the port is driven and the outgoing wave `b` in the solution `u`, `driven` or not.
function _power_waves(model, data::WavePortData, ω, u, driven::Bool)
    β = _propagation_constant(data, model.mu0 * model.eps0, ω)
    # sqrt(∮ e⋅e dS / Z) with the TE10 wave impedance Z = ωμ0μr / β.
    w = sqrt(data.norm * model.length_scale^2 * β / (ω * model.mu0 * data.mu_r))
    shift = exp(im * β * data.port.deembed * model.length_scale)
    return w / shift, (dot(data.profile, u) / data.norm - driven) * w * shift
end
function _power_waves(model, data::LumpedPortData, ω, u, driven::Bool)
    # a = (V + Z0 I)/(2 sqrt Z0) with I = driven - V/Z0 delivered into the structure.
    Z0 = data.port.Z0
    V = model.length_scale * dot(data.profile, u) / data.width
    return sqrt(Z0) / 2, (2V - Z0 * driven) / (2 * sqrt(Z0))
end

"""
    s_parameters(model::MaxwellModel, ω, U)

The scattering matrix `Sᵢⱼ = bᵢ / aⱼ` from the solutions `U[:, j]` with port `j` driven by
[`port_excitations`](@ref), in wave coefficients, which are power waves for lumped ports and propagating wave ports. A wave port has `aⱼ = sqrt(Pⱼ / Zⱼ) exp(-iβⱼdⱼ)` and
`bᵢ = (pᵢᵀuⱼ / Pᵢ - δᵢⱼ) sqrt(Pᵢ / Zᵢ) exp(iβᵢdᵢ)`, with `Pᵢ = ∮ e⋅e dS`, the TE10 wave impedance
`Zᵢ = ωμ0μr / βᵢ` and the de-embedding distances `dᵢ`. A lumped port has `aⱼ = sqrt(Z0ⱼ) / 2` and
`bᵢ = (2Vᵢ - Z0ᵢ δᵢⱼ) / (2 sqrt(Z0ᵢ))`, so between lumped ports `Sᵢⱼ = 2Vᵢ / sqrt(Z0ᵢ Z0ⱼ) - δᵢⱼ`. At exactly a wave port's cutoff `βᵢ = 0`, `Zᵢ` is
infinite and the S-parameters are undefined, so this throws an `ArgumentError`; so does `ω = 0` with a wave port, where the
normalisation factor `sqrt(βᵢ / (ωμ0μr))` diverges (`βᵢ = -iπ/(aL)` and `Zᵢ = 0` there).
The wave coefficients are power waves for propagating ports only; below cutoff they are generalized traveling-wave
coefficients (see [`WavePort`](@ref)).
"""
function s_parameters(model, ω::Real, U::AbstractMatrix)
    ω = float(ω)
    _check_not_at_cutoff(model, ω)
    ports = model.ports
    np = length(ports)
    S = zeros(ComplexF64, np, np)
    for j in 1:np
        a = first(_power_waves(model, ports[j], ω, view(U, :, j), true))
        for i in 1:np
            S[i, j] = last(_power_waves(model, ports[i], ω, view(U, :, j), i == j)) / a
        end
    end
    return S
end
