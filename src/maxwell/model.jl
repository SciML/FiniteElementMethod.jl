# Frequency-domain Maxwell model: lowest-order Nédélec edge elements on tetrahedra, assembled directly with
# Ferrite. Independent of the PDESystem front end.

"""
Vacuum permeability in H/m (CODATA 2022).
"""
const MU0 = 1.25663706127e-6
"""
Speed of light in vacuum in m/s (exact).
"""
const C0 = 299792458.0
"""
Vacuum permittivity in F/m, `1 / (MU0 * C0^2)`.
"""
const EPS0 = 1 / (MU0 * C0^2)

# The edge element of every MaxwellModel.
const _NEDELEC = Ferrite.Nedelec{RefTetrahedron, 1}()

"""
    MaxwellMaterial(; eps_r = 1, mu_r = 1, sigma = 0)

An isotropic linear material: relative permittivity `eps_r = ε' - iε''` (real or complex; with the
`exp(+iωt)` convention a passive medium has `ε'' ≥ 0`), real relative permeability `mu_r` and
conductivity `sigma` in S/m. The default is vacuum.
"""
struct MaxwellMaterial
    eps_r::ComplexF64
    mu_r::Float64
    sigma::Float64
end
MaxwellMaterial(; eps_r = 1.0, mu_r = 1.0, sigma = 0.0) = MaxwellMaterial(eps_r, mu_r, sigma)

"""
    MaxwellTerm(block, coefficient)

One term `coefficient(ω) * block` of the system matrix of a [`MaxwellModel`](@ref): `block` is a real sparse
matrix on the model's sparsity pattern and `coefficient` maps the angular frequency to a complex number.
"""
struct MaxwellTerm
    block::SparseMatrixCSC{Float64, Int}
    coefficient::Function
end

"""
    MaxwellModel(grid; materials = Dict(), pec = String[], absorbing = String[], ports = WavePort[],
                 length_scale = 1.0, eps0 = EPS0, mu0 = MU0, quadrature = 2)

The time-harmonic electric-field model on a Ferrite `grid` of `Tetrahedron` cells, with the field `:E` in
`Nedelec{RefTetrahedron, 1}`. The equation, for fields varying as `exp(+iωt)`, is

    ∇ × (μr⁻¹ ∇ × E) + iωμ0 σ E - ω² μ0 ε0 εr E = -iωμ0 J,

with `n × E = 0` on the facet sets named in `pec` and the natural condition `n × (μr⁻¹ ∇ × E) = 0` on the
rest of the boundary, except for the facet sets named in `absorbing`, which carry the first-order
Silver-Müller absorbing condition `n × (μr⁻¹ ∇ × E) = -i k0 sqrt(εc/μr) n × (n × E)` with `εc = εr - iσ/(ωε0)`, and the
[`WavePort`](@ref)s in `ports`.

`materials` maps cell set names of `grid` (as `FerriteGmsh.togrid` makes from Gmsh physical groups) to
[`MaxwellMaterial`](@ref)s; cells in no named set are vacuum, and a cell in two named sets is an error.
It may also be a vector of `MaxwellMaterial`s, one per cell of `grid` in cell order.
Mesh coordinates are in units of `length_scale` metres. `eps0` and `mu0` default to their SI values;
`eps0 = mu0 = 1` with `length_scale = 1` gives normalised units with `c = 1`.

Galerkin discretisation divided through by `length_scale` (`L`) gives `A(ω) u = b`, where `A` is a sum of real
sparse blocks on one sparsity pattern, each with a complex coefficient that is a function of `ω`:

    A(ω) = Kc - ω² μ0 ε0 L² (Mε' + i Mε'') + iω μ0 L² Mσ,

`Kc = ∫ μr⁻¹ ∇×Nᵢ ⋅ ∇×Nⱼ`, `Mε' = ∫ Re(εr) Nᵢ⋅Nⱼ`, `Mε'' = ∫ Im(εr) Nᵢ⋅Nⱼ` and `Mσ = ∫ σ Nᵢ⋅Nⱼ` over the
mesh (so `Im(εr) = -ε''` gives the `+iω²μ0ε0L² ε''` loss term). The absorbing boundary adds
`+i k0 L ∮ sqrt(εc/μr) (n × Nᵢ)⋅(n × Nⱼ) dS` (`k0 = ω sqrt(μ0 ε0)`, material of the adjacent cell) and each
wave port `+i (β(ω)/μr) L ∮ (n × Nᵢ)⋅(n × Nⱼ) dS` over its face. `A` is complex symmetric, not Hermitian.

Fields: `grid`, `dh`, `ch` (zero `AffineConstraint`s on the PEC edge dofs), `cellvalues`, `materials` (one
per cell), `pec`, `eps0`, `mu0`, `length_scale`, `blocks` (the volume blocks `curlcurl`, `eps_real`,
`eps_imag` and `sigma` above), `absorbing`, `ports` (the [`FiniteElementMethod.WavePortData`](@ref) in the
order given) and `terms`, the [`MaxwellTerm`](@ref)s that [`maxwell_operator!`](@ref)
combines; [`FiniteElementMethod.add_term!`](@ref) appends further blocks, such as boundary terms.
"""
struct MaxwellModel{G, DH, CH, CV}
    grid::G
    dh::DH
    ch::CH
    cellvalues::CV
    materials::Vector{MaxwellMaterial}
    pec::Vector{String}
    eps0::Float64
    mu0::Float64
    length_scale::Float64
    blocks::NamedTuple{(:curlcurl, :eps_real, :eps_imag, :sigma), NTuple{4, SparseMatrixCSC{Float64, Int}}}
    terms::Vector{MaxwellTerm}
    absorbing::Vector{String}
    ports::Vector{WavePortData}
end

function MaxwellModel(
        grid::Ferrite.Grid; materials = Dict{String, MaxwellMaterial}(), pec = String[], absorbing = String[],
        ports = WavePort[], length_scale::Real = 1.0, eps0::Real = EPS0, mu0::Real = MU0, quadrature::Int = 2
    )
    Ferrite.getcelltype(grid) === Ferrite.Tetrahedron ||
        throw(ArgumentError("MaxwellModel needs a grid of Tetrahedron cells, got $(Ferrite.getcelltype(grid))"))
    pec = String.(collect(pec))
    absorbing = String.(collect(absorbing))
    foreach(name -> _check_facetset(grid, name), pec)
    foreach(name -> _check_facetset(grid, name), absorbing)
    cell_materials = _cell_materials(grid, materials)

    ip = _NEDELEC
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :E, ip)
    Ferrite.close!(dh)
    ch = Ferrite.ConstraintHandler(dh)
    for dof in _pec_dofs(dh, grid, pec)
        Ferrite.add!(ch, Ferrite.AffineConstraint(dof, Pair{Int, Float64}[], 0.0))
    end
    Ferrite.close!(ch)
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{RefTetrahedron}(quadrature), ip)

    blocks = _assemble_blocks(dh, cv, cell_materials)
    L2 = float(length_scale)^2
    k2 = mu0 * eps0 * L2
    terms = [
        MaxwellTerm(blocks.curlcurl, Returns(1.0 + 0.0im)),
        MaxwellTerm(blocks.eps_real, ω -> complex(-k2 * ω^2)),
        MaxwellTerm(blocks.eps_imag, ω -> -im * k2 * ω^2),
        MaxwellTerm(blocks.sigma, ω -> im * mu0 * L2 * ω),
    ]
    L = float(length_scale)
    isempty(absorbing) || append!(terms, _absorbing_terms(dh, blocks.curlcurl, absorbing, cell_materials, mu0, eps0, L))
    port_data = WavePortData[_wave_port(port, dh, blocks.curlcurl, cell_materials, L, ch.prescribed_dofs) for port in ports]
    append!(terms, [_port_term(data, mu0 * eps0, L) for data in port_data])
    return MaxwellModel(
        grid, dh, ch, cv, cell_materials, pec, float(eps0), float(mu0), L, blocks, terms, absorbing, port_data
    )
end

function _check_cellset(grid, name)
    return haskey(grid.cellsets, name) ||
        throw(ArgumentError("no cell set $(repr(name)) in the grid; it has $(sort!(collect(keys(grid.cellsets))))"))
end

function _cell_materials(grid, materials)
    cell_materials = fill(MaxwellMaterial(), Ferrite.getncells(grid))
    assigned = falses(Ferrite.getncells(grid))
    for (name, material) in materials
        _check_cellset(grid, String(name))
        for cell in Ferrite.getcellset(grid, String(name))
            assigned[cell] && throw(ArgumentError("cell $cell is in more than one material cell set"))
            assigned[cell] = true
            cell_materials[cell] = material
        end
    end
    return cell_materials
end

function _cell_materials(grid, materials::AbstractVector{MaxwellMaterial})
    length(materials) == Ferrite.getncells(grid) || throw(
        ArgumentError("materials has $(length(materials)) entries but the grid has $(Ferrite.getncells(grid)) cells; give one MaxwellMaterial per cell")
    )
    return collect(materials)
end

# The edge dofs on the PEC facets: the tangential traces there are exactly these dofs.
function _pec_dofs(dh, grid, pec)
    local_dofs = Ferrite.dirichlet_facetdof_indices(_NEDELEC)
    dofs = Set{Int}()
    cell_dofs = zeros(Int, Ferrite.ndofs_per_cell(dh))
    for name in pec, facet in Ferrite.getfacetset(grid, name)
        cell, local_facet = facet[1], facet[2]
        Ferrite.celldofs!(cell_dofs, dh, cell)
        for i in local_dofs[local_facet]
            push!(dofs, cell_dofs[i])
        end
    end
    return sort!(collect(dofs))
end

function _assemble_blocks(dh, cv, cell_materials)
    pattern = Ferrite.allocate_matrix(dh)
    mats = (curlcurl = pattern, eps_real = copy(pattern), eps_imag = copy(pattern), sigma = copy(pattern))
    assemblers = map(Ferrite.start_assemble, mats)
    n = Ferrite.getnbasefunctions(cv)
    kc, m = zeros(n, n), zeros(n, n)
    for cc in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cc)
        fill!(kc, 0.0)
        fill!(m, 0.0)
        for q in 1:Ferrite.getnquadpoints(cv)
            dΩ = Ferrite.getdetJdV(cv, q)
            for i in 1:n
                Ni, Ci = Ferrite.shape_value(cv, q, i), Ferrite.shape_curl(cv, q, i)
                for j in 1:n
                    kc[i, j] += dot(Ci, Ferrite.shape_curl(cv, q, j)) * dΩ
                    m[i, j] += dot(Ni, Ferrite.shape_value(cv, q, j)) * dΩ
                end
            end
        end
        (; eps_r, mu_r, sigma) = cell_materials[Ferrite.cellid(cc)]
        dofs = Ferrite.celldofs(cc)
        Ferrite.assemble!(assemblers.curlcurl, dofs, kc ./ mu_r)
        Ferrite.assemble!(assemblers.eps_real, dofs, real(eps_r) .* m)
        Ferrite.assemble!(assemblers.eps_imag, dofs, imag(eps_r) .* m)
        Ferrite.assemble!(assemblers.sigma, dofs, sigma .* m)
    end
    return mats
end

_same_pattern(A, B) = A.colptr == B.colptr && rowvals(A) == rowvals(B)

"""
    add_term!(model::MaxwellModel, block, coefficient)

Append the term `coefficient(ω) * block` to the system matrix of `model`. `block` must have the sparsity
pattern of `model.blocks`, as every matrix from `Ferrite.allocate_matrix(model.dh)` does.
"""
function add_term!(model::MaxwellModel, block::SparseMatrixCSC{Float64, Int}, coefficient)
    _same_pattern(block, model.blocks.curlcurl) ||
        throw(ArgumentError("the block must have the sparsity pattern of the model's blocks"))
    push!(model.terms, MaxwellTerm(block, coefficient))
    return model
end

"""
    maxwell_operator!(A, model::MaxwellModel, ω)

Overwrite the nonzero values of the complex sparse matrix `A`, which must have the sparsity pattern of
`model.blocks`, with `Σₖ cₖ(ω) Bₖ` over the [`MaxwellTerm`](@ref)s of `model`. No constraints are applied.
"""
function maxwell_operator!(A::SparseMatrixCSC{<:Complex}, model::MaxwellModel, ω::Number)
    _same_pattern(A, model.blocks.curlcurl) ||
        throw(ArgumentError("A must have the sparsity pattern of the model's blocks"))
    ω = float(ω)  # an integer ω^2 overflows above ~3e9 rad/s
    nz = nonzeros(A)
    fill!(nz, 0)
    for term in model.terms
        nz .+= term.coefficient(ω) .* nonzeros(term.block)
    end
    return A
end

"""
    maxwell_operator(model::MaxwellModel, ω)

The complex symmetric system matrix `A(ω)` of [`MaxwellModel`](@ref), without the PEC constraints, as a new
`SparseMatrixCSC{ComplexF64}` on the pattern of `model.blocks`.
"""
function maxwell_operator(model::MaxwellModel, ω::Number)
    pattern = model.blocks.curlcurl
    A = SparseMatrixCSC(size(pattern)..., copy(pattern.colptr), copy(rowvals(pattern)), zeros(ComplexF64, nnz(pattern)))
    return maxwell_operator!(A, model, ω)
end
