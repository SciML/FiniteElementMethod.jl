# Cartesian uniaxial perfectly matched layer (UPML) of a MaxwellModel, with the frequency-independent
# coordinate stretching s = 1 - iα(x), so that the layer is a fixed set of real blocks with polynomial
# coefficients in ω.

"""
    CartesianPML(cellsets; omega, R = 1e-6, order = 2, box = nothing)

A Cartesian perfectly matched layer for the `pml` keyword of [`MaxwellModel`](@ref): the cells of the named
cell set(s) of the grid, outside an axis-aligned inner box, become the uniaxial anisotropic absorber
`εr → εr Λ`, `σ → σ Λ`, `μr → μr Λ` with

    Λ = diag(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z),   s_j = 1 - i α_j(x_j)   (exp(+iωt)),

which is reflectionless at every angle of incidence on the continuum level and attenuates an outgoing wave of
wavenumber `k` by `exp(-k ∫ α dx)`. `Λ` is diagonal, so the operator stays complex symmetric.

`box = (lo, hi)` gives the corners of the inner (physical) box in mesh units; by default it is the bounding
box of the cells not in the layer. Each of the six faces of the box has the layer thickness `d` from the box
to the bounding box of the layer cells (zero means no stretching across that face), and on it
`α = α_max (ξ/d)^order` at depth `ξ`, with `α_max = -(order + 1) ln R / (2 k_ref d)` and
`k_ref = omega sqrt(μ0 ε0)`: a plane wave of the reference angular frequency `omega` (rad/s) at normal
incidence in vacuum comes back from a PEC-backed layer with amplitude `R`. Edges and corners of the layer
multiply the stretchings of their faces.

The stretching is frequency-independent (`α = σ/omega` frozen at the reference), so the attenuation scales
with `k`: at `ω` the round trip is `R^(k/k_ref)` (in a waveguide `R^(β/k_ref)`). Name the outer faces of the
layer in `pec` to back it with a conductor. Wave ports and absorbing facet sets cannot touch layer cells.
"""
struct CartesianPML
    cellsets::Vector{String}
    omega::Float64
    R::Float64
    order::Float64
    box::Union{Nothing, NTuple{2, Ferrite.Vec{3, Float64}}}
end
function CartesianPML(cellsets; omega::Real, R::Real = 1.0e-6, order::Real = 2, box = nothing)
    names = cellsets isa AbstractString ? [String(cellsets)] : String.(collect(cellsets))
    isempty(names) && throw(ArgumentError("CartesianPML needs at least one cell set"))
    omega > 0 || throw(ArgumentError("the PML reference angular frequency must be positive, got $omega"))
    0 < R < 1 || throw(ArgumentError("the PML reflection R must be in (0, 1), got $R"))
    order >= 0 || throw(ArgumentError("the PML grading order must be nonnegative, got $order"))
    corners = box === nothing ? nothing : map(c -> Ferrite.Vec{3, Float64}(Tuple(c)), (box[1], box[2]))
    corners === nothing || all(corners[1] .< corners[2]) ||
        throw(ArgumentError("the PML box needs lo < hi in every coordinate, got $box"))
    return CartesianPML(names, float(omega), float(R), float(order), corners)
end

function _bounding_box(grid, cells)
    lo, hi = fill(Inf, 3), fill(-Inf, 3)
    for cell in cells, x in Ferrite.getcoordinates(grid, cell)
        lo .= min.(lo, x)
        hi .= max.(hi, x)
    end
    return lo, hi
end

_centroid(grid, cell) = sum(Ferrite.getcoordinates(grid, cell)) / 4

# The stretching α(x) of the layer, from the inner box (lo, hi), the face thicknesses and their α_max.
function _pml_profile(pml::CartesianPML, grid, in_pml, k_ref, L)
    pml_cells = findall(in_pml)
    isempty(pml_cells) && throw(ArgumentError("the PML cell sets $(pml.cellsets) are empty"))
    outer_lo, outer_hi = _bounding_box(grid, pml_cells)
    pml.box === nothing && all(in_pml) && throw(ArgumentError("every cell is in the PML; give its inner `box`"))
    lo, hi = pml.box === nothing ? _bounding_box(grid, findall(!, in_pml)) : (collect(pml.box[1]), collect(pml.box[2]))
    tol = 1.0e-9 * maximum(outer_hi .- outer_lo)
    thickness(t) = t > tol ? t : 0.0
    d_lo, d_hi = thickness.(lo .- outer_lo), thickness.(outer_hi .- hi)
    all(iszero, d_lo) && all(iszero, d_hi) &&
        throw(ArgumentError("the PML cells do not extend beyond the inner box $((lo, hi)); the layer has zero thickness"))
    for cell in eachindex(in_pml)
        x = _centroid(grid, cell)
        outside = any((x .< lo .- tol) .| (x .> hi .+ tol))
        outside == in_pml[cell] ||
            throw(ArgumentError("cell $cell is $(outside ? "outside" : "inside") the inner PML box but $(outside ? "not " : "")in the PML cell sets"))
    end
    α_max(d) = d > 0 ? -(pml.order + 1) * log(pml.R) / (2 * k_ref * L * d) : 0.0
    a_lo, a_hi = α_max.(d_lo), α_max.(d_hi)
    m = pml.order
    return function (x)
        return ntuple(3) do j
            x[j] < lo[j] ? a_lo[j] * min((lo[j] - x[j]) / d_lo[j], 1.0)^m :
                x[j] > hi[j] ? a_hi[j] * min((x[j] - hi[j]) / d_hi[j], 1.0)^m : 0.0
        end
    end
end

# The layer as additions to the isotropic blocks of its cells: (Λ⁻¹ - I)/μr in the curl-curl term and
# (Λ - I) εr, (Λ - I) σ in the mass terms, each split into real blocks for the real and imaginary parts.
function _pml_terms(pml::CartesianPML, dh, cv, pattern, cell_materials, mu0, eps0, L, boundary_sets)
    grid = dh.grid
    in_pml = falses(Ferrite.getncells(grid))
    for name in pml.cellsets
        _check_cellset(grid, name)
        in_pml[collect(Ferrite.getcellset(grid, name))] .= true
    end
    for (kind, names) in boundary_sets, name in names, facet in get(grid.facetsets, name, ())
        in_pml[facet[1]] && throw(ArgumentError("$kind facet set $(repr(name)) touches PML cells"))
    end
    α = _pml_profile(pml, grid, in_pml, pml.omega * sqrt(mu0 * eps0), L)

    names = (:curl_re, :curl_im, :eps_re, :eps_im, :sigma_re, :sigma_im)
    mats = NamedTuple{names}(ntuple(_ -> copy(pattern), 6))
    assemblers = map(Ferrite.start_assemble, mats)
    n = Ferrite.getnbasefunctions(cv)
    kc, m = zeros(ComplexF64, n, n), zeros(ComplexF64, n, n)
    for cc in Ferrite.CellIterator(dh, findall(in_pml))
        Ferrite.reinit!(cv, cc)
        fill!(kc, 0)
        fill!(m, 0)
        x = Ferrite.getcoordinates(cc)
        for q in 1:Ferrite.getnquadpoints(cv)
            s = 1 .- im .* α(Ferrite.spatial_coordinate(cv, q, x))
            Λ = (s[2] * s[3] / s[1], s[1] * s[3] / s[2], s[1] * s[2] / s[3])
            dΩ = Ferrite.getdetJdV(cv, q)
            for i in 1:n
                Ni, Ci = Ferrite.shape_value(cv, q, i), Ferrite.shape_curl(cv, q, i)
                for j in 1:n
                    Nj, Cj = Ferrite.shape_value(cv, q, j), Ferrite.shape_curl(cv, q, j)
                    kc[i, j] += sum(k -> (1 / Λ[k] - 1) * (Ci[k] * Cj[k]), 1:3) * dΩ
                    m[i, j] += sum(k -> (Λ[k] - 1) * (Ni[k] * Nj[k]), 1:3) * dΩ
                end
            end
        end
        (; eps_r, mu_r, sigma) = cell_materials[Ferrite.cellid(cc)]
        dofs = Ferrite.celldofs(cc)
        Ferrite.assemble!(assemblers.curl_re, dofs, real.(kc) ./ mu_r)
        Ferrite.assemble!(assemblers.curl_im, dofs, imag.(kc) ./ mu_r)
        Ferrite.assemble!(assemblers.eps_re, dofs, real.(eps_r .* m))
        Ferrite.assemble!(assemblers.eps_im, dofs, imag.(eps_r .* m))
        Ferrite.assemble!(assemblers.sigma_re, dofs, sigma .* real.(m))
        Ferrite.assemble!(assemblers.sigma_im, dofs, sigma .* imag.(m))
    end
    k2, μL2 = mu0 * eps0 * L^2, mu0 * L^2
    coefficients = (
        Returns(1.0 + 0.0im), Returns(1.0im), ω -> complex(-k2 * ω^2), ω -> -im * k2 * ω^2,
        ω -> im * μL2 * ω, ω -> complex(-μL2 * ω),
    )
    terms = [MaxwellTerm(block, c) for (block, c) in zip(mats, coefficients)]
    return filter(term -> any(!iszero, nonzeros(term.block)), terms)
end
