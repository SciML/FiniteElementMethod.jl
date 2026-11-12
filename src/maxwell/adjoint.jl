# Discrete adjoint sensitivities of a MaxwellModel with respect to the material parameters of named cell sets.
# A is complex symmetric (Aᵀ = A), so the adjoint of a wave port's output is the forward field with that port
# driven, and the S-parameter sensitivities need no solve beyond the forward sweep.

const _PARAMETER_KINDS = (:eps_r, :eps_r_imag, :mu_r, :sigma)

# The adjoint of a port is its forward field only for a symmetric A, but `add_term!` takes any block with the
# model's sparsity pattern. Only the free dofs count: the constrained rows and columns are replaced by the identity,
# so a large entry on a constrained dof must not hide the asymmetry of the operator that is solved.
function _check_symmetric(model)
    free = setdiff(axes(model.blocks.curlcurl, 1), model.ch.prescribed_dofs)
    for (k, term) in pairs(model.terms)
        B = term.block[free, free]
        norm(B - transpose(B), 1) <= 1.0e-12 * norm(B, 1) ||
            throw(ArgumentError("term $k of the model has a nonsymmetric block, so A is not symmetric and the adjoint needs transpose(A); the sensitivities support symmetric terms only"))
    end
    return nothing
end

# ∫ w ∇×Nᵢ⋅∇×Nⱼ (curl = true) or ∫ w Nᵢ⋅Nⱼ over `cells`, with the real weight `w = weight(cell)`, with the
# zeros outside `cells` dropped.
function _region_block(model, cells, curl::Bool, weight)
    block = copy(model.blocks.curlcurl)
    asm = Ferrite.start_assemble(block)
    cv = model.cellvalues
    n = Ferrite.getnbasefunctions(cv)
    Ke = zeros(n, n)
    for cc in Ferrite.CellIterator(model.dh, collect(cells))
        Ferrite.reinit!(cv, cc)
        fill!(Ke, 0.0)
        for q in 1:Ferrite.getnquadpoints(cv)
            dΩ = Ferrite.getdetJdV(cv, q)
            for i in 1:n, j in 1:n
                Ke[i, j] += dΩ * (
                    curl ? dot(Ferrite.shape_curl(cv, q, i), Ferrite.shape_curl(cv, q, j)) :
                        dot(Ferrite.shape_value(cv, q, i), Ferrite.shape_value(cv, q, j))
                )
            end
        end
        Ferrite.assemble!(asm, Ferrite.celldofs(cc), weight(Ferrite.cellid(cc)) .* Ke)
    end
    return dropzeros!(block)
end

# d sqrt(εc/μr) / dp for the material `m`, with εc = εr - iσ/(ωε0).
function _abc_admittance_derivative(kind, m, ω, eps0)
    εc = m.eps_r - im * m.sigma / (ω * eps0)
    y = _admittance(εc, m.mu_r)
    kind === :mu_r && return -y / (2 * m.mu_r)
    dεc = kind === :eps_r ? 1.0 + 0.0im : kind === :eps_r_imag ? 1.0im : -im / (ω * eps0)
    return y / (2εc) * dεc
end

# dA/dp as MaxwellTerms, for p the parameter `kind` of every cell of the cell set `name` shifted together.
function _parameter_terms(model, (kind, name))
    kind in _PARAMETER_KINDS ||
        throw(ArgumentError("unknown material parameter $(repr(kind)); use one of $(_PARAMETER_KINDS)"))
    name = String(name)
    _check_cellset(model.grid, name)
    cells = Ferrite.getcellset(model.grid, name)
    for data in model.ports, (cell, _) in Ferrite.getfacetset(model.grid, data.port.facetset)
        cell in cells && throw(
            ArgumentError(
                "cell set $(repr(name)) touches wave port $(repr(data.port.facetset)), whose mode depends on its material; " *
                    "sensitivities need the ports in fixed material"
            )
        )
    end
    (; mu0, eps0, length_scale) = model
    L2 = length_scale^2
    k2 = mu0 * eps0 * L2
    terms = if kind === :mu_r
        [MaxwellTerm(_region_block(model, cells, true, cell -> -1 / model.materials[cell].mu_r^2), Returns(1.0 + 0.0im))]
    else
        M = _region_block(model, cells, false, Returns(1.0))
        coefficient = kind === :eps_r ? (ω -> complex(-k2 * ω^2)) : kind === :eps_r_imag ? (ω -> -im * k2 * ω^2) :
            (ω -> im * mu0 * L2 * ω)
        [MaxwellTerm(M, coefficient)]
    end
    # The absorbing boundary on cells of the set: +i k0 L sqrt(εc/μr) ∮ (n × Nᵢ)⋅(n × Nⱼ) dS, per material.
    abc_cells = [facet[1] for facetset in model.absorbing for facet in Ferrite.getfacetset(model.grid, facetset) if facet[1] in cells]
    for m in unique(model.materials[abc_cells])
        block = _tangential_mass(
            model.dh, model.blocks.curlcurl, model.absorbing, cell -> Float64(cell in cells && model.materials[cell] == m)
        )
        push!(
            terms, MaxwellTerm(
                dropzeros!(block),
                # 0 at ω = 0, as the forward absorbing coefficient is, where εc = εr - iσ/(ωε0) is 0/0 for σ = 0.
                ω -> iszero(ω) ? 0.0im : im * ω * sqrt(mu0 * eps0) * length_scale * _abc_admittance_derivative(kind, m, ω, eps0)
            )
        )
    end
    return terms
end

# Σₜ cₜ(ω) Bₜ U for the terms of one dA/dp.
_derivative_times(terms, ω, U) = sum(term.coefficient(ω) .* (term.block * U) for term in terms)

"""
    MaxwellSensitivities

The result of [`s_parameter_sensitivities`](@ref): `omegas`, the scattering matrices `S[:, :, k]` as in
[`MaxwellSweep`](@ref), `params` and `dS[i, j, k, l]`, the derivative of `S[i, j, k]` with respect to the real
parameter `params[l]`.
"""
struct MaxwellSensitivities
    omegas::Vector{Float64}
    S::Array{ComplexF64, 3}
    params::Vector{Tuple{Symbol, String}}
    dS::Array{ComplexF64, 4}
end

_normalise_params(params) = Tuple{Symbol, String}[(Symbol(kind), String(name)) for (kind, name) in params]

"""
    s_parameter_sensitivities(model::MaxwellModel, omegas, params; solver = umfpack_solver)
    s_parameter_sensitivities(model::MaxwellModel, sweep::MaxwellSweep, params)

The wave-port scattering matrices of `model` at the angular frequencies `omegas` and their derivatives with
respect to the material parameters `params`, as [`MaxwellSensitivities`](@ref). Each parameter is a pair
`(kind, cellset)`: `kind` is `:eps_r` (the real part of `εr`), `:eps_r_imag` (its imaginary part, `-ε''`),
`:mu_r` or `:sigma`, and the derivative is with respect to that parameter of every cell of the named cell set
shifted by the same amount (the set need not be one of the `materials` keys). The second method reuses the
fields of a sweep run with `fields = true`.

These are the exact derivatives of the discrete S-parameters. Since `A(ω)` is complex symmetric, the adjoint
field of port `i` is the forward field `uᵢ` with port `i` driven, so

    dSᵢⱼ/dp = -uᵢᵀ (dA/dp) uⱼ / (2iβᵢ L Pᵢ / μr,ᵢ) ⋅ (power-wave and de-embedding factors of Sᵢⱼ),

at no linear solve beyond [`solve_sweep`](@ref). `dA/dp` is assembled once per parameter over the cells of its
set, including the absorbing boundary's `sqrt(εc/μr)` on those cells. A cell set adjacent to a wave port is
rejected: the port's mode, excitation and normalisation depend on its material, and are taken as fixed.
Terms added with [`FiniteElementMethod.add_term!`](@ref) are taken as independent of the parameters, and must have
symmetric blocks (an `ArgumentError` otherwise).
"""
function s_parameter_sensitivities(model::MaxwellModel, omegas, params; solver = umfpack_solver)
    params = _normalise_params(params)
    _check_symmetric(model)
    terms = [_parameter_terms(model, param) for param in params]  # validate before solving
    return _s_parameter_sensitivities(model, solve_sweep(model, omegas; fields = true, solver), params, terms)
end

function s_parameter_sensitivities(model::MaxwellModel, sweep::MaxwellSweep, params)
    sweep.fields === nothing &&
        throw(ArgumentError("the sweep has no fields; run solve_sweep with fields = true"))
    all(U -> size(U) == (Ferrite.ndofs(model.dh), length(model.ports)), sweep.fields) ||
        throw(ArgumentError("the sweep's fields do not match the dofs and ports of the model"))
    params = _normalise_params(params)
    _check_symmetric(model)
    terms = [_parameter_terms(model, param) for param in params]
    return _s_parameter_sensitivities(model, sweep, params, terms)
end

function _s_parameter_sensitivities(model, sweep, params, terms)
    (; omegas, S, fields) = sweep
    ports = model.ports
    np = length(ports)
    mu0eps0, L = model.mu0 * model.eps0, model.length_scale
    dS = zeros(ComplexF64, np, np, length(omegas), length(terms))
    for (k, ω) in pairs(omegas)
        U = fields[k]
        β = [_propagation_constant(data, mu0eps0, ω) for data in ports]
        w = [sqrt(data.norm * β[i] / data.mu_r) for (i, data) in pairs(ports)]
        shift = [exp(im * β[i] * data.port.deembed * L) for (i, data) in pairs(ports)]
        # Aλᵢ = pᵢ gives λᵢ = uᵢ / cᵢ, with cᵢ = 2iβᵢL/μr,ᵢ the excitation scale of port_excitations.
        c = [2im * β[i] * L / data.mu_r for (i, data) in pairs(ports)]
        for (l, param_terms) in pairs(terms)
            G = transpose(U) * _derivative_times(param_terms, ω, U)
            for j in 1:np, i in 1:np
                dS[i, j, k, l] = -G[i, j] / (c[i] * ports[i].norm) * w[i] / w[j] * shift[i] * shift[j]
            end
        end
    end
    return MaxwellSensitivities(omegas, S, params, dS)
end

"""
    adjoint_gradient(model::MaxwellModel, ω, u, dJ_du, params; solver = umfpack_solver, cache = nothing)

The gradient of a real functional `J(u)` of the field `u` (edge dofs solving `A(ω) u = b`, with `b`
independent of the parameters, as for [`driven_problem`](@ref) or [`port_excitations`](@ref) away from the
parameter sets) with respect to the material parameters `params` (as for
[`s_parameter_sensitivities`](@ref)), as a vector of real numbers. `dJ_du` is the complex gradient
`∂J/∂Re(u) + i ∂J/∂Im(u)`. One adjoint solve `A λ = conj(dJ_du)` (the transpose of `A` is `A`) gives

    dJ/dp = -Re(λᵀ (dA/dp) u).

The solve is `solver(A, B, cache)` on the constrained `A(ω)`; to reuse an existing factorisation `F` of it, pass
`solver = (A, B, F) -> (F \\ B, F)` and `cache = F`.
"""
function adjoint_gradient(model::MaxwellModel, ω::Real, u::AbstractVector, dJ_du::AbstractVector, params; solver = umfpack_solver, cache = nothing)
    _check_symmetric(model)
    terms = [_parameter_terms(model, param) for param in _normalise_params(params)]
    ω = float(ω)
    A = maxwell_operator(model, ω)
    Ferrite.apply!(A, model.ch)
    rhs = conj.(ComplexF64.(dJ_du))
    rhs[model.ch.prescribed_dofs] .= 0
    λ, _ = solver(A, reshape(rhs, :, 1), cache)
    return Float64[-real(dot(conj(vec(λ)), _derivative_times(param_terms, ω, u))) for param_terms in terms]
end
