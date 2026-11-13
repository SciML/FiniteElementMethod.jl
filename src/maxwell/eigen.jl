# Resonant modes of a MaxwellModel: shift-invert on the PEC-free dofs with the gradient null space deflated.

# Dispatch tag for the eigensolver of the KrylovKit extension.
struct KrylovKitEigensolver end

# `_eigsolve(solver, op, x0, nev; krylovdim, tol, maxiter) -> (values, vectors)`: the `nev` eigenpairs of
# largest magnitude of the linear map `op` from the start vector `x0`. The KrylovKit extension implements it for
# `KrylovKitEigensolver()`; without that package loaded only this fallback exists.
function _eigsolve(solver, op, x0, nev; kwargs...)
    throw(ArgumentError("maxwell_eigenmodes needs an eigensolver: run `using KrylovKit` first"))
end

# Every eigenpair of the linear map `op` on vectors of length `m` and type `T`, from its dense matrix: for problems
# too small for a Krylov space of the requested dimension.
function _dense_eigs(op, m, T)
    E = eigen(reduce(hcat, [op(setindex!(zeros(T, m), one(T), j)) for j in 1:m]))
    return E.values, collect.(eachcol(E.vectors))
end

# The discrete gradient: column j holds the Nédélec dofs of the gradient of the j-th nodal hat function (each
# entry is ±1 or 0, with Ferrite's edge orientation), together with the Lagrange DofHandler of the columns.
function _discrete_gradient(model)
    (; grid, dh) = model
    ip = Ferrite.Lagrange{RefTetrahedron, 1}()
    dhn = Ferrite.DofHandler(grid)
    Ferrite.add!(dhn, :phi, ip)
    Ferrite.close!(dhn)
    qr = Ferrite.QuadratureRule{RefTetrahedron}(2)
    cve, cvn = Ferrite.CellValues(qr, _NEDELEC), Ferrite.CellValues(qr, ip)
    nq, ne, nn = Ferrite.getnquadpoints(qr), Ferrite.getnbasefunctions(cve), Ferrite.getnbasefunctions(cvn)
    N, g = zeros(3nq, ne), zeros(3nq)
    rows, cols, vals = Int[], Int[], Float64[]
    for (ce, cn) in zip(Ferrite.CellIterator(dh), Ferrite.CellIterator(dhn))
        Ferrite.reinit!(cve, ce)
        Ferrite.reinit!(cvn, cn)
        for q in 1:nq, i in 1:ne
            N[(3q - 2):(3q), i] .= Ferrite.shape_value(cve, q, i)
        end
        for a in 1:nn
            for q in 1:nq
                g[(3q - 2):(3q)] .= Ferrite.shape_gradient(cvn, q, a)
            end
            # The gradient lies in the Nédélec space exactly, so the least-squares coefficients are integers.
            c = round.(N \ g)
            for i in 1:ne
                iszero(c[i]) && continue
                push!(rows, Ferrite.celldofs(ce)[i])
                push!(cols, Ferrite.celldofs(cn)[a])
                push!(vals, c[i])
            end
        end
    end
    G = sparse(rows, cols, vals, Ferrite.ndofs(dh), Ferrite.ndofs(dhn), (x, y) -> x)
    return G, dhn
end

# The potential dofs whose hat-function gradients span the gradient null space on the free edge dofs, without
# dependence: every node off the PEC facets, less one node of each connected part of the mesh with no PEC node
# (there the gradients of all its hat functions sum to zero).
function _potential_dofs(dhn, grid, pec)
    on_pec = falses(Ferrite.ndofs(dhn))
    on_pec[_pec_dofs(dhn, grid, pec, Ferrite.Lagrange{RefTetrahedron, 1}())] .= true
    root = collect(1:Ferrite.ndofs(dhn))
    find(i) = (
        while root[i] != i
            i = root[i] = root[root[i]]
        end; i
    )
    for cell in 1:Ferrite.getncells(grid)
        dofs = Ferrite.celldofs(dhn, cell)
        for d in dofs
            root[find(d)] = find(dofs[1])
        end
    end
    pinned = falses(length(root))
    for i in eachindex(root)
        on_pec[i] && (pinned[find(i)] = true)
    end
    drop = copy(on_pec)
    for i in eachindex(root)
        r = find(i)
        pinned[r] || (drop[i] = pinned[r] = true)  # the first node met in an unpinned part
    end
    return findall(!, drop)
end

# x ↦ x - G (GᵀMG)⁻¹ GᵀM x, with transposes, not adjoints: M is complex symmetric when lossy.
function _gradient_projection(G, M)
    GtM = sparse(transpose(G) * M)
    FG = lu(GtM * G)
    return x -> x - G * (FG \ (GtM * x))
end

# The projector that removes the span of `vecs` from a vector in the bilinear form xᵀ M y (the form of the symmetric
# pencil, with the transpose also for complex data): x ↦ x - B G⁺ Bᵀ M x with B the vectors as columns and
# G = Bᵀ M B their Gram matrix. There is no orthonormalisation, so an indefinite form (negative Re(εr)), a vector
# isotropic in it or a dependent one needs no square root or division: the rank-revealing pseudo-inverse G⁺ drops the
# directions that G cannot resolve.
function _deflator(vecs, M)
    B = reduce(hcat, vecs)
    MB = M * B
    Gp = pinv(transpose(B) * MB; rtol = 4 * sqrt(eps()))
    return x -> x - B * (Gp * (transpose(MB) * x))
end

# The largest ratio between the εr (real part) or μr values of the cells.
function _material_contrast(materials)
    spread(values) = maximum(values) / minimum(values)
    return max(spread(abs.(real.(getfield.(materials, :eps_r)))), spread(getfield.(materials, :mu_r)))
end

"""
    maxwell_eigenmodes(model::MaxwellModel; nev, target_frequency, krylovdim = max(30, 2nev + 10),
                       tol = 1e-10, maxiter = 300, nullspace_tol = nothing)

The `nev` resonant modes of `model` whose complex angular frequencies `ω` are nearest `ωt = 2π target_frequency`
(`target_frequency` in Hz, real and positive), that is with the least `|ω - ωt|`, sorted by `Re(ω)`. For a
lossless model this is the nearest eigenfrequency; with loss a strongly damped mode counts as further from the
target than its real frequency alone suggests. Requires `using KrylovKit`.

Source-free modes satisfy `Kc u = λ (Mε' + i Mε'') u` (see [`MaxwellModel`](@ref)) on the dofs off the PEC
facets, which are eliminated rather than constrained, with `λ = ω² μ0 ε0 L²`. The problem is solved by
shift-invert Arnoldi: one sparse LU of `Kc - σ Mε` at `σ` from `target_frequency` (complex when a material is
lossy), with KrylovKit's `eigsolve` on `u ↦ (Kc - σ Mε) \\ (Mε u)`. Shift-invert ranks modes by distance in
`λ ∝ ω²`, so it computes at least `2nev` candidates, enough to contain the `nev` nearest in `ω`, and returns
those; `krylovdim` grows with the candidate count. A Krylov space from one starting vector holds one vector per
distinct eigenvalue, so repeated eigenvalues (such as those of disconnected or symmetric parts) would be found once;
after the first pass the search is therefore restarted from vectors M-orthogonal to everything found, until a restart
finds nothing nearer than the current `nev`-th mode (at most `nev` restarts), which preserves multiplicities. The
extra restart costs about one more Krylov solve with the same factorisation.

Edge elements carry the gradients of the nodal hat functions as an exact `λ = 0` null space (one mode per node
off the PEC facets, less one per connected part of the mesh without PEC), which would swamp the shift-invert for any target below the lowest physical mode. Every
Krylov vector is therefore projected, along those gradients, onto the discretely divergence-free fields, so the
null space never enters; eigenvalues with `|λ| < nullspace_tol ‖Kc‖₁/‖Mε‖₁` (such as the one constant mode of
each further disconnected PEC body) are dropped as well, and the search widens until `nev` modes survive. By default
`nullspace_tol = √eps(Float64)`, the relative precision to which an eigenvalue of the spectrum scale `‖Kc‖₁/‖Mε‖₁`
can be told from zero. This cutoff is a heuristic: a physical mode below it is discarded although it is still
resolved, which happens when the spectrum scale exceeds the lowest mode by more than about `1/√eps`, as with strongly
contrasting materials (a warning says so). A smaller explicit `nullspace_tol`, down to the floating-point zero bound
of about `1e4 eps(Float64)`, recovers such modes; a larger value discards more.

The shift `σ` is raised to at least `1e-6 ‖Kc‖₁/‖Mε‖₁`, which keeps `Kc - σ Mε` from being numerically singular; the
modes are still ranked by distance from the target itself. Where `εr` or `μr` differ by more than a factor 1e6 between cells, the shifted
solve is ill-conditioned and the modes returned may be inaccurate; a warning says so, and also when an explicit
`nullspace_tol` puts the cutoff above the target.

Returns `(; frequency, Q, omega, modes)`: `omega` the complex angular frequencies `ω = √λ / (L √(μ0 ε0))`,
`frequency = Re(ω) / 2π` in Hz, `Q = Re(ω) / (2 Im(ω))` (`Inf` for a lossless model; with `exp(+iωt)` a passive
mode has `Im(ω) > 0` and decays as `exp(-Im(ω) t)`), and `modes` the [`MaxwellSolution`](@ref)s at
`Re(ω)`, with zeros on the PEC dofs, scaled to `∫ Re(εr) |E|² = 1` over the mesh (mesh units; to a unit dof
norm where that integral is not positive) with the largest dof real. Conductivity and terms added with
[`FiniteElementMethod.add_term!`](@ref) make the problem nonlinear in `ω` and are rejected, and so is any material
with `real(εr) ≤ 0` or `μr ≤ 0` (the gradient projection needs a definite mass form; loss as `Im(εr)` is fine). A problem too
small for a Krylov space of dimension `krylovdim` is solved densely, and `nev` may not exceed the number of
modes outside the gradient null space.
"""
function maxwell_eigenmodes(
        model::MaxwellModel; nev::Integer, target_frequency::Real, krylovdim::Integer = max(30, 2nev + 10),
        tol::Real = 1.0e-10, maxiter::Integer = 300, nullspace_tol::Union{Nothing, Real} = nothing
    )
    nev >= 1 || throw(ArgumentError("nev must be positive, got $nev"))
    0 < target_frequency < Inf ||
        throw(ArgumentError("target_frequency must be positive and finite, got $target_frequency"))
    all(iszero, nonzeros(model.blocks.sigma)) ||
        throw(ArgumentError("maxwell_eigenmodes does not support conductivity (the problem is quadratic in ω); model it as Im(eps_r)"))
    bad = findfirst(m -> !(real(m.eps_r) > 0 && m.mu_r > 0), model.materials)
    bad === nothing || throw(
        ArgumentError(
            "maxwell_eigenmodes supports ordinary passive materials only: cell $bad (cell sets " *
                "$(sort!([name for (name, cells) in model.grid.cellsets if bad in cells]))) has eps_r = $(model.materials[bad].eps_r) " *
                "and mu_r = $(model.materials[bad].mu_r), but real(eps_r) and mu_r must be positive (loss as Im(eps_r) is fine)"
        )
    )
    length(model.terms) == 4 ||
        throw(ArgumentError("maxwell_eigenmodes supports only the volume terms of MaxwellModel, not terms added with add_term!"))
    (; blocks, eps0, mu0, length_scale) = model
    scale = mu0 * eps0 * length_scale^2  # λ = scale ω²
    lossy = !all(iszero, nonzeros(blocks.eps_imag))
    T = lossy ? ComplexF64 : Float64
    n = Ferrite.ndofs(model.dh)
    free = setdiff(1:n, _pec_dofs(model.dh, model.grid, model.pec))
    isempty(free) && throw(ArgumentError("every edge dof of the model is on a PEC facet, so it has no modes"))
    K = blocks.curlcurl[free, free]
    M = lossy ? blocks.eps_real[free, free] + im * blocks.eps_imag[free, free] : blocks.eps_real[free, free]
    # Classifying a mode as static is a question of precision: its λ is zero up to roundoff in the spectrum scale
    # ‖K‖/‖M‖, so the cutoff is relative to that scale, not to the target (a high target would discard the lowest
    # physical modes) and not to a conditioning bound (the shift floor below, which is far above the lowest modes
    # of a model with strongly contrasting materials). It is a heuristic: sqrt(eps) of the scale can still lie above
    # a resolved mode, which a smaller explicit nullspace_tol recovers.
    spectrum_scale = opnorm(K, 1) / opnorm(M, 1)
    λ_null = something(nullspace_tol, sqrt(eps())) * spectrum_scale
    # A shift near zero makes K - σ M numerically singular, because K itself is: raise a low target to a floor. It
    # has the same nearest modes.
    λ_shift = 1.0e-6 * spectrum_scale
    contrast = _material_contrast(model.materials)
    contrast > 1.0e6 && @warn "maxwell_eigenmodes: the materials of the model differ by a factor $contrast in εr or μr, so the shift-invert is ill-conditioned and modes may be inaccurate"
    nullspace_tol !== nothing && λ_null > scale * (2π * target_frequency)^2 && @warn "maxwell_eigenmodes: nullspace_tol puts the null-space cutoff $λ_null above the target, so the modes there are discarded"
    ωt = 2π * target_frequency
    σ0 = max(scale * ωt^2, λ_shift)
    shift_offset = σ0 - scale * ωt^2
    # A shift exactly on a discrete eigenvalue makes the factorisation singular; nudge it off.
    F, σ = try
        lu(K - σ0 * M), σ0
    catch err
        err isa SingularException || rethrow()
        σ1 = σ0 * (1 + 1.0e-8)
        lu(K - σ1 * M), σ1
    end

    # The gradients of the hat functions of nodes off the PEC facets vanish on the PEC edges; project along them.
    Gfull, dhn = _discrete_gradient(model)
    G = Gfull[free, _potential_dofs(dhn, model.grid, model.pec)]
    nphysical = length(free) - size(G, 2)  # the dimension of the discretely divergence-free fields
    nev <= nphysical ||
        throw(ArgumentError("nev = $nev exceeds the $nphysical modes of the model outside the gradient null space"))
    # With every node on a PEC facet no gradient lies in the free dofs, and there is nothing to project out.
    project = size(G, 2) == 0 ? identity : _gradient_projection(G, M)
    op(x) = project(F \ (M * x))

    x0 = project(T.(sin.(1.3 .* (1:length(free)) .^ 1.1)))
    # Shift-invert finds the modes nearest σ in λ ∝ ω², which favours those below the target. Every mode with
    # |ω - ωt| ≤ d has |λ - σ| ≤ scale |ω - ωt| |ω + ωt| + |σ - scale ωt²| ≤ scale d (2ωt + d) + shift_offset,
    # complex ω included, so the candidates (all modes within the largest λ distance found) hold the nev nearest
    # in ω once it reaches that.
    # The dense spectrum also holds the projected-out gradients (θ = 0): their eigenvectors lie in the range of G,
    # which the projection annihilates, while every other eigenvector is left unchanged by it.
    physical(v) = norm(project(v)) > norm(v) / 2
    # The candidates of the operator `op_s` from `start`, widened until at least `need` of them survive the filters
    # and, with the distance `d` (by default that of the `need`-th), they reach the bound above, or the space is
    # exhausted (dense).
    function search(op_s, start, need, d_fixed = nothing)
        local θ, λ, vecs, keep, dense, dist  # not the variables of the same names of the enclosing function
        howmany = 2nev
        while true
            dim = max(krylovdim, howmany + 20)
            dense = dim >= length(free)
            θ, vecs = dense ? _dense_eigs(op_s, length(free), T) :
                _eigsolve(KrylovKitEigensolver(), op_s, start, howmany; krylovdim = dim, tol, maxiter)
            λ = σ .+ 1 ./ θ
            lossy || (λ = real.(λ))
            keep = findall(j -> isfinite(λ[j]) && physical(vecs[j]) && abs(λ[j]) >= λ_null, eachindex(λ))
            dist = abs.(sqrt.(complex.(λ[keep]) ./ scale) .- ωt)
            keep = keep[sortperm(dist)]
            if length(keep) < need
                dense && return λ, vecs, keep, dense
                howmany = 2howmany  # the filter took too many of the candidates, so look further out
                continue
            end
            d = something(d_fixed, sort(dist)[need])
            (dense || maximum(l -> abs(l - σ), λ[keep]) >= (2ωt + d) * d * scale + shift_offset) &&
                return λ, vecs, keep, dense
            howmany = 2howmany
        end
        return
    end
    λ, vecs, keep, dense = search(op, x0, nev)
    if !dense && length(keep) >= nev
        # A Krylov space from one starting vector holds one vector per distinct eigenvalue, so a repeated eigenvalue
        # may have been found once only. Restart from vectors M-orthogonal to everything found, widening the
        # candidates like the first search, until a pass finds nothing nearer than the current nev-th mode: each pass
        # can add one copy of a repeated eigenvalue.
        pool_λ, pool_vecs = λ[keep], vecs[keep]
        pool_dist = abs.(sqrt.(complex.(pool_λ) ./ scale) .- ωt)
        for pass in 1:nev
            d_nev = sort(pool_dist)[nev]
            deflate = _deflator(pool_vecs, M)
            raw = project(T.(cos.(2.399963 .* pass .* (1:length(free)) .^ 1.37 .+ pass)))
            start = deflate(raw)
            norm(start) > 1.0e-8 * norm(raw) || break  # nothing is left outside the span of what was found
            λ_found, found, new, _ = search(x -> deflate(op(x)), start, 1, d_nev)
            dist_new = abs.(sqrt.(complex.(λ_found[new]) ./ scale) .- ωt)
            any(<(d_nev * (1 - 1.0e-8)), dist_new) || break
            append!(pool_λ, λ_found[new])
            append!(pool_vecs, found[new])
            append!(pool_dist, dist_new)
        end
        λ, vecs = pool_λ, pool_vecs
        keep = sortperm(pool_dist)
    end
    length(keep) < nev && @warn "maxwell_eigenmodes found $(length(keep)) of $nev modes"
    keep = keep[1:min(nev, end)]
    keep = keep[sortperm(real.(sqrt.(complex.(λ[keep]))))]

    omega = sqrt.(complex.(λ[keep])) ./ sqrt(scale)
    Q = [iszero(imag(ω)) ? Inf : real(ω) / (2imag(ω)) for ω in omega]
    Mr = blocks.eps_real[free, free]
    modes = map(keep, omega) do j, ω
        v = ComplexF64.(vecs[j])
        w = real(dot(v, Mr * v))
        v ./= w > 0 ? sqrt(w) : norm(v)  # Re(εr) ≤ 0 somewhere can make the weighted norm vanish
        v .*= conj(sign(v[argmax(abs.(v))]))
        u = zeros(ComplexF64, n)
        u[free] = v
        MaxwellSolution(model, real(ω), u)
    end
    return (; frequency = real.(omega) ./ 2π, Q, omega, modes)
end
