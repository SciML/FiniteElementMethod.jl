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

"""
    maxwell_eigenmodes(model::MaxwellModel; nev, target_frequency, krylovdim = max(30, 2nev + 10),
                       tol = 1e-10, maxiter = 300, nullspace_tol = 1e-6)

The `nev` resonant modes of `model` whose complex angular frequencies `ω` are nearest `ωt = 2π target_frequency`
(`target_frequency` in Hz, real and positive), that is with the least `|ω - ωt|`, sorted by `Re(ω)`. For a
lossless model this is the nearest eigenfrequency; with loss a strongly damped mode counts as further from the
target than its real frequency alone suggests. Requires `using KrylovKit`.

Source-free modes satisfy `Kc u = λ (Mε' + i Mε'') u` (see [`MaxwellModel`](@ref)) on the dofs off the PEC
facets, which are eliminated rather than constrained, with `λ = ω² μ0 ε0 L²`. The problem is solved by
shift-invert Arnoldi: one sparse LU of `Kc - σ Mε` at `σ` from `target_frequency` (complex when a material is
lossy), with KrylovKit's `eigsolve` on `u ↦ (Kc - σ Mε) \\ (Mε u)`. Shift-invert ranks modes by distance in
`λ ∝ ω²`, so it computes at least `2nev` candidates, enough to contain the `nev` nearest in `ω`, and returns
those; `krylovdim` grows with the candidate count.

Edge elements carry the gradients of the nodal hat functions as an exact `λ = 0` null space (one mode per node
off the PEC facets, less one per connected part of the mesh without PEC), which would swamp the shift-invert for any target below the lowest physical mode. Every
Krylov vector is therefore projected, along those gradients, onto the discretely divergence-free fields, so the
null space never enters; eigenvalues with `|λ| < nullspace_tol ‖Kc‖₁/‖Mε‖₁` (such as the one constant mode of
each further disconnected PEC body) are dropped as well, and the search widens until `nev` modes survive.

Returns `(; frequency, Q, omega, modes)`: `omega` the complex angular frequencies `ω = √λ / (L √(μ0 ε0))`,
`frequency = Re(ω) / 2π` in Hz, `Q = Re(ω) / (2 Im(ω))` (`Inf` for a lossless model; with `exp(+iωt)` a passive
mode has `Im(ω) > 0` and decays as `exp(-Im(ω) t)`), and `modes` the [`MaxwellSolution`](@ref)s at
`Re(ω)`, with zeros on the PEC dofs, scaled to `∫ Re(εr) |E|² = 1` over the mesh (mesh units; to a unit dof
norm where that integral is not positive) with the largest dof real. Conductivity and terms added with
[`FiniteElementMethod.add_term!`](@ref) make the problem nonlinear in `ω` and are rejected. A problem too
small for a Krylov space of dimension `krylovdim` is solved densely, and `nev` may not exceed the number of
modes outside the gradient null space.
"""
function maxwell_eigenmodes(
        model::MaxwellModel; nev::Integer, target_frequency::Real, krylovdim::Integer = max(30, 2nev + 10),
        tol::Real = 1.0e-10, maxiter::Integer = 300, nullspace_tol::Real = 1.0e-6
    )
    nev >= 1 || throw(ArgumentError("nev must be positive, got $nev"))
    0 < target_frequency < Inf ||
        throw(ArgumentError("target_frequency must be positive and finite, got $target_frequency"))
    all(iszero, nonzeros(model.blocks.sigma)) ||
        throw(ArgumentError("maxwell_eigenmodes does not support conductivity (the problem is quadratic in ω); model it as Im(eps_r)"))
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
    # The cutoff for the null space is relative to the spectrum of the model, ‖K‖/‖M‖ of the order of its largest λ,
    # not to the target: relative to a high target it would discard the lowest physical modes, and relative to a
    # low one it would admit the static modes, whose roundoff λ is far above it. A target below the cutoff is
    # raised to it, which also keeps `K - σ M` from being numerically singular; it then has the same nearest modes.
    λ_null = nullspace_tol * opnorm(K, 1) / opnorm(M, 1)
    ωt = max(2π * target_frequency, sqrt(λ_null / scale))
    σ = scale * ωt^2
    # A shift exactly on a discrete eigenvalue makes the factorisation singular; nudge it off.
    F = try
        lu(K - σ * M)
    catch err
        err isa SingularException || rethrow()
        σ *= 1 + 1.0e-8
        lu(K - σ * M)
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
    # |ω - ωt| ≤ d has |λ - σ| = scale |ω - ωt| |ω + ωt| ≤ scale d (2ωt + d), complex ω included, so the
    # candidates (all modes within the largest λ distance found) hold the nev nearest in ω once it reaches that.
    howmany = 2nev
    local λ, vecs, keep
    while true
        dim = max(krylovdim, howmany + 20)
        dense = dim >= length(free)
        θ, vecs = dense ? _dense_eigs(op, length(free), T) :
            _eigsolve(KrylovKitEigensolver(), op, x0, howmany; krylovdim = dim, tol, maxiter)
        λ = σ .+ 1 ./ θ
        lossy || (λ = real.(λ))
        # The dense spectrum also holds the projected-out gradients (θ = 0): their eigenvectors lie in the range of
        # G, which the projection annihilates, while every other eigenvector is left unchanged by it.
        physical(v) = norm(project(v)) > norm(v) / 2
        keep = findall(j -> physical(vecs[j]) && abs(λ[j]) >= λ_null, eachindex(λ))
        dist = abs.(sqrt.(complex.(λ[keep]) ./ scale) .- ωt)
        keep = keep[sortperm(dist)]
        if length(keep) < nev
            dense && break
            howmany = 2howmany  # the filter took too many of the candidates, so look further out
            continue
        end
        d = sort(dist)[nev]
        (dense || maximum(l -> abs(l - σ), λ[keep]) >= (2ωt + d) * d * scale) && break
        howmany = 2howmany
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
