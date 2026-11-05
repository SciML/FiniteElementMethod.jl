# Resonant modes of a MaxwellModel: shift-invert on the PEC-free dofs with the gradient null space deflated.

# Dispatch tag for the eigensolver of the KrylovKit extension.
struct KrylovKitEigensolver end

# `_eigsolve(solver, op, x0, nev; krylovdim, tol, maxiter) -> (values, vectors)`: the `nev` eigenpairs of
# largest magnitude of the linear map `op` from the start vector `x0`. The KrylovKit extension implements it for
# `KrylovKitEigensolver()`; without that package loaded only this fallback exists.
function _eigsolve(solver, op, x0, nev; kwargs...)
    throw(ArgumentError("maxwell_eigenmodes needs an eigensolver: run `using KrylovKit` first"))
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

"""
    maxwell_eigenmodes(model::MaxwellModel; nev, target_frequency, krylovdim = max(30, 2nev + 10),
                       tol = 1e-10, maxiter = 300, nullspace_tol = 1e-6)

The `nev` resonant modes of `model` with eigenfrequencies nearest `target_frequency` (Hz, real and positive),
sorted by frequency. Requires `using KrylovKit`.

Source-free modes satisfy `Kc u = λ (Mε' + i Mε'') u` (see [`MaxwellModel`](@ref)) on the dofs off the PEC
facets, which are eliminated rather than constrained, with `λ = ω² μ0 ε0 L²`. The problem is solved by
shift-invert Arnoldi: one sparse LU of `Kc - σ Mε` at `σ` from `target_frequency` (complex when a material is
lossy), with KrylovKit's `eigsolve` on `u ↦ (Kc - σ Mε) \\ (Mε u)`.

Edge elements carry the gradients of the nodal hat functions as an exact `λ = 0` null space (one mode per node
off the PEC facets), which would swamp the shift-invert for any target below the lowest physical mode. Every
Krylov vector is therefore projected, along those gradients, onto the discretely divergence-free fields, so the
null space never enters; eigenvalues with `|λ| < nullspace_tol |σ|` (such as the one constant mode of each
further disconnected PEC body) are dropped as well.

Returns `(; frequency, Q, omega, modes)`: `omega` the complex angular frequencies `ω = √λ / (L √(μ0 ε0))`,
`frequency = Re(ω) / 2π` in Hz, `Q = Re(ω) / (2 Im(ω))` (`Inf` for a lossless model; with `exp(+iωt)` a passive
mode has `Im(ω) > 0` and decays as `exp(-Im(ω) t)`), and `modes` the [`MaxwellSolution`](@ref)s at
`Re(ω)`, with zeros on the PEC dofs, scaled to `∫ Re(εr) |E|² = 1` over the mesh (mesh units) with the
largest dof real. Conductivity and terms added with [`FiniteElementMethod.add_term!`](@ref) make the problem
nonlinear in `ω` and are rejected.
"""
function maxwell_eigenmodes(
        model::MaxwellModel; nev::Integer, target_frequency::Real, krylovdim::Integer = max(30, 2nev + 10),
        tol::Real = 1.0e-10, maxiter::Integer = 300, nullspace_tol::Real = 1.0e-6
    )
    target_frequency > 0 || throw(ArgumentError("target_frequency must be positive, got $target_frequency"))
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
    K = blocks.curlcurl[free, free]
    M = lossy ? blocks.eps_real[free, free] + im * blocks.eps_imag[free, free] : blocks.eps_real[free, free]
    σ = scale * (2π * target_frequency)^2
    F = lu(K - σ * M)

    # The gradients of the hat functions of nodes off the PEC facets vanish on the PEC edges; project along them
    # with x ↦ x - G (GᵀMG)⁻¹ GᵀM x (transposes, not adjoints: M is complex symmetric when lossy).
    Gfull, dhn = _discrete_gradient(model)
    nodes = setdiff(1:Ferrite.ndofs(dhn), _pec_dofs(dhn, model.grid, model.pec, Ferrite.Lagrange{RefTetrahedron, 1}()))
    G = Gfull[free, nodes]
    GtM = sparse(transpose(G) * M)
    FG = lu(GtM * G)
    project(x) = x - G * (FG \ (GtM * x))
    op(x) = project(F \ (M * x))

    x0 = project(T.(sin.(1.3 .* (1:length(free)) .^ 1.1)))
    θ, vecs = _eigsolve(KrylovKitEigensolver(), op, x0, nev; krylovdim, tol, maxiter)
    λ = σ .+ 1 ./ θ
    lossy || (λ = real.(λ))
    keep = findall(l -> abs(l) >= nullspace_tol * σ, λ)
    length(keep) < nev && @warn "maxwell_eigenmodes found $(length(keep)) of $nev modes"
    keep = keep[sortperm(abs.(θ[keep]); rev = true)][1:min(nev, end)]  # nearest the shift
    keep = keep[sortperm(real.(λ[keep]))]

    omega = sqrt.(complex.(λ[keep])) ./ sqrt(scale)
    Q = [iszero(imag(ω)) ? Inf : real(ω) / (2imag(ω)) for ω in omega]
    Mr = blocks.eps_real[free, free]
    modes = map(keep, omega) do j, ω
        v = ComplexF64.(vecs[j])
        v ./= sqrt(real(dot(v, Mr * v)))
        v .*= conj(sign(v[argmax(abs.(v))]))
        u = zeros(ComplexF64, n)
        u[free] = v
        MaxwellSolution(model, real(ω), u)
    end
    return (; frequency = real.(omega) ./ 2π, Q, omega, modes)
end
