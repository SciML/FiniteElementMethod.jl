# Frequency sweep of the wave-port scattering matrix of a MaxwellModel: one operator matrix recombined per
# frequency, one factorisation per frequency for all port excitations.

"""
    umfpack_solver(A, B, cache) -> (X, cache)

The default linear solver of [`solve_sweep`](@ref): an UMFPACK LU factorisation of `A`, reusing the symbolic
analysis of the previous frequency (`cache`, `nothing` on the first call) since the pattern never changes,
applied to all columns of `B`. A replacement solver has the same signature.
"""
function umfpack_solver(A, B, cache)
    F = cache === nothing ? lu(A) : lu!(cache, A)
    return F \ B, F
end

"""
    MaxwellSweep

The result of [`solve_sweep`](@ref): `omegas`, the scattering matrices `S[:, :, k]` at `omegas[k]`
(`S[i, j, k]` is the wave leaving port `i` for a unit wave incident on port `j`) and `fields`, either
`nothing` or, per frequency, the matrix whose column `j` is the edge-dof vector with port `j` driven (wrap it
with `MaxwellSolution(model, omegas[k], fields[k][:, j])`).
"""
struct MaxwellSweep
    omegas::Vector{Float64}
    S::Array{ComplexF64, 3}
    fields::Union{Nothing, Vector{Matrix{ComplexF64}}}
end

"""
    solve_sweep(model::MaxwellModel, omegas; fields = false, solver = umfpack_solver)

The scattering matrix of the [`WavePort`](@ref)s of `model` at each angular frequency in `omegas` (rad/s),
as a [`MaxwellSweep`](@ref). Each frequency recombines the nonzero values of one operator matrix with
[`maxwell_operator!`](@ref), applies the PEC constraints, and solves for all port excitations at once with
`solver(A, B, cache)` (see [`FiniteElementMethod.umfpack_solver`](@ref)), which may keep its factorisation
between frequencies in `cache`. `fields = true` keeps the solutions (`ndofs × nports` complex values per
frequency).
"""
function solve_sweep(model::MaxwellModel, omegas; fields::Bool = false, solver = umfpack_solver)
    isempty(model.ports) && throw(ArgumentError("the model has no wave ports"))
    omegas = float.(collect(omegas))
    np = length(model.ports)
    S = zeros(ComplexF64, np, np, length(omegas))
    kept = fields ? Matrix{ComplexF64}[] : nothing
    A = maxwell_operator(model, first(omegas))
    prescribed = model.ch.prescribed_dofs
    cache = nothing
    for (k, ω) in pairs(omegas)
        maxwell_operator!(A, model, ω)
        Ferrite.apply!(A, model.ch)
        B = port_excitations(model, ω)
        B[prescribed, :] .= 0  # the PEC constraints are homogeneous
        U, cache = solver(A, B, cache)
        S[:, :, k] .= s_parameters(model, ω, U)
        fields && push!(kept, U)
    end
    return MaxwellSweep(omegas, S, kept)
end
