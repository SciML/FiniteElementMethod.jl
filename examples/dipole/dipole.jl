# A strip dipole fed by a lumped port, radiating inside a first-order absorbing sphere: input impedance against
# frequency, the resonance, and the far field (pattern, directivity, half-power beamwidth) at resonance.
#
#     julia --project=<test environment> examples/dipole/dipole.jl
#
# Mesh units are metres and the reference frequency is f0 = c / (1 m), so lengths are in wavelengths at f0. The
# dipole is a zero-thickness PEC strip along z in the plane y = 0, of width w = 4a (the equivalent radius of a
# flat strip is a quarter of its width), with a gap of length `gap` at its centre carrying a 50 Ω LumpedPort.
using FiniteElementMethod, Ferrite, LinearAlgebra
include(joinpath(@__DIR__, "mesh.jl"))

const DIPOLE_F0 = FiniteElementMethod.C0  # Hz: λ = 1 m

"""
    dipole_model(; length, radius, gap, outer_radius, huygens_radius, h, h_wire = 2radius, growth = 0.3,
                 max_dofs = 3.0e5)

The [`MaxwellModel`](@ref) of a strip dipole of `length` and equivalent `radius` (width `4radius`) with a
centre `gap`, in a ball of radius `outer_radius` with the absorbing condition on `"outer"`, a Huygens sphere
`"huygens"` of radius `huygens_radius`, and element size `h_wire` on the strip growing to `h`. A mesh above
`max_dofs` edge dofs is refused before anything is assembled on it.
"""
function dipole_model(;
        length, radius, gap, outer_radius, huygens_radius, h, h_wire = 2radius, growth = 0.3, max_dofs = 3.0e5
    )
    grid = antenna_grid(;
        outer_radius, huygens_radius, h, h_fine = h_wire, growth, strip = (; length, width = 4radius, gap)
    )
    dh = DofHandler(grid)
    add!(dh, :E, Nedelec{RefTetrahedron, 1}())
    close!(dh)
    ndofs(dh) <= max_dofs || error("the dipole mesh has $(ndofs(dh)) edge dofs, above max_dofs = $max_dofs")
    return MaxwellModel(
        grid; pec = ["strip"], absorbing = ["outer"], ports = [LumpedPort("gap"; direction = (0, 0, 1), Z0 = 50.0)]
    )
end

# The sinusoidal-current pattern of a thin dipole of electrical half-length kl/2, |E_θ| up to a constant.
thin_dipole_pattern(θ, half_kl) = abs((cos(half_kl * cos(θ)) - cos(half_kl)) / sin(θ))

# The half-power beamwidth (degrees) of a power pattern `D` sampled at increasing `θs` (rad) on (0, π/2]
# ending at broadside, by linear interpolation of the half-power crossing.
function half_power_beamwidth(θs, D)
    i = findfirst(>=(D[end] / 2), D)
    (i === nothing || i == 1) && error("the pattern does not cross half power on the sampled range")
    t = (D[end] / 2 - D[i - 1]) / (D[i] - D[i - 1])
    return rad2deg(π - 2 * (θs[i - 1] + t * (θs[i] - θs[i - 1])))
end

"""
    dipole_analysis(model; frequencies, length)

Input impedance `Z(f)` of the dipole `model` of `length` (mesh units) at `frequencies` (Hz), the resonance (where `Im Z` changes sign,
linearly interpolated and then solved again there), and at resonance: `Z`, the input and radiated powers, the
directivity at broadside, the half-power beamwidth and the E-plane pattern (dB, normalised at broadside) on
`θ = 30°:5°:150°` against `cos(π/2 cos θ)/sin θ` (the half-wave target) and against the sinusoidal-current
pattern of the dipole's actual electrical length.
"""
function dipole_analysis(model; frequencies, length)
    sweep = solve_sweep(model, 2π .* frequencies)
    Z = [only(FiniteElementMethod.z_parameters(model, sweep.S[:, :, k])) for k in eachindex(frequencies)]
    k = findfirst(i -> sign(imag(Z[i])) != sign(imag(Z[i + 1])), 1:(Base.length(Z) - 1))
    k === nothing && error("no resonance in the sweep: Im Z = $(imag.(Z))")
    t = imag(Z[k]) / (imag(Z[k]) - imag(Z[k + 1]))
    f_res = frequencies[k] + t * (frequencies[k + 1] - frequencies[k])

    at_res = solve_sweep(model, [2π * f_res]; fields = true)
    Z_res = only(FiniteElementMethod.z_parameters(model, at_res.S[:, :, 1]))
    sol = MaxwellSolution(model, 2π * f_res, at_res.fields[1][:, 1])
    V = FiniteElementMethod.port_voltage(model, 1, sol.u)
    I_in = FiniteElementMethod.port_current(model, 1, sol.u; source = 1)
    P_in = real(V * conj(I_in)) / 2
    P_flux = radiated_power(sol, "huygens")
    P_far = radiated_power(sol, "huygens"; method = :far_field)

    E_plane(θ) = Vec(sin(θ), 0.0, cos(θ))
    D_broadside = directivity(sol, "huygens", [Vec(1.0, 0.0, 0.0), Vec(0.0, 1.0, 0.0)])
    θs = deg2rad.(30:5:150)
    pattern = directivity(sol, "huygens", E_plane.(θs))
    pattern_dB = 10 .* log10.(pattern ./ directivity(sol, "huygens", [E_plane(π / 2)]))
    half_wave_dB = 20 .* log10.(thin_dipole_pattern.(θs, π / 2))
    half_kl = π * length * f_res / DIPOLE_F0
    sinusoidal_dB = 20 .* log10.(thin_dipole_pattern.(θs, half_kl) ./ thin_dipole_pattern(π / 2, half_kl))
    θ_fine = deg2rad.(1:0.25:90)
    hpbw = half_power_beamwidth(θ_fine, directivity(sol, "huygens", E_plane.(θ_fine)))
    return (;
        frequencies, Z, f_res, resonant_length = length * f_res / DIPOLE_F0, Z_res, P_in, P_flux, P_far,
        D_broadside, hpbw, θ_deg = rad2deg.(θs), pattern_dB,
        pattern_err_half_wave = maximum(abs, pattern_dB - half_wave_dB),
        pattern_err_sinusoidal = maximum(abs, pattern_dB - sinusoidal_dB), ndofs = ndofs(model.dh),
    )
end

"Strip dipole meshes (lengths in λ at f0): `coarse` (~2e4 dofs, for the Maxwell test group) and `full` (~1.4e5, 8 GB peak)."
const DIPOLE_CASES = (
    coarse = (;
        length = 0.48, radius = 5.0e-3, gap = 0.02, outer_radius = 0.6, huygens_radius = 0.4, h = 1 / 7,
        h_wire = 0.0075, frequencies = DIPOLE_F0 .* (0.85:0.05:1.0),
    ),
    full = (;
        length = 0.48, radius = 5.0e-3, gap = 0.01, outer_radius = 1.0, huygens_radius = 0.5, h = 1 / 12,
        h_wire = 0.00125, growth = 0.4, frequencies = DIPOLE_F0 .* (0.91:0.015:0.97),
    ),
)

"""
    run_dipole(case; max_dofs = 3.0e5)

Build and analyse one of [`DIPOLE_CASES`](@ref) (`:coarse` or `:full`), with the build and analysis times.
"""
function run_dipole(case::Symbol; max_dofs = 3.0e5)
    frequencies = DIPOLE_CASES[case].frequencies
    mesh = Base.structdiff(DIPOLE_CASES[case], (; frequencies))
    build = @elapsed model = dipole_model(; mesh..., max_dofs)
    analysis = @elapsed result = dipole_analysis(model; frequencies, mesh.length)
    return (; result..., build, analysis)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_dipole(Symbol(get(ARGS, 1, "full")))
    @info "strip dipole" result...
end
