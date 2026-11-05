# A strip dipole near half a wavelength, on a coarse graded Gmsh mesh with named physical groups: the lumped-port
# impedance against frequency, the resonance and the far field at resonance, with loose bands. The full-size run
# (1.4e5 dofs, 8 GB peak, under a minute) is gated by FEM_MAXWELL_DIPOLE=1.
using FiniteElementMethod, Ferrite, LinearAlgebra, Test
include(joinpath(@__DIR__, "..", "..", "examples", "dipole", "dipole.jl"))

# Thin half-wave dipole targets (sinusoidal current, η0 = 376.73 Ω): D = 4 / Cin(2π) and its beamwidth.
const D_HALF_WAVE = 1.6409
const HPBW_HALF_WAVE = 78.08

function report(name, r)
    @info name r.ndofs r.build r.analysis r.resonant_length r.Z_res r.P_in r.P_flux r.P_far D = Tuple(r.D_broadside) r.hpbw r.pattern_err_half_wave r.pattern_err_sinusoidal
    @info "$name: Z(f)" collect(zip(r.frequencies ./ DIPOLE_F0, r.Z))
    return r
end

@testset "coarse strip dipole" begin
    r = report("coarse dipole", run_dipole(:coarse; max_dofs = 3.0e4))
    @test 0.42 < r.resonant_length < 0.48
    @test 45 < real(r.Z_res) < 90
    @test abs(imag(r.Z_res)) < 5
    # Lossless: the port's input power leaves through the Huygens surface.
    @test r.P_flux ≈ r.P_in rtol = 0.05
    # The absorbing sphere at 0.6λ reflects about 1/(kR)² = 7% of the field back onto the dipole, which the
    # flux and the impedance see and the outgoing far field does not.
    @test r.P_far ≈ r.P_flux rtol = 0.15
    @test r.D_broadside ≈ [D_HALF_WAVE, D_HALF_WAVE] rtol = 0.05
    @test r.hpbw ≈ HPBW_HALF_WAVE atol = 4
    @test r.pattern_err_half_wave < 0.5
end

if get(ENV, "FEM_MAXWELL_DIPOLE", "0") == "1"
    @testset "full strip dipole" begin
        r = report("full dipole", run_dipole(:full))
        @test r.P_flux ≈ r.P_in rtol = 0.02
        @test r.D_broadside ≈ [D_HALF_WAVE, D_HALF_WAVE] rtol = 0.023  # ±0.1 dB
        @test r.pattern_err_half_wave < 0.2
    end
end
