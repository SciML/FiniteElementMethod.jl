# Near-to-far-field transform against the exact radiation of a small z-directed current ball (a Hertzian
# dipole): λ = 1 m, an absorbing sphere of radius 0.75λ and a Huygens sphere of radius 0.3λ.
using FiniteElementMethod, Ferrite, LinearAlgebra, Test
include(joinpath(@__DIR__, "..", "..", "examples", "dipole", "mesh.jl"))

const OMEGA = 2π * FiniteElementMethod.C0  # λ = 1 m
const K0 = 2π
const ETA = sqrt(FiniteElementMethod.MU0 / FiniteElementMethod.EPS0)

grid = antenna_grid(; outer_radius = 0.75, huygens_radius = 0.3, h = 1 / 12, h_fine = 0.025, source_radius = 0.05)
model = MaxwellModel(grid; absorbing = ["outer"])
sol = solve_driven(model, OMEGA, Dict("source" => (0.0, 0.0, 1.0)))  # J = ẑ A/m²
@info "Hertzian dipole mesh" getncells(grid) ndofs(model.dh)

# The exact far field of the discrete source, F_θ = (ikη/4π) sin θ ∫ J_z exp(ik r̂⋅x) dV, by the cells' quadrature.
# The input power of the source is -½ Re ∫ E⋅J̄ dV.
cv = CellValues(QuadratureRule{RefTetrahedron}(4), Nedelec{RefTetrahedron, 1}())
source_points = Tuple{Vec{3, Float64}, Float64}[]
P_in = 0.0
for cc in CellIterator(model.dh, collect(getcellset(grid, "source")))
    reinit!(cv, cc)
    u_cell = sol.u[celldofs(cc)]
    for q in 1:getnquadpoints(cv)
        push!(source_points, (spatial_coordinate(cv, q, getcoordinates(cc)), getdetJdV(cv, q)))
        global P_in -= real(function_value(cv, q, u_cell)[3]) * getdetJdV(cv, q) / 2
    end
end
exact_E_theta(r̂) = im * K0 * ETA / 4π * sqrt(1 - r̂[3]^2) * sum(dV * cis(K0 * dot(r̂, x)) for (x, dV) in source_points)

@testset "Hertzian dipole far field" begin
    θs = deg2rad.(10:10:170)
    directions = [Vec(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for φ in (0.0, π / 3, π) for θ in θs]
    ff = far_field(sol, "huygens", directions)
    exact = exact_E_theta.(directions)
    err = maximum(abs, ff.E_theta - exact) / maximum(abs, exact)
    @info "Hertzian far field" err maximum(abs, ff.E_phi) / maximum(abs, exact)
    @test err < 0.05
    @test maximum(abs, ff.E_phi) < 0.02 * maximum(abs, exact)

    D = directivity(sol, "huygens", [Vec(1.0, 0.0, 0.0), Vec(0.0, -1.0, 0.0)])
    P_flux = radiated_power(sol, "huygens")
    P_far = radiated_power(sol, "huygens"; method = :far_field)
    P_exact = sum(abs2(exact_E_theta(Vec(sin(θ), 0.0, cos(θ)))) / (2 * ETA) * 2π * sin(θ) * (π / 200) for θ in (0.5:200) .* (π / 200))
    @info "Hertzian power and directivity" D P_in P_flux P_far P_exact
    @test D ≈ [1.5, 1.5] rtol = 0.01
    @test P_far ≈ P_exact rtol = 0.03
    # The flux counts the net power, which the wave the absorbing sphere reflects back onto the source changes
    # by about 2|Γ| ≈ 1/(kR)² (5% at R = 0.75λ, 10% at λ/2); the far field sees only the outgoing wave.
    @test P_flux ≈ P_in rtol = 0.02
    @test P_flux ≈ P_far rtol = 0.08
    # The pole is regular: θ̂ = x̂ there and the z-dipole radiates nothing along its axis.
    pole = far_field(sol, "huygens", [Vec(0.0, 0.0, 2.0), Vec(0.0, 0.0, -1.0)])
    @test all(isfinite, pole.E_theta) && maximum(abs, pole.E_theta) < 0.02 * maximum(abs, exact)
end

@testset "Huygens surface orientation and degenerate inputs" begin
    # The set lists each facet from both cells; listing it from either one alone gives the same far field.
    directions = [Vec(1.0, 0.0, 0.0), Vec(0.6, 0.0, 0.8)]
    reference = far_field(sol, "huygens", directions)
    # A cell is outside the sphere when it lies on the far side of the facet plane from the origin.
    function outside(facet)
        x = getcoordinates(grid, facet[1])
        a, b, c = (x[i] for i in FiniteElementMethod._TET_FACET_NODES[facet[2]])
        n = cross(b - a, c - a)
        return sign(dot(sum(x) / 4 - a, n)) == sign(dot(a, n))
    end
    for side in (true, false)
        one_side = Set(f for f in getfacetset(grid, "huygens") if outside(f) == side)
        grid.facetsets["one side"] = one_side
        @test length(one_side) < length(getfacetset(grid, "huygens"))
        ff = far_field(sol, "one side", directions)
        @test ff.E_theta ≈ reference.E_theta && ff.E_phi ≈ reference.E_phi
    end
    @test far_field(sol, "huygens", directions; center = (0.01, 0.0, 0.0)).E_theta ≈ reference.E_theta

    grid.facetsets["cap"] = Set(f for f in getfacetset(grid, "huygens") if getcoordinates(grid, f[1])[1][3] > 0)
    @test_throws "is not a closed surface" far_field(sol, "cap", directions)
    @test_throws "star-shaped" far_field(sol, "huygens", directions; center = (1.0, 0.0, 0.0))
    @test_throws "nonzero vector" far_field(sol, "huygens", [Vec(0.0, 0.0, 0.0)])
    @test_throws "method must be" radiated_power(sol, "huygens"; method = :poynting)
    @test_throws "no facet set" far_field(sol, "nowhere", directions)

    silent = MaxwellSolution(model, OMEGA, zeros(ComplexF64, ndofs(model.dh)))
    @test radiated_power(silent, "huygens") == 0
    @test_throws "radiates no power" directivity(silent, "huygens", directions)

    shielded = MaxwellModel(grid; absorbing = ["outer"], pec = ["huygens"])
    @test_throws "touches a PEC facet" far_field(MaxwellSolution(shielded, OMEGA, sol.u), "huygens", directions)
    dielectric = MaxwellModel(grid; absorbing = ["outer"], materials = Dict("domain" => MaxwellMaterial(; eps_r = 2.0)))
    @test_throws "must lie in vacuum" far_field(MaxwellSolution(dielectric, OMEGA, sol.u), "huygens", directions)
end
