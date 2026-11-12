# The PDESystem front end with the first-order absorbing boundary, wave ports and frequency sweeps, against the
# operator-level MaxwellModel as oracle. A WR-90 guide in millimetres with ε0 = μ0 = 1, so ω is k0 in rad/mm.
using FiniteElementMethod, Ferrite, ModelingToolkit, DomainSets, LinearAlgebra, Test
using SciMLBase: LinearProblem, discretize
using FiniteElementMethod: PDEBase

@parameters x y z t
@variables E(..)[1:3]
const X = (x, y, z)
const Dt = Differential(t)
Ev = [E(t, x, y, z)[c] for c in 1:3]

const WR90_A, WR90_B = 22.86, 10.16
const K0 = 2π * 10.0e9 / FiniteElementMethod.C0 * 1.0e-3  # rad/mm at 10 GHz
const WALLS = ["left", "right", "front", "back"]

guide(len, h) = generate_grid(Tetrahedron, ceil.(Int, (WR90_A, WR90_B, len) ./ h), Vec(0.0, 0.0, 0.0), Vec(WR90_A, WR90_B, float(len)))
trace(Γ) = [E(t, Γ)[c] for c in 1:3]
pec(Ω, names) = reduce(vcat, [cross(facet_normal(Γ), trace(Γ)) .~ 0 for Γ in boundarypiece.(Ref(Ω), Symbol.(names))])
# n × (ν curl E) = -sqrt(ε/μ) n × (n × ∂ₜE), the matched first-order absorbing condition.
function abc(Γ; nu = 1, eps = 1)
    n, Eb = facet_normal(Γ), trace(Γ)
    return cross(n, nu .* curl(Eb, X)) .~ -sqrt(eps * nu) .* cross(n, cross(n, Dt.(Eb)))
end
function system(Ω, bcs; nu = 1, a2 = 1, a1 = 0, a0 = 0, f = zeros(3))
    cc = curl(nu .* curl(Ev, X), X)
    eqs = [cc[c] + a2 * Dt(Dt(Ev[c])) + a1 * Dt(Ev[c]) + a0 * Ev[c] ~ f[c] for c in 1:3]
    return PDESystem(eqs, bcs, [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z)]; name = :guide)
end
port(name) = WavePort(name; a_direction = (1, 0, 0))
edge_disc(Ω, ports = WavePort[]; omega = K0) = FEMDiscretization(Ω, t; omega, spaces = Dict(E => Nedelec{RefTetrahedron, 1}()), ports)

function message(f)
    try
        f()
    catch err
        err isa ArgumentError && return err.msg
        rethrow()
    end
    return "no error"
end

@testset "ports keyword" begin
    Ω = MeshDomain(guide(10.0, 5.0))
    @test edge_disc(Ω, [port("bottom")]).ports == [port("bottom")]
    @test WavePort(boundarypiece(Ω, :bottom); a_direction = (1, 0, 0)) == port("bottom")
    @test edge_disc(Ω, Iterators.Stateful([port("bottom")])).ports == [port("bottom")]
    @test_throws ArgumentError("ports belong to the frequency-domain Maxwell path; give the angular frequency omega") FEMDiscretization(Ω, t; ports = [port("bottom")])
    @test_throws ArgumentError("ports must be WavePorts, as ports = [WavePort(Γ; a_direction = (1, 0, 0))]; got [\"bottom\"]") edge_disc(Ω, ["bottom"])
    @test_throws ArgumentError("solve_sweep needs wave ports; give them as FEMDiscretization(Ω, t; omega, spaces, ports = [WavePort(...)])") solve_sweep(system(Ω, Equation[]), edge_disc(Ω), [K0])
end

@testset "absorbing end and a wave port: S11 equals the operator-level model" begin
    grid = guide(30.0, 2.0)
    Ω = MeshDomain(grid)
    sys = system(Ω, [pec(Ω, WALLS); abc(boundarypiece(Ω, :top))])
    disc = edge_disc(Ω, [WavePort(boundarypiece(Ω, :bottom); a_direction = (1, 0, 0))])
    (; model) = maxwell_model(sys, disc)
    @test model.pec == sort(WALLS) && model.absorbing == ["top"]
    oracle = MaxwellModel(grid; pec = WALLS, absorbing = ["top"], ports = [port("bottom")], eps0 = 1.0, mu0 = 1.0)
    S = solve_sweep(sys, disc, [K0]).S
    @test S == solve_sweep(oracle, [K0]).S
    β = sqrt(K0^2 - (π / WR90_A)^2)
    Γ = (β - K0) / (β + K0)
    @info "Maxwell front end: WR-90 with an absorbing end" S11 = S[1] abs(S[1]) Γ
    @test Γ ≈ -0.1396 atol = 1.0e-4
    @test abs(S[1]) ≈ abs(Γ) rtol = 0.03
    # discretize keeps the port as a matched termination, with the load of the (zero) source.
    prob = discretize(sys, disc)
    expected = driven_problem(oracle, K0; load = zeros(ndofs(oracle.dh)))
    @test prob.A == expected.A && prob.b == expected.b
end

@testset "two-port dielectric guide: the sweep equals the operator-level sweep" begin
    grid = guide(20.0, 2.5)
    Ω = MeshDomain(grid)
    sys = system(Ω, pec(Ω, WALLS); nu = 0.5, a2 = 2.0)
    ωs = K0 .* [0.8, 1.0, 1.2]
    sweep = solve_sweep(sys, edge_disc(Ω, port.(["bottom", "top"])), ωs)
    material = MaxwellMaterial(; eps_r = 2.0, mu_r = 2.0)
    oracle = MaxwellModel(grid; materials = fill(material, getncells(grid)), pec = WALLS, ports = port.(["bottom", "top"]), eps0 = 1.0, mu0 = 1.0)
    expected = solve_sweep(oracle, ωs)
    @test size(sweep.S) == (2, 2, 3)
    @test sweep.S == expected.S && sweep.omegas == expected.omegas
    # The model does not depend on the omega of the discretization.
    @test solve_sweep(sys, edge_disc(Ω, port.(["bottom", "top"]); omega = 0.3), ωs).S == expected.S
end

@testset "equivalent forms of the absorbing condition" begin
    grid = guide(10.0, 5.0)
    Ω = MeshDomain(grid)
    Γ = boundarypiece(Ω, :top)
    n, Eb = facet_normal(Γ), trace(Γ)
    oracle = MaxwellModel(
        grid; materials = fill(MaxwellMaterial(; eps_r = 3.0, mu_r = 2.0), getncells(grid)), pec = WALLS, absorbing = ["top"], eps0 = 1.0, mu0 = 1.0
    )
    expected = driven_problem(oracle, K0; load = zeros(ndofs(oracle.dh))).A
    forms = [
        abc(Γ; nu = 0.5, eps = 3.0),
        # multiplied by μ = 2
        cross(n, curl(Eb, X)) .~ -sqrt(6.0) .* cross(n, cross(n, Dt.(Eb))),
        # the other cross order on both sides
        cross(curl(Eb, X) ./ 2, n) .~ -sqrt(1.5) .* cross(cross(n, Dt.(Eb)), n),
    ]
    for bc in forms
        prob = discretize(system(Ω, [pec(Ω, WALLS); bc]; nu = 0.5, a2 = 3.0), edge_disc(Ω))
        @test prob.A == expected
    end
end

@testset "the absorbing condition is checked at SI scale too" begin
    # ν = 1/μ0 and ε = ε0 make the coefficient of ∂ₜE about 1e-9 of that of curl E; the sign must still be checked.
    mu0 = FiniteElementMethod.MU0
    eps0 = 1 / (mu0 * FiniteElementMethod.C0^2)
    ω = 1.0e9
    grid = generate_grid(Tetrahedron, (2, 2, 2), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    Ω = MeshDomain(grid)
    Γ = boundarypiece(Ω, :top)
    n, Eb = facet_normal(Γ), trace(Γ)
    function si(sign)
        bcs = cross(n, (1 / mu0) .* curl(Eb, X)) .~ sign * sqrt(eps0 / mu0) .* cross(n, cross(n, Dt.(Eb)))
        return discretize(system(Ω, bcs; nu = 1 / mu0, a2 = eps0), edge_disc(Ω; omega = ω))
    end
    @test si(-1) isa LinearProblem
    @test occursin("are not n × (ν curl E) = -sqrt(ε/μ) n × (n × ∂ₜE) of the adjacent cell", message(() -> si(1)))
end

@testset "a constraint added to the absorbing condition is seen at a large frequency" begin
    Ω = MeshDomain(guide(10.0, 5.0))
    Γ = boundarypiece(Ω, :top)
    Eb = trace(Γ)
    bcs = [pec(Ω, WALLS); abc(Γ); Eb[3] ~ 0]
    @test occursin("independent equations", message(() -> discretize(system(Ω, bcs), edge_disc(Ω; omega = 1.0e12))))
    @test occursin("independent equations", message(() -> discretize(system(Ω, bcs), edge_disc(Ω; omega = 1.0))))
    # A term in E next to the time derivative at a frequency where it is tiny in field units is still a term.
    withE = cross(facet_normal(Γ), curl(Eb, X)) .~ -cross(facet_normal(Γ), cross(facet_normal(Γ), Dt.(Eb) .+ Eb))
    @test occursin("are not n × (ν curl E)", message(() -> discretize(system(Ω, [pec(Ω, WALLS); withE]), edge_disc(Ω; omega = 1.0e12))))
    # Scaling an equation changes nothing: a tiny extra constraint is still one, and a tiny valid row is still valid.
    tiny = [pec(Ω, WALLS); abc(Γ); 1.0e-13 .* [Eb[3]] .~ 0]
    @test occursin("independent equations", message(() -> discretize(system(Ω, tiny), edge_disc(Ω; omega = 1.0))))
    scaled = abc(Γ)
    scaled[1] = scaled[1].lhs * 1.0e-13 ~ scaled[1].rhs * 1.0e-13
    @test discretize(system(Ω, [pec(Ω, WALLS); scaled]), edge_disc(Ω; omega = 1.0)) isa LinearProblem
end

@testset "rejections of absorbing conditions and ports" begin
    grid = guide(10.0, 5.0)
    addfacetset!(grid, "lid", x -> x[3] ≈ 10.0)  # the facets of top again
    Ω = MeshDomain(grid)
    Γ = boundarypiece(Ω, :top)
    n, Eb = facet_normal(Γ), trace(Γ)
    walls = pec(Ω, WALLS)
    bad(bcs; ports = WavePort[], kw...) = discretize(system(Ω, [walls; bcs]; kw...), edge_disc(Ω, ports))
    @test bad(abc(Γ)) isa LinearProblem
    mismatch = r"^the conditions on top at facet \(\d+, \d+\) with normal \(-?0\.0, -?0\.0, 1\.0\) are not n × \(ν curl E\) = -sqrt\(ε/μ\) n × \(n × ∂ₜE\) of the adjacent cell \(ν = 1\.0, sqrt\(ε/μ\) = 1\.0\)"
    @test occursin(mismatch, message(() -> bad(abc(Γ; eps = 2.0))))                                # impedance
    @test occursin(mismatch, message(() -> bad(cross(n, curl(Eb, X)) .~ cross(n, cross(n, Dt.(Eb))))))  # sign
    @test occursin(mismatch, message(() -> bad(cross(n, curl(Eb, X) ./ 2) .~ -cross(n, cross(n, Dt.(Eb))))))  # ν
    # An impedance condition in E instead of ∂ₜE, and the absorbing condition plus a term in E.
    @test occursin(mismatch, message(() -> bad(cross(n, curl(Eb, X)) .~ -cross(n, cross(n, Eb)))))
    @test occursin(mismatch, message(() -> bad(cross(n, curl(Eb, X)) .~ -cross(n, cross(n, Dt.(Eb) .+ Eb)))))
    @test occursin(r"^the conditions on top at facet .* have 1 independent equations; the absorbing condition", message(() -> bad([abc(Γ)[1]])))
    @test occursin(r"^the conditions on top at facet .* have 4 independent equations", message(() -> bad([abc(Γ); cross(n, Eb) .~ 0])))
    # The third row vanishes at the actual normal e₃ of top, so two rows are the whole condition there, and a
    # term added to that row is a third equation.
    @test bad(abc(Γ)[1:2]) isa LinearProblem
    @test occursin(r"^the conditions on top at facet .* have 3 independent equations", message(() -> bad([abc(Γ)[1:2]; abc(Γ)[3].lhs + Eb[3] ~ abc(Γ)[3].rhs])))
    @test occursin("has the derivative", message(() -> bad([abc(Γ)[1:2]; Dt(Dt(Eb[1])) ~ 0])))
    @test occursin("has the derivative", message(() -> bad([abc(Γ)[1:2]; Differential(x)(Differential(y)(Eb[1])) ~ 0])))
    @test occursin("has the derivative", message(() -> bad([abc(Γ)[1:2]; Differential(x)(2 * Eb[1]) ~ 0])))
    @test occursin("has a normal derivative", message(() -> bad([abc(Γ)[1:2]; NormalDerivative(Γ)(Eb[1]) ~ 0])))
    @test occursin(r"depends on t;", message(() -> bad(cross(n, curl(Eb, X)) .~ -(1 + t) .* cross(n, cross(n, Dt.(Eb))))))
    @test occursin(r"^the absorbing condition on top has a source", message(() -> bad(cross(n, curl(Eb, X)) .~ -cross(n, cross(n, Dt.(Eb))) .+ [1.0, 0, 0])))
    @test occursin(r"^the absorbing condition on top at facet .* borders cell \d+ with ν = 1\.0, a₂ = 1\.0, a₁ = 0\.1 and a₀ = 0\.0", message(() -> bad(abc(Γ); a1 = 0.1)))
    @test occursin(r"borders cell \d+ with ν = 1\.0, a₂ = 1\.0, a₁ = 0\.0 and a₀ = 0\.2", message(() -> bad(abc(Γ); a0 = 0.2)))
    @test occursin(r"borders cell \d+ with ν = 1\.0, a₂ = -1\.0", message(() -> bad(abc(Γ); a2 = -1.0)))
    # Coefficients of the condition are constants on the piece, checked whole at every facet.
    @test occursin(r"depends on x;", message(() -> bad(cross(n, curl(Eb, X)) .~ -(1 + 0.1x) .* cross(n, cross(n, Dt.(Eb))))))
    # A second piece on the same facets with the other condition.
    overlap = "the pieces lid and top share facets and one of them is absorbing; give each facet one condition"
    @test_throws ArgumentError(overlap) bad([abc(Γ); pec(Ω, ["lid"])])
    # The matched condition on two pieces with the same facets would double the admittance.
    @test_throws ArgumentError(overlap) bad([abc(Γ); abc(boundarypiece(Ω, :lid))])
    # Two perfect-conductor pieces on the same facets constrain the same dofs and are fine.
    @test bad(pec(Ω, ["top", "lid"])) isa LinearProblem
    # The operator-level model refuses the same overlaps among absorbing sets and ports.
    twice = "the facet sets \"top\" and \"lid\" of `absorbing` and `ports` share facets, whose boundary terms would be added twice; give each facet one absorbing condition or port"
    @test_throws ArgumentError(twice) MaxwellModel(grid; pec = WALLS, absorbing = ["top", "lid"])
    @test_throws ArgumentError(replace(twice, "\"lid\"" => "\"top\"")) MaxwellModel(grid; pec = WALLS, absorbing = ["top", "top"])
    @test_throws ArgumentError(twice) MaxwellModel(grid; pec = WALLS, absorbing = ["top"], ports = [port("lid")])
    @test_throws ArgumentError(replace(twice, "\"top\"" => "\"bottom\"", "\"lid\"" => "\"bottom\"")) MaxwellModel(grid; pec = WALLS, ports = [port("bottom"), port("bottom")])

    # Ports carry their own condition and need a frequency-independent model.
    portmsg(name) = "the wave port on bottom is the boundary condition of its facets, but the PDESystem also has a condition on $name, which shares facets with it; remove that condition"
    @test_throws ArgumentError(portmsg("bottom")) bad(pec(Ω, ["bottom"]); ports = [port("bottom")])
    @test_throws ArgumentError(portmsg("bottom")) bad(abc(boundarypiece(Ω, :bottom)); ports = [port("bottom")])
    @test bad(Equation[]; ports = [port("bottom")]) isa LinearProblem
    @test occursin(r"^a₀ = 0\.2 in cell 1, but with wave ports the coefficient of E must be zero", message(() -> bad(Equation[]; ports = [port("bottom")], a0 = 0.2)))
    @test_throws ArgumentError("solve_sweep drives the wave ports only, but the PDESystem has a source; solve a source with discretize(pdesys, disc) at each omega") solve_sweep(
        system(Ω, walls; f = [0, 1.0, 0]), edge_disc(Ω, [port("bottom")]), [K0]
    )
end
