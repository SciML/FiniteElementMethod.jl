# The PDESystem front end of the Maxwell model, against the operator-level MaxwellModel as oracle. Tests with a
# perfect-conductor condition need PDEBase's facet normal (SciML/PDEBase.jl#164) and are skipped without it.
using FiniteElementMethod, Ferrite, ModelingToolkit, DomainSets, LinearAlgebra, SparseArrays, Test
using LinearSolve: LinearSolve  # the default solver of the LinearProblem; its exports clash with Ferrite's
using SciMLBase: LinearProblem, discretize, solve, symbolic_discretize
using FiniteElementMethod: PDEBase
using Symbolics: expand_derivatives
include(joinpath(@__DIR__, "..", "shared", "maxwell.jl"))

const HAS_NORMAL = hasmethod(PDEBase.normal, Tuple{PDEBase.BoundaryPiece})
HAS_NORMAL || @warn "PDEBase has no facet normal (SciML/PDEBase.jl#164); the perfect-conductor tests are skipped"

@parameters x y z t
@variables E(..)[1:3]
const X = (x, y, z)
const Dt = Differential(t)
Ev = [E(t, x, y, z)[c] for c in 1:3]

# curl(ν curl E) + a₂ ∂ₜ²E + a₁ ∂ₜE + a₀ E = f on Ω, one equation per component.
function maxwell_system(Ω; nu = 1, a2 = 1, a1 = 0, a0 = 0, f = zeros(3), bcs = Equation[], ps = [], extra = c -> 0)
    cc = curl(curl(Ev, X) .* nu, X)
    eqs = [cc[c] + a2 * Dt(Dt(Ev[c])) + a1 * Dt(Ev[c]) + a0 * Ev[c] + extra(c) ~ f[c] for c in 1:3]
    return PDESystem(eqs, bcs, [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z)], ps; name = :maxwell)
end
edge_space() = Dict(E => Nedelec{RefTetrahedron, 1}())
maxwell_disc(Ω, ω; kw...) = FEMDiscretization(Ω, t; omega = ω, spaces = edge_space(), kw...)

pec_conditions(Ω, names = Symbol.(CUBE_FACES)) =
    reduce(vcat, [cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0 for Γ in boundarypiece.(Ref(Ω), names)])

relerr(a, b) = norm(a - b) / norm(b)
# The message of the ArgumentError that `f()` throws.
function message(f)
    try
        f()
    catch err
        err isa ArgumentError && return err.msg
        rethrow()
    end
    return "no error"
end

@testset "curl and the omega keyword" begin
    @test isequal(curl([y, z, x], X), [Differential(y)(x) - Differential(z)(z), Differential(z)(y) - Differential(x)(x), Differential(x)(z) - Differential(y)(y)])
    @test_throws ArgumentError("curl needs a 3-vector, got 2 components") curl([x, y], X)
    @test curl === Ferrite.curl  # one curl with both packages loaded
    Ω = MeshDomain(unit_cube(1))
    @test maxwell_disc(Ω, 2).omega === 2.0
    for bad in (0, -1.0, Inf, 1 + 2im)
        @test_throws ArgumentError("omega must be a positive, finite angular frequency in rad/s, got $(repr(bad))") maxwell_disc(Ω, bad)
    end
    @test_throws ArgumentError("omega needs the time variable of the time-domain PDESystem, as FEMDiscretization(Ω, t; omega)") FEMDiscretization(Ω; omega = 1.0, spaces = edge_space())
    @test_throws ArgumentError("a Nédélec space is discretized by the frequency-domain Maxwell path; give the angular frequency, as FEMDiscretization(Ω, t; omega, spaces)") FEMDiscretization(Ω, t; spaces = edge_space())
    @test_throws ArgumentError("`test_space` does not apply to the frequency-domain Maxwell path (omega); leave it at its default") maxwell_disc(Ω, 1; test_space = SUPG(0.1))
    @test_throws ArgumentError("`ibp` does not apply to the frequency-domain Maxwell path (omega); leave it at its default") maxwell_disc(Ω, 1; ibp = Dict(1 => PDEBase.StrongForm.Reject()))
end

@testset "two materials: discretize equals the operator-level model" begin
    grid = unit_cube(4)
    addcellset!(grid, "left", x -> x[1] <= 0.5)
    addcellset!(grid, "right", x -> x[1] >= 0.5)
    Ω = MeshDomain(grid)
    ω = 3.0
    source(p) = [sin(π * p[2]), p[1] * p[3], 1.0]
    left, right = MaxwellMaterial(; eps_r = 1.0), MaxwellMaterial(; eps_r = 4.0, mu_r = 2.0, sigma = 0.3)
    piecewise(a, b) = ifelse(x < 0.5, a, b)
    sys = maxwell_system(
        Ω; nu = piecewise(1.0, 0.5), a2 = piecewise(1.0, 4.0), a1 = piecewise(0.0, 0.3), f = source([x, y, z]),
        bcs = HAS_NORMAL ? pec_conditions(Ω) : Equation[]
    )
    prob = discretize(sys, maxwell_disc(Ω, ω))
    @test prob isa LinearProblem
    @test eltype(prob.A) == ComplexF64
    oracle = MaxwellModel(
        grid; materials = Dict("left" => left, "right" => right), pec = HAS_NORMAL ? CUBE_FACES : String[], eps0 = 1.0, mu0 = 1.0
    )
    # -iω J = f gives the oracle's load ∫ f ⋅ Nᵢ.
    expected = driven_problem(oracle, ω, p -> im .* source(p) ./ ω)
    @info "Maxwell front end oracle (two materials)" relerr(prob.A, expected.A) relerr(prob.b, expected.b) pec = HAS_NORMAL
    @test relerr(prob.A, expected.A) <= 1.0e-12
    @test relerr(prob.b, expected.b) <= 1.0e-12
    u = prob.A \ prob.b
    @test relerr(u, expected.A \ expected.b) <= 1.0e-10
    (; model, omega, load) = maxwell_model(sys, maxwell_disc(Ω, ω))
    @test omega == ω
    @test model.materials == oracle.materials
    @test model.pec == sort(oracle.pec)

    # Keywords reach the LinearProblem.
    @test discretize(sys, maxwell_disc(Ω, ω); alias_A = true).kwargs[:alias_A] == true
end

@testset "parameters, a reaction term and a lossy medium" begin
    grid = unit_cube(3)
    Ω = MeshDomain(grid)
    @parameters μ σ
    ω = 2.0
    sys = maxwell_system(Ω; nu = 1 / μ, a2 = 2.5, a1 = σ, a0 = 1.5, f = [1.0, y, 0.0], ps = [μ => 4.0, σ => 0.7])
    prob = discretize(sys, maxwell_disc(Ω, ω))
    # a₀ enters as -a₀/ω² in eps_r, so that -ω² eps_r = a₀ - ω² a₂.
    oracle = MaxwellModel(grid; materials = fill(MaxwellMaterial(; eps_r = 2.5 - 1.5 / ω^2, mu_r = 4.0, sigma = 0.7), getncells(grid)), eps0 = 1.0, mu0 = 1.0)
    expected = driven_problem(oracle, ω, p -> im .* [1.0, p[2], 0.0] ./ ω)
    @test relerr(prob.A, expected.A) <= 1.0e-12
    @test relerr(prob.b, expected.b) <= 1.0e-12

    @test_throws ArgumentError("load has 3 entries but the model has $(ndofs(oracle.dh)) dofs") driven_problem(oracle, ω; load = zeros(3))
    @test_throws ArgumentError("materials has 1 entries but the grid has $(getncells(grid)) cells; give one MaxwellMaterial per cell") MaxwellModel(
        grid; materials = [MaxwellMaterial()]
    )
end

@testset "PEC cavity, manufactured solution" begin
    if !HAS_NORMAL
        @test_skip "needs PDEBase.normal (SciML/PDEBase.jl#164)"
    else
        εr, μr, ω = 2.0, 2.0, 1.0
        Es = E_smooth([x, y, z])
        f = expand_derivatives.(curl(curl(Es, X) ./ μr, X)) .- ω^2 * εr .* Es
        errs = map((4, 8, 16)) do n
            grid = unit_cube(n)
            Ω = MeshDomain(grid)
            sys = maxwell_system(Ω; nu = 1 / μr, a2 = εr, f, bcs = pec_conditions(Ω))
            disc = maxwell_disc(Ω, ω)
            (; model, omega) = maxwell_model(sys, disc)
            sol = solve(discretize(sys, disc))
            msol = MaxwellSolution(model, omega, sol.u)
            p = Vec(0.4, 0.55, 0.6)
            point = norm(collect(only(field_values(msol, [p]).E)) - E_smooth(p))
            (; hcurl_errors(model, sol.u, E_smooth)..., point)
        end
        hs = 1 ./ (4, 8, 16)
        @info "Maxwell front end MMS (PEC cavity)" errs rates(getfield.(errs, :hcurl), hs)
        @test all(>=(0.9), rates(getfield.(errs, :hcurl), hs))
        @test all(>=(0.9), rates(getfield.(errs, :l2), hs))
        @test last(errs).point < first(errs).point / 2
    end
end

@testset "rejections of the volume equation" begin
    Ω = MeshDomain(unit_cube(1))
    D = Differential.(collect(X))
    form = "write the operator as curl(ν*curl(E)) with curl(F, (x, y, z)), as in curl(curl(E, X) / μ, X)"
    err(msg; kw...) = @test_throws ArgumentError(msg) discretize(maxwell_system(Ω; kw...), maxwell_disc(Ω, 1.0))

    laplacian(c) = -sum(D[d](D[d](Ev[c])) for d in 1:3)
    @test_throws ArgumentError("equation 1 has the vector Laplacian of E, which is component-wise H¹ and not H(curl)-conforming; $form") discretize(
        maxwell_system(Ω; nu = 0, extra = laplacian), maxwell_disc(Ω, 1.0)
    )
    err("the second-order part of equation 1 adds a component-wise Laplacian of E to curl(ν*curl(E)), which is not H(curl)-conforming; $form"; extra = laplacian)
    graddiv(c) = -D[c](sum(D[d](Ev[d]) for d in 1:3))
    err("equation 1 has a grad div E term, which has no meaning on Nédélec elements; $form"; extra = graddiv)
    # The expanded curl curl, ∂ᵢ∂ⱼEⱼ - ΔEᵢ, has the grad div flux and is rejected too.
    err("equation 1 has a grad div E term, which has no meaning on Nédélec elements; $form"; nu = 0, extra = c -> laplacian(c) - graddiv(c))
    err("the second-order part of equation 1 is anisotropic or couples the components of E differently from curl(ν*curl(E)); $form"; nu = [1.0, 2.0, 3.0])
    err("equation 1 has a first-order term in the derivatives of E, which curl(ν*curl(E)) does not have; $form"; extra = c -> D[1](Ev[c]))
    # ∂ₓ∂ₓ(x Eᶜ) - ∂ₓ(x ∂ₓEᶜ) leaves the flux Eᶜ, which is first order.
    err("equation 1 has a first-order term inside a derivative; $form"; extra = c -> D[1](D[1](x * Ev[c])) - D[1](x * D[1](Ev[c])))
    err("the zeroth-order term of equation 1 couples the components of E; the coefficients of E and its time derivatives must be scalars"; extra = c -> Ev[mod1(c + 1, 3)])
    anisotropic = "differs between the component equations at x = "
    @test occursin(anisotropic, message(() -> discretize(maxwell_system(Ω; extra = c -> c * Ev[c]), maxwell_disc(Ω, 1.0))))
    err("the Maxwell front end discretizes linear equations only, but equation 1 is nonlinear in E"; extra = c -> Ev[c]^3)
    err("the Maxwell front end discretizes linear equations only, but equation 1 is nonlinear in E"; nu = 1 + Ev[1]^2)
    err("the coefficient of E is nonlinear in its time derivatives"; a2 = 0, extra = c -> Dt(Ev[c]) * Dt(Ev[c]) / Ev[c])
    err("a source of the Maxwell equation depends on time; with omega the coefficients and sources are time-independent, and a source is the phasor amplitude"; f = [sin(t), 0, 0])
    err("a coefficient of the Maxwell equation depends on time; with omega the coefficients and sources are time-independent, and a source is the phasor amplitude"; a2 = 1 + t)
    err("ν = 0.0 in cell 1; ν = 1/μ must be nonzero"; nu = 0)
    @test_throws ArgumentError discretize(maxwell_system(Ω; extra = c -> Dt(Dt(Dt(Ev[c])))), maxwell_disc(Ω, 1.0))

    # A second derivative of a field-free expression lands in the flux; it must not vanish from the load.
    @test_throws ArgumentError("equation 1 has a derivative of an expression without E, a source inside a divergence, which is not integrated by parts; write that source term out explicitly as a function of the coordinates") discretize(
        maxwell_system(Ω; extra = c -> 3 * D[1](D[1](x^2))), maxwell_disc(Ω, 2.0)
    )
    @test discretize(maxwell_system(Ω; f = [6.0, 6.0, 6.0]), maxwell_disc(Ω, 2.0)) isa LinearProblem
    @test occursin(
        "is inside a spatial derivative; time derivatives may only multiply E",
        message(() -> discretize(maxwell_system(Ω; extra = c -> D[1](D[1](Dt(Ev[c])))), maxwell_disc(Ω, 2.0)))
    )

    # Terms confined to a small ball around one interior centroid are seen: every centroid is checked.
    grid = unit_cube(3)
    Ωf = MeshDomain(grid)
    cell = getncells(grid) ÷ 2 + 1
    xc = sum(getcoordinates(grid, cell)) / 4
    h = ifelse((x - xc[1])^2 + (y - xc[2])^2 + (z - xc[3])^2 < 1.0e-4, 1.0, 0.0)
    local_term(extra) = message(() -> discretize(maxwell_system(Ωf; extra), maxwell_disc(Ωf, 1.0)))
    @test local_term(c -> c == 2 ? h * Ev[2] : 0) ==
        "the coefficient of E differs between the component equations at x = $(Tuple(xc)), so the medium is anisotropic; the coefficients must be scalars"
    @test local_term(c -> h * D[1](Ev[c])) ==
        "equation 1 has a first-order term in the derivatives of E, which curl(ν*curl(E)) does not have; $form"
    @test local_term(c -> c == 1 ? h * Ev[2] : 0) ==
        "the zeroth-order term of equation 1 couples the components of E; the coefficients of E and its time derivatives must be scalars"
end

@testset "rejections of the unknowns, spaces and mesh" begin
    Ω = MeshDomain(unit_cube(1))
    sys = maxwell_system(Ω)
    @test_throws ArgumentError("omega needs one vector unknown with a Nédélec space and no other spaces, as spaces = Dict(E => Nedelec{RefTetrahedron, 1}())") discretize(
        sys, FEMDiscretization(Ω, t; omega = 1.0)
    )
    @test_throws ArgumentError("omega needs one vector unknown with a Nédélec space and no other spaces, as spaces = Dict(E => Nedelec{RefTetrahedron, 1}())") discretize(
        sys, FEMDiscretization(Ω, t; omega = 1.0, spaces = Dict(E => Lagrange{RefTetrahedron, 1}()^3))
    )

    @variables u(..) q(..)
    lap(w) = sum(Differential(s)(Differential(s)(w)) for s in X)
    scalar = PDESystem([Dt(u(t, x, y, z)) ~ lap(u(t, x, y, z))], Equation[], [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [u(t, x, y, z)], []; name = :scalar)
    @test_throws ArgumentError("omega discretizes one 3-component vector unknown with a Nédélec space, as @variables E(..)[1:3]; u(t, x, y, z) is not a component of it") discretize(
        scalar, FEMDiscretization(Ω, t; omega = 1.0, spaces = Dict(u => Nedelec{RefTetrahedron, 1}()))
    )
    @variables w(..)::Complex
    complex_sys = PDESystem([Dt(w(t, x, y, z)) ~ lap(w(t, x, y, z))], Equation[], [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [w(t, x, y, z)], []; name = :complex_sys)
    @test_throws ArgumentError("the unknown must be real: with omega the PDESystem is the real time-domain equation and the phasor is complex") discretize(
        complex_sys, FEMDiscretization(Ω, t; omega = 1.0, spaces = Dict(w => Nedelec{RefTetrahedron, 1}()))
    )
    cc = curl(curl(Ev, X), X)
    mixed = PDESystem(
        [[cc[c] + Dt(Dt(Ev[c])) ~ q(t, x, y, z) for c in 1:3]; Dt(q(t, x, y, z)) ~ lap(q(t, x, y, z))], Equation[],
        [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z), q(t, x, y, z)], []; name = :mixed
    )
    @test_throws ArgumentError("omega discretizes one 3-component vector unknown with a Nédélec space, as @variables E(..)[1:3]; q(t, x, y, z) is not a component of it") discretize(
        mixed, maxwell_disc(Ω, 1.0)
    )
    two = PDESystem(cc[1:2] .+ Dt.(Dt.(Ev[1:2])) .~ 0, Equation[], [t ∈ Interval(0.0, 1.0), (x, y, z) ∈ Ω], [t, x, y, z], [E(t, x, y, z)], []; name = :two)
    @test_throws ArgumentError("the system has 2 equations for the 3 components of the vector unknown") discretize(two, maxwell_disc(Ω, 1.0))

    for ip in (Nedelec{RefTriangle, 2}(), Nedelec{RefTriangle, 1}())
        @test_throws ArgumentError("omega supports the space Nedelec{RefTetrahedron, 1}() only, got $ip") discretize(
            sys, FEMDiscretization(Ω, t; omega = 1.0, spaces = Dict(E => ip))
        )
    end

    hexes = MeshDomain(generate_grid(Hexahedron, (1, 1, 1)))
    @test_throws ArgumentError("omega needs a mesh of Tetrahedron cells, got Hexahedron") discretize(
        maxwell_system(hexes), FEMDiscretization(hexes, t; omega = 1.0, spaces = Dict(E => Nedelec{RefHexahedron, 1}()))
    )

    disc = maxwell_disc(Ω, 1.0)
    for (f, what) in ((FiniteElementMethod._daeproblem, "DAEProblem"), (FiniteElementMethod._nonlinearproblem, "NonlinearProblem"), (symbolic_discretize, "symbolic_discretize"), (FiniteElementMethod.operators, "operators"))
        @test_throws ArgumentError("$what has no frequency-domain form; a FEMDiscretization with omega lowers to a LinearProblem through discretize(pdesys, disc)") f(sys, disc)
    end
    @test_throws ArgumentError("maxwell_model needs a FEMDiscretization with omega, as FEMDiscretization(Ω, t; omega = 2π, spaces)") maxwell_model(sys, FEMDiscretization(Ω, t))

    initial = maxwell_system(Ω; bcs = [E(0, x, y, z)[c] ~ 0 for c in 1:3])
    @test_throws ArgumentError("with omega every condition must be a boundary condition on one piece, and initial conditions are not used; got $(first(PDEBase.make_pdesys_compatible(initial)[1].bcs))") discretize(initial, disc)
end

@testset "rejections of the boundary conditions" begin
    if !HAS_NORMAL
        @test_skip "needs PDEBase.normal (SciML/PDEBase.jl#164)"
    else
        Ω = MeshDomain(unit_cube(1))
        Γ = boundarypiece(Ω, :left)
        n = facet_normal(Γ)
        Etr = [E(t, Γ)[c] for c in 1:3]
        pform = "write the perfect conductor as cross(facet_normal(Γ), [E(t, Γ)[c] for c in 1:3]) .~ 0"
        bad(bcs) = discretize(maxwell_system(Ω; bcs), maxwell_disc(Ω, 1.0))
        @test bad(cross(Etr, n) .~ 0) isa LinearProblem
        # The facets of `left` have the normal -e₁, on which E₂ ~ 0 is tangential but E₁ ~ 0 is not.
        @test occursin(r"^the conditions on left at facet \(\d+, \d+\) with normal \(-1\.0, -?0\.0, -?0\.0\) constrain the normal component of E", message(() -> bad([Etr[1] ~ 0])))
        @test occursin(r"^the conditions on left at facet .* constrain 1 of the two tangential components of E", message(() -> bad([cross(n, Etr)[2] ~ 0])))
        # (n × E)₁ = n₂E₃ - n₃E₂ vanishes identically where n = -e₁.
        @test occursin(r"constrain 0 of the two tangential components", message(() -> bad([cross(n, Etr)[1] ~ 0])))
        @test occursin(r"constrain 1 of the two tangential components", message(() -> bad([Etr[2] ~ 0])))
        # Codex r3: a factor that vanishes on the actual normal of the piece is no conductor there.
        Γt = boundarypiece(Ω, :top)
        top(bcs) = discretize(maxwell_system(Ω; bcs), maxwell_disc(Ω, 1.0))
        @test occursin(
            r"^the conditions on top at facet .* constrain 0 of the two tangential components of E",
            message(() -> top(facet_normal(Γt)[1] .* cross(facet_normal(Γt), [E(t, Γt)[c] for c in 1:3]) .~ 0))
        )
        @test top(facet_normal(Γt)[3] .* cross(facet_normal(Γt), [E(t, Γt)[c] for c in 1:3]) .~ 0) isa LinearProblem
        @test_throws ArgumentError("the conditions on left prescribe a nonzero tangential field, which is not supported; $pform") bad(cross(n, Etr) .~ [1.0, 0, 0])
        @test_throws ArgumentError("the conditions on left are nonlinear in E; $pform") bad(cross(n, Etr .^ 2) .~ 0)
        @test_throws ArgumentError("the conditions on left prescribe a nonzero tangential field, which is not supported; $pform") bad(cross(n, Etr) .~ cross(n, [1.0, 0, 0]))
        @test_throws ArgumentError("the conditions on left prescribe a nonzero tangential field, which is not supported; $pform") bad(cross(n, Etr) .+ [n[2] * n[3], 0, 0] .~ 0)
        @test_throws ArgumentError bad([NormalDerivative(Γ)(Etr[1]) ~ 0])
        other = MeshDomain(unit_cube(1))
        @test_throws ArgumentError bad(cross(facet_normal(boundarypiece(other, :left)), [E(t, boundarypiece(other, :left))[c] for c in 1:3]) .~ 0)
    end
end
