# The PDESystem front end of the Maxwell model, against the operator-level MaxwellModel as oracle. Tests with a
# perfect-conductor condition use `facet_normal`.
using FiniteElementMethod, Ferrite, ModelingToolkit, DomainSets, LinearAlgebra, SparseArrays, Test
using LinearSolve: LinearSolve  # the default solver of the LinearProblem; its exports clash with Ferrite's
using SciMLBase: LinearProblem, discretize, solve, symbolic_discretize
using FiniteElementMethod: PDEBase
using Symbolics: expand_derivatives
include(joinpath(@__DIR__, "..", "shared", "maxwell.jl"))

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
    for bad in (0, -1.0, Inf, 1 + 2im, big"1e-400", big"1e400", 1.0e-200, 1.0e160)
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
        bcs = pec_conditions(Ω)
    )
    prob = discretize(sys, maxwell_disc(Ω, ω))
    @test prob isa LinearProblem
    @test eltype(prob.A) == ComplexF64
    oracle = MaxwellModel(
        grid; materials = Dict("left" => left, "right" => right), pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0
    )
    # -iω J = f gives the oracle's load ∫ f ⋅ Nᵢ.
    expected = driven_problem(oracle, ω, p -> im .* source(p) ./ ω)
    @info "Maxwell front end oracle (two materials)" relerr(prob.A, expected.A) relerr(prob.b, expected.b)
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
    @test occursin("supported window", message(() -> discretize(maxwell_system(Ω; nu = 1.0e-309, a2 = 1.0e-309), maxwell_disc(Ω, 1.0))))  # subnormal
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

    # A coupling is judged against the zeroth-order coefficients, not against a much larger ν: ν = 1e12 sits
    # beside a coupling of 1e-3 in a₀ = 1.
    err("the zeroth-order term of equation 1 couples the components of E; the coefficients of E and its time derivatives must be scalars"; nu = 1.0e12, a0 = 1.0, extra = c -> 1.0e-3 * Ev[mod1(c + 1, 3)])

    # The same for a first-order term next to a large ν.
    err("equation 1 has a first-order term in the derivatives of E, which curl(ν*curl(E)) does not have; $form"; nu = 1.0e12, extra = c -> D[1](Ev[c]))
    err("equation 1 has a first-order term inside a derivative; $form"; nu = 1.0e12, extra = c -> D[1](D[1](x * Ev[c])) - D[1](x * D[1](Ev[c])))

    # Nor does a large reaction coefficient hide a component-wise Laplacian beside curl(ν*curl(E)).
    err("the second-order part of equation 1 adds a component-wise Laplacian of E to curl(ν*curl(E)), which is not H(curl)-conforming; $form"; a0 = 1.0e12, a2 = 1.0e12, extra = laplacian)

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

@testset "coefficients outside the supported window are rejected, not turned into NaN or silent underflow" begin
    Ω = MeshDomain(unit_cube(1))
    outside(ω; kw...) = occursin("supported window", message(() -> discretize(maxwell_system(Ω; kw...), maxwell_disc(Ω, ω))))
    @test outside(1.0e-160; a0 = 1.0)
    @test outside(1.0e154; a0 = 5.0, a2 = 5.0e-308)  # the net reaction 8.4e-17 against ε = -net/ω² underflowing
    @test outside(2.0e-162; nu = 1.0e-30, a2 = 1.0e308, a0 = 0.0, f = [-4.0e-16, 0.0, 0.0])  # ω² subnormal
    @test outside(2.0^-511; nu = 1.0e-307, a0 = 2.0^-1074, f = [2.0^-1074, 0.0, 0.0])  # assembled reaction and load underflow
    @test outside(1.0e154; nu = 1.0e-30, a2 = 0, a0 = 1.0e-20)
    @test outside(1.0e154; nu = 1.0e-180, a0 = 0, a2 = 0, a1 = 2.0^-1074)  # the conductivity block underflows in assembly
    @test outside(1.0; nu = 1.0e101)
    @test outside(1.0; f = [1.0e-101, 0.0, 0.0])
    # SI-scale physics is well inside the window: ε₀, 1/μ₀, ω = 2π 10 GHz and the conductivity of copper.
    ε0, ν, σ, ω = 8.854187817e-12, 1 / (4π * 1.0e-7), 5.8e7, 2π * 1.0e10
    sys = maxwell_system(Ω; nu = ν, a2 = ε0, a1 = σ, f = [1.0, 0.0, 0.0])
    disc = maxwell_disc(Ω, ω)
    (; model) = maxwell_model(sys, disc)
    prob = discretize(sys, disc)
    E = field_values(MaxwellSolution(model, ω, prob.A \ prob.b), [Vec(0.3, 0.4, 0.5)]).E[1]
    reaction = -ω^2 * ε0 + im * ω * σ
    @test collect(E) ≈ [1 / reaction, 0, 0] rtol = 1.0e-6
end

@testset "a large reaction coefficient hides no other coefficient" begin
    Ω = MeshDomain(unit_cube(2))
    D = Differential.(collect(X))
    rejected(ω; kw...) = message(() -> discretize(maxwell_system(Ω; kw...), maxwell_disc(Ω, ω))) != "no error"
    # a0 - ω² a2 = 0: a first-order term must not vanish into the cancelling reaction and inertia.
    @test rejected(2.0; a0 = 4.0e16, a2 = 1.0e16, extra = c -> D[1](Ev[c]))
    # Couplings and component-dependent reactions next to a large a0 - ω² a2.
    @test rejected(2.0; a0 = 4.0e12, a2 = 1.0e12, extra = c -> Ev[mod1(c + 1, 3)])
    @test rejected(2.0; a0 = 4.0e12, a2 = 1.0e12, extra = c -> Dt(Ev[mod1(c + 1, 3)]))
    @test rejected(2.0; a0 = 4.0e12, a2 = 1.0e12, extra = c -> c * Ev[c])
    # Unequal inertia next to a large reaction.
    @test rejected(2.0e6; a0 = 1.0e12, a2 = 1.0, extra = c -> (c - 1) * Dt(Dt(Ev[c])))
    @test rejected(2.0e6; a0 = 1.0e12, a2 = 1.0, extra = c -> (c - 1) * Dt(Ev[c]))
    # Scaling the whole equation changes nothing: a small coupling next to small coefficients is still a coupling.
    @test rejected(1.0; a2 = 1.0e-16, nu = 1.0e-16, extra = c -> 1.0e-16 * Ev[mod1(c + 1, 3)])
    @test rejected(1.0e8; a2 = 1.0e-16, extra = c -> 1.0e-16 * Dt(Dt(Ev[mod1(c + 1, 3)])))
    # Component reactions 2^48 + c against the inertia 2^46 cancel at ω = 2 to the anisotropic diag(1, 2, 3); the
    # differences 1 and 2 are far below a tolerance scaled to 2^48, and must still be seen, exactly.
    anisotropic = message(() -> discretize(maxwell_system(Ω; a2 = 2.0^46, extra = c -> (2.0^48 + c) * Ev[c]), maxwell_disc(Ω, 2.0)))
    @test occursin("so the medium is anisotropic", anisotropic)
    # The same large coefficients with scalar lower-order terms are fine.
    @test !rejected(2.0e6; a0 = 1.0e12, a2 = 2.0, a1 = 3.0)
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
    # A factor that vanishes on the actual normal of the piece is no conductor there.
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

@testset "the net reaction is formed exactly" begin
    # a₀ - ω² a₂ = 1 exactly with ω = 3, a₂ = 2^49 and a₀ = 9 a₂ + 1: the constant field (1, 0, 0) solves the equation.
    Ω = MeshDomain(unit_cube(2))
    ω = 3.0
    sys = maxwell_system(Ω; nu = 1.0, a2 = 2.0^49, a0 = 9 * 2.0^49 + 1, f = [1.0, 0.0, 0.0])
    disc = maxwell_disc(Ω, ω)
    (; model) = maxwell_model(sys, disc)
    u = discretize(sys, disc).A \ discretize(sys, disc).b
    E = field_values(MaxwellSolution(model, ω, u), [Vec(0.31, 0.37, 0.41)]).E[1]
    @test collect(E) ≈ [1.0, 0.0, 0.0] atol = 1.0e-10
end

@testset "the net reaction does not depend on the BigFloat precision" begin
    # a₀ - ω² a₂ = -1 exactly for ω = 1 + 2^-30, a₂ = 2^60 and a₀ = 2^60 + 2^31, so ε = 1/ω².
    Ω = MeshDomain(unit_cube(1))
    ω = 1 + 2.0^-30
    sys = maxwell_system(Ω; nu = 1.0, a2 = 2.0^60, a0 = 2.0^60 + 2.0^31)
    eps_r() = maxwell_model(sys, maxwell_disc(Ω, ω)).model.materials[1].eps_r
    @test eps_r() ≈ 1 / ω^2 rtol = 1.0e-12
    setprecision(BigFloat, 53) do
        @test eps_r() ≈ 1 / ω^2 rtol = 1.0e-12
    end
end

@testset "a source with no quadrature point in its support gives a zero load, not an underflow error" begin
    Ω = MeshDomain(unit_cube(1))
    xc = sum(Ferrite.getcoordinates(Ω.mesh, 1)) / 4
    r = 1.0e-3
    bump = ifelse(sum((X[i] - xc[i])^2 for i in 1:3) < r^2, 1.0, 0.0)
    sys = maxwell_system(Ω; f = [bump, 0.0, 0.0])
    prob = discretize(sys, maxwell_disc(Ω, 1.0))
    @test prob isa LinearProblem && iszero(prob.b)
end

@testset "source amplitudes are validated at the quadrature points" begin
    Ω = MeshDomain(unit_cube(1))
    cell = Ferrite.getcoordinates(Ω.mesh, 1)
    # The first point of the degree-2 rule of the reference tetrahedron, mapped to cell 1: not a centroid.
    ξ = Ferrite.QuadratureRule{RefTetrahedron}(2).points[1]
    xq = cell[1] + ξ[1] * (cell[2] - cell[1]) + ξ[2] * (cell[3] - cell[1]) + ξ[3] * (cell[4] - cell[1])
    ball(amplitude, r = 1.0e-3) = ifelse(sum((X[i] - xq[i])^2 for i in 1:3) < r^2, amplitude, 0.0)
    solve(amplitude) = discretize(maxwell_system(Ω; f = [ball(amplitude), 0.0, 0.0]), maxwell_disc(Ω, 1.0))
    @test solve(1.0) isa LinearProblem  # inside the window, evaluated at a quadrature point
    @test occursin("must be real and finite as a Float64", message(() -> solve(big"1e400")))
    @test occursin("supported window", message(() -> solve(2.0^-1074)))
    @test occursin("supported window", message(() -> solve(1.0e-101)))
end

@testset "a nonzero value that underflows in the conversion to Float64 is rejected" begin
    Ω = MeshDomain(unit_cube(1))
    tiny = big"1e-400"
    run(; kw...) = message(() -> discretize(maxwell_system(Ω; kw...), maxwell_disc(Ω, 1.0)))
    for kw in ((; f = [tiny, 0.0, 0.0]), (; a0 = tiny), (; a2 = tiny), (; a1 = tiny), (; nu = tiny))
        @test occursin("supported window", run(; kw...))
    end
    # ω = 1e-400 is not a positive Float64 either.
    @test message(() -> discretize(maxwell_system(Ω), maxwell_disc(Ω, tiny))) != "no error"
    # Exact zeros of any type are fine.
    for zero_value in (0, 0.0, big"0")
        @test discretize(maxwell_system(Ω; a1 = zero_value, f = [zero_value, 0.0, 0.0]), maxwell_disc(Ω, 1.0)) isa LinearProblem
    end
end

@testset "parameter values are validated before they become Float64" begin
    Ω = MeshDomain(unit_cube(1))
    @parameters p
    run(value; kw...) = message(() -> discretize(maxwell_system(Ω; ps = [p => value], kw...), maxwell_disc(Ω, 1.0)))
    for kw in ((; f = [p, 0.0, 0.0]), (; a0 = p), (; a2 = p), (; a1 = p), (; nu = p))
        @test occursin("supported window", run(big"1e-400"; kw...))
        @test occursin("real and finite as a Float64", run(big"1e400"; kw...))
    end
    # Exact zero and in-window parameters are accepted.
    @test discretize(maxwell_system(Ω; ps = [p => big"0"], f = [p, 0.0, 0.0]), maxwell_disc(Ω, 1.0)) isa LinearProblem
    @test discretize(maxwell_system(Ω; ps = [p => 2.5], a2 = p), maxwell_disc(Ω, 1.0)) isa LinearProblem
end

@testset "structural terms are rejected however small they are next to ν" begin
    Ω = MeshDomain(unit_cube(2))
    D = Differential.(collect(X))
    laplacian(c) = -(D[1](D[1](Ev[c])) + D[2](D[2](Ev[c])) + D[3](D[3](Ev[c])))
    graddiv(c) = -D[c](D[1](Ev[1]) + D[2](Ev[2]) + D[3](Ev[3]))
    rejected(; kw...) = message(() -> discretize(maxwell_system(Ω; kw...), maxwell_disc(Ω, 2.0))) != "no error"
    for scale in (1.0, 1.0e12)
        @test !rejected(; nu = scale, a2 = scale)
        @test rejected(; nu = scale, a2 = scale, extra = c -> scale * laplacian(c))
        @test rejected(; nu = scale, a2 = scale, extra = c -> scale * graddiv(c))
    end
    # A coefficient 1 next to ν = 1e12 is a term all the same.
    @test rejected(; nu = 1.0e12, extra = laplacian)
    @test rejected(; nu = 1.0e12, extra = graddiv)
end

@testset "a boundary condition at one time is not a perfect conductor" begin
    Ω = MeshDomain(unit_cube(2))
    left = boundarypiece(Ω, :left)
    sys(trace) = maxwell_system(Ω; bcs = cross(facet_normal(left), trace) .~ 0)
    documented = [E(t, left)[c] for c in 1:3]
    @test discretize(sys(documented), maxwell_disc(Ω, 1.0)) isa LinearProblem
    for time in (0.0, t + 1, 2t)
        trace = [E(time, left)[c] for c in 1:3]
        @test occursin("not a condition at one time", message(() -> discretize(sys(trace), maxwell_disc(Ω, 1.0))))
    end
end
