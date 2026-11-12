# Cartesian UPML: a PEC-backed layer terminating a WR-90 guide, a conducting filling continued into the layer,
# a radiating source in a PML-truncated vacuum box against a larger domain, and input validation.
using FiniteElementMethod, Ferrite, SparseArrays, LinearAlgebra, Random, Test

const PML_A, PML_B = 22.86, 10.16  # mm
const PML_F10 = 2π * 10.0e9
const PML_WALLS = ["left", "right", "front", "back"]

# A WR-90 guide of `len` mm with a wave port at z = 0, continued by a `d` mm PML backed by PEC at its end.
# Cells from `lossy_from` on form the cell set "lossy".
function wr90_pml(h, len, d; lossy_from = len, materials = Dict{String, MaxwellMaterial}(), kwargs...)
    n = (ceil(Int, PML_A / h), ceil(Int, PML_B / h), round(Int, (len + d) / h))
    grid = generate_grid(Tetrahedron, n, Vec(0.0, 0.0, 0.0), Vec(PML_A, PML_B, len + d))
    addcellset!(grid, "pml", x -> x[3] >= len - 1.0e-9)
    addcellset!(grid, "lossy", x -> x[3] >= lossy_from - 1.0e-9)
    return MaxwellModel(
        grid; materials, pec = [PML_WALLS; "top"], ports = [WavePort("bottom"; a_direction = (1, 0, 0))],
        pml = CartesianPML("pml"; omega = PML_F10, kwargs...), length_scale = 1.0e-3,
    )
end

@testset "PEC-backed PML terminating WR-90" begin
    model = wr90_pml(2.0, 20.0, 20.0)
    A = maxwell_operator(model, PML_F10)
    @test issymmetric(A) && !ishermitian(A)
    # The layer adds blocks for the real and imaginary parts of its curl and permittivity terms only.
    @test length(model.terms) == 4 + 4 + 1
    sweep = solve_sweep(model, 2π .* [8.0e9, 10.0e9, 12.0e9])
    S11 = abs.(sweep.S[1, 1, :])
    @info "WR-90 terminated by a PML" S11
    @test all(<(0.01), S11)  # the first-order ABC reflects 0.1396 at 10 GHz
end

@testset "conducting filling continued into the PML" begin
    # The guide is vacuum to z = 20 mm, then conducting through the PML: a matched lossy half-space, which
    # reflects (β0 - β1)/(β0 + β1) at the interface.
    σ = 0.2
    model = wr90_pml(2.0, 30.0, 20.0; lossy_from = 20.0, materials = Dict("lossy" => MaxwellMaterial(; sigma = σ)))
    @test length(model.terms) == 4 + 6 + 1
    kc = π / (PML_A * 1.0e-3)
    k0 = PML_F10 / FiniteElementMethod.C0
    β0 = sqrt(k0^2 - kc^2)
    β1 = sqrt(complex(k0^2 * (1 - im * σ / (PML_F10 * FiniteElementMethod.EPS0)) - kc^2))
    β1 = imag(β1) > 0 ? -β1 : β1
    expected = (β0 - β1) / (β0 + β1) * exp(-2im * β0 * 20.0e-3)
    S11 = only(solve_sweep(model, [PML_F10]).S)
    @info "WR-90 conducting load in the PML" S11 expected abs(S11 - expected)
    @test abs(S11 - expected) < 0.01
end

@testset "radiating source in a PML-truncated box" begin
    # λ = 1 and a Gaussian current at the centre of the box [-a, a]³: the layer meets the radiated field at
    # every angle, corners included. The reference is the same mesh with the layer twice as far out; the
    # discretisation error itself (h = λ/10) is far larger and common to both.
    k, w, a, d, h = 2π, 0.08, 0.4, 0.3, 0.1
    J(x) = [0.0, 0.0, exp(-sum(abs2, x) / w^2)]
    faces = ["left", "right", "bottom", "top", "front", "back"]
    function box_model(inner, ext; kwargs...)
        n = round(Int, 2ext / h)
        grid = generate_grid(Tetrahedron, (n, n, n), Vec(-ext, -ext, -ext), Vec(ext, ext, ext))
        # By centroid: a tetrahedron with every node on the box surface can still lie inside the box.
        inner < ext && addcellset!(grid, "pml", Set(c for c in 1:getncells(grid) if maximum(abs.(sum(getcoordinates(grid, c)) / 4)) > inner))
        return MaxwellModel(grid; eps0 = 1.0, mu0 = 1.0, quadrature = 4, kwargs...)
    end
    pml_model(inner) = box_model(inner, inner + d; pec = faces, pml = CartesianPML("pml"; omega = k, R = 1.0e-3))
    Random.seed!(1)
    points = [Vec(Tuple(r * normalize(randn(3)))) for r in range(0.2, 0.38; length = 200)]
    field(model) = field_values(solve_driven(model, k, J), points).E
    relative(E, ref) = sqrt(sum(e -> sum(abs2, e), E .- ref) / sum(e -> sum(abs2, e), ref))
    model = pml_model(a)
    ref = field(pml_model(a + 0.2))
    errors = (pml = relative(field(model), ref), abc = relative(field(box_model(a, a; absorbing = faces)), ref))
    @info "radiating source in a box, against a larger domain" ndofs(model.dh) errors
    @test issymmetric(maxwell_operator(model, k))
    @test errors.pml < 0.03
    @test errors.pml < errors.abc / 3
end

@testset "PML input validation" begin
    @test_throws "positive" CartesianPML("pml"; omega = 0.0)
    @test_throws "in (0, 1)" CartesianPML("pml"; omega = 1.0, R = 1.0)
    @test_throws "nonnegative" CartesianPML("pml"; omega = 1.0, order = -1)
    @test_throws "at least one" CartesianPML(String[]; omega = 1.0)
    @test_throws "lo < hi" CartesianPML("pml"; omega = 1.0, box = ((0, 0, 0), (1, 0, 1)))
    grid() = (g = generate_grid(Tetrahedron, (2, 2, 4)); addcellset!(g, "pml", x -> x[3] >= 0.5 - 1.0e-9); g)
    @test_throws "no cell set" MaxwellModel(grid(); pml = CartesianPML("nope"; omega = 1.0))
    g = grid()
    addcellset!(g, "none", Int[])
    @test_throws "are empty" MaxwellModel(g; pml = CartesianPML("none"; omega = 1.0))
    g = grid()
    addcellset!(g, "all", x -> true)
    @test_throws "every cell is in the PML" MaxwellModel(g; pml = CartesianPML("all"; omega = 1.0))
    # An explicit box: zero thickness when it holds every cell, and cells on the wrong side of it are errors.
    @test_throws "zero thickness" MaxwellModel(g; pml = CartesianPML("all"; omega = 1.0, box = ((-1, -1, -1), (1, 1, 1))))
    @test_throws "inside the inner PML box but in" MaxwellModel(
        grid(); pml = CartesianPML("pml"; omega = 1.0, box = ((-1, -1, -1), (1, 1, 0.8)))
    )
    @test_throws "outside the inner PML box but not" MaxwellModel(
        grid(); pml = CartesianPML("pml"; omega = 1.0, box = ((-1, -1, -1), (1, 1, -0.2)))
    )
    @test_throws "absorbing facet set \"top\" touches PML cells" MaxwellModel(
        grid(); absorbing = ["top"], pml = CartesianPML("pml"; omega = 1.0)
    )
    @test_throws "wave port facet set \"top\" touches PML cells" MaxwellModel(
        grid(); pec = ["left", "right", "front", "back"], ports = [WavePort("top"; a_direction = (1, 0, 0))],
        pml = CartesianPML("pml"; omega = 1.0)
    )
    # A box given explicitly equal to the inferred one builds the same operator.
    inferred = MaxwellModel(grid(); pec = ["top"], pml = CartesianPML("pml"; omega = 2.0))
    explicit = MaxwellModel(grid(); pec = ["top"], pml = CartesianPML("pml"; omega = 2.0, box = ((-1, -1, -1), (1, 1, 0.5))))
    @test maxwell_operator(inferred, 2.0) == maxwell_operator(explicit, 2.0)
end

@testset "the PML operator is conjugate symmetric in ω" begin
    grid = generate_grid(Tetrahedron, (1, 1, 4), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 4.0))
    addcellset!(grid, "pml", x -> x[3] >= 3.0 - 1.0e-9)
    model = MaxwellModel(
        grid; pec = ["left", "right", "front", "back", "top"], pml = CartesianPML("pml"; omega = 2.0), eps0 = 1.0, mu0 = 1.0
    )
    A, Aneg = maxwell_operator(model, 2.0), maxwell_operator(model, -2.0)
    @test norm(Aneg - conj(A)) / norm(A) < 1.0e-12
end
