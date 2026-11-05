# The WR-90 waveguide model shared by the Maxwell and GPU groups (exp(+iωt), ports at z = 0 and z = L,
# broad wall along x).
using FiniteElementMethod, Ferrite

const WR90_A, WR90_B = 22.86, 10.16  # mm
const F10 = 2π * 10.0e9
const BETA10 = 158.2383  # rad/m, TE10 at 10 GHz
const K010 = F10 / FiniteElementMethod.C0

# A WR-90 section of length `len` mm with cells of about `h` mm; the bottom face z = 0 is a wave port, the top
# face a wave port too unless `top` names another termination.
function wr90(h, len; top = :port, kwargs...)
    n = ceil.(Int, (WR90_A, WR90_B, len) ./ h)
    grid = generate_grid(Tetrahedron, Tuple(n), Vec(0.0, 0.0, 0.0), Vec(WR90_A, WR90_B, float(len)))
    walls = ["left", "right", "front", "back"]
    ports = [WavePort("bottom"; a_direction = (1, 0, 0), kwargs...)]
    top === :port && push!(ports, WavePort("top"; a_direction = (1, 0, 0), kwargs...))
    return MaxwellModel(
        grid; pec = top === :pec ? [walls; "top"] : walls, absorbing = top === :abc ? ["top"] : String[],
        ports, length_scale = 1.0e-3,
    )
end
