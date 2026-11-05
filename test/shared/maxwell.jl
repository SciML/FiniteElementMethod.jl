# Manufactured fields and H(curl) error norms for the Maxwell group.
using FiniteElementMethod, Ferrite, ForwardDiff, LinearAlgebra

# Tangential trace zero on every face of the unit cube: each component carries sin(π·) of the two other
# coordinates. The (1 + x_i) factors keep it off the eigenfunctions of curl curl.
E_smooth(x) = [
    sin(π * x[2]) * sin(π * x[3]) * (1 + x[1]),
    sin(π * x[1]) * sin(π * x[3]) * (1 + x[2]^2),
    sin(π * x[1]) * sin(π * x[2]) * (1 + x[1] * x[3]),
]
E_complex(x) = E_smooth(x) .+ im .* [sin(π * x[2]) * sin(π * x[3]), 0.0, sin(2π * x[1]) * sin(π * x[2])]

_jac(F, x) = ForwardDiff.jacobian(F, collect(float.(x)))
_split(F) = (x -> real.(F(x)), x -> imag.(F(x)))

function curl_exact(F, x)
    c(G) = (J = _jac(G, x); [J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2]])
    Fr, Fi = _split(F)
    return c(Fr) .+ im .* c(Fi)
end

# curl curl F = ∇(∇·F) - ΔF, component-wise from Hessians.
function curlcurl_exact(F, x)
    function cc(G)
        H = [ForwardDiff.hessian(y -> G(y)[j], collect(float.(x))) for j in 1:3]
        return [sum(H[j][i, j] for j in 1:3) - tr(H[i]) for i in 1:3]
    end
    Fr, Fi = _split(F)
    return cc(Fr) .+ im .* cc(Fi)
end

"""
L2 and H(curl) errors of the dof vector `u` against `F` on `model`, with a quadrature finer than the assembly's.
"""
function hcurl_errors(model, u, F; order = 4)
    ip = Nedelec{RefTetrahedron, 1}()
    cv = CellValues(QuadratureRule{RefTetrahedron}(order), ip)
    e2 = c2 = 0.0
    for cc in CellIterator(model.dh)
        reinit!(cv, cc)
        u_cell = u[celldofs(cc)]
        x = getcoordinates(cc)
        for q in 1:getnquadpoints(cv)
            xq = spatial_coordinate(cv, q, x)
            dΩ = getdetJdV(cv, q)
            e2 += sum(abs2, collect(function_value(cv, q, u_cell)) .- F(xq)) * dΩ
            c2 += sum(abs2, collect(function_curl(cv, q, u_cell)) .- curl_exact(F, xq)) * dΩ
        end
    end
    return (l2 = sqrt(e2), hcurl = sqrt(e2 + c2))
end

unit_cube(n) = generate_grid(Tetrahedron, (n, n, n), Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
const CUBE_FACES = ["left", "right", "bottom", "top", "front", "back"]

rates(errs, hs) = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]
