# The manufactured collapsing core of the axisymmetric swirl example, and the force that makes it an exact
# solution. Both are ours: the force is defined as the residual of the equations of `axisym_swirl.jl` at this
# field, so the finite element error against it is the discretization error and nothing else.
#
# With tau = 1 - t, h = 0.01, xi = r / sqrt(tau) and zeta = z / tau^(1/2 - h), the field has radial scale
# tau^(1/2), axial scale tau^(1/2 - h), velocities of size tau^(-1/2-h) (tau^(-1/2) radially), and a start
# from rest through the ramp s(t) = 1 - exp(-(t / 0.1)^2). It is divergence free (it comes from the Stokes
# streamfunction a s tau^(-1/2-h) r^2 exp(-xi^2/2) G(zeta)) and the pressure balances the swirl (d_r p = u_theta^2 / r).
using ModelingToolkit, Symbolics
using ModelingToolkit: t_nounits as t

@variables r z
const Dt, Dr, Dz = Differential(t), Differential(r), Differential(z)

const H, J0, AMP_Z, AMP_SWIRL = 0.01, 0.05, 1.0, 5.0
const EXPONENT_A, EXPONENT_D = 1 / 2 + H, 1 / 2 - H
τ = 1 - t
ξ, ζ = r / sqrt(τ), z / τ^EXPONENT_D
ramp = 1 - exp(-(t / 0.1)^2)
profile = (ζ + J0) * exp(-ζ^2 / 2)
profile_z = (1 - ζ * (ζ + J0)) * exp(-ζ^2 / 2)
const EXACT = Dict(
    :ur => -AMP_Z * ramp * τ^(-1 / 2) * ξ * exp(-ξ^2 / 2) * profile_z,
    :uz => AMP_Z * ramp * τ^(-EXPONENT_A) * (2 - ξ^2) * exp(-ξ^2 / 2) * profile,
    :ut => AMP_SWIRL * ramp * τ^(-EXPONENT_A) * ξ * exp(-ξ^2 / 2) * exp(-ζ^2 / 2),
    :p => -(AMP_SWIRL^2 / 2) * ramp^2 * τ^(-2 * EXPONENT_A) * exp(-ξ^2) * exp(-ζ^2),
)
"The exact fields as functions `(t, r, z) -> value`, for the error norms."
const EXACT_FUNCTIONS = Dict(k => build_function(v, t, r, z; expression = Val(false)) for (k, v) in EXACT)

laplacian(w) = (1 / r) * Dr(r * Dr(w)) + Dz(Dz(w))

"Strong momentum residuals (left minus right side, no force) of the axisymmetric swirl equations, and the divergence."
function residuals(UR, UT, UZ, P)
    return (
        Dt(UR) + UR * Dr(UR) + UZ * Dz(UR) - UT^2 / r + Dr(P) - (laplacian(UR) - UR / r^2),
        Dt(UT) + UR * Dr(UT) + UZ * Dz(UT) + UR * UT / r - (laplacian(UT) - UT / r^2),
        Dt(UZ) + UR * Dr(UZ) + UZ * Dz(UZ) + Dz(P) - laplacian(UZ),
        (1 / r) * Dr(r * UR) + Dz(UZ),
    )
end
const FORCE = map(expand_derivatives, residuals(EXACT[:ur], EXACT[:ut], EXACT[:uz], EXACT[:p]))

# The force expressions are thousands of characters long. The front end compiles source terms without common
# subexpression elimination and evaluates them at every quadrature point and test function, which made the raw
# expressions 98% of the solve time. Compiled once with `cse = true` and registered as opaque functions of
# (t, r, z), they cost a few percent. They depend on the coordinates only, so the analysis accepts them as sources.
const FORCE_FUNCTIONS = Tuple(build_function(f, t, r, z; expression = Val(false), cse = true) for f in FORCE[1:3])
force_r(t, r, z) = FORCE_FUNCTIONS[1](t, r, z)
force_t(t, r, z) = FORCE_FUNCTIONS[2](t, r, z)
force_z(t, r, z) = FORCE_FUNCTIONS[3](t, r, z)
@register_symbolic force_r(t, r, z)
@register_symbolic force_t(t, r, z)
@register_symbolic force_z(t, r, z)
