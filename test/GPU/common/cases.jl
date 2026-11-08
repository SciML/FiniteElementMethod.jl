# The kernels FEM emits for real PDESystems: each kernel as the `Expr` (`expression = Val(true)`) and as the host function
# `discretize` compiles, for the device-pass tests on every backend.
import FiniteElementMethod as FEM
using FiniteElementMethod: FEMDiscretization, MeshDomain, boundarypiece, NormalDerivative, Galerkin, SUPG, CellCoeffs,
    weak_form, emit_kernels, emit_dirichlet
using ModelingToolkit: ModelingToolkit, PDESystem, Differential, @parameters, @named, t_nounits as t
using DomainSets: Interval
using Ferrite: generate_grid, Triangle, Vec
using Symbolics: Symbolics, @variables, @register_symbolic

# A registered function GPU code can compile, and one that allocates a Dict, which it cannot.
goodf(z) = 1 + z^2 / (1 + z^2)
@register_symbolic goodf(z)
function badf(z)
    d = Dict{Int, Float64}()
    d[1] = z
    return d[1] + 1
end
@register_symbolic badf(z)

@variables x y u(..)
@parameters k a
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)
const Ω = MeshDomain(generate_grid(Triangle, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0)))
const LEFT, RIGHT, TOP = boundarypiece.(Ω, (:left, :right, :top))

# The analysis `discretize` makes of `pdesys` (the front of `FiniteElementMethod._lower`), and its weak form.
function analysis(pdesys, disc)
    PDEBase = FEM.PDEBase
    sys, replaced = PDEBase.make_pdesys_compatible(pdesys)
    v = PDEBase.VariableMap(sys, disc; replaced_vars = replaced)
    PDEBase.interface_errors(sys, v, disc)
    boundarymap = FEM._boundary_map(sys.bcs, v, disc)
    ctx = FEM.StrongForm.SlotContext(sys, v)
    an = FEM._paired(FEM.StrongForm.analyse(sys, v, boundarymap; ibp = disc.ibp), pdesys.eqs, ctx, disc)
    return an, weak_form(an; test_space = disc.test_space)
end

function system(eq, bcs; params = [])
    @named pdesys = PDESystem([eq], [bcs; u(0, x, y) ~ 0], [(x, y) ∈ Ω, t ∈ Interval(0.0, 1.0)], [t, x, y], [U], params)
    return pdesys
end

# Mass, bilinear and source coefficients with a parameter, and a Neumann condition (a facet kernel).
heat() = system(
    Dt(U) ~ Dx((1 + x^2 + y + k * sin(x)) * Dx(U)) + Dy((1 + x^2 + y + k * sin(x)) * Dy(U)) + x * y + 1,
    [u(t, LEFT) ~ 0.1 * y, NormalDerivative(RIGHT)(u(t, RIGHT)) ~ 0.5]; params = [k => 0.3]
)
# SUPG: `CellCoeffs` with the element size hK.
advection() = system(Dt(U) ~ 0.01 * (Dx(Dx(U)) + Dy(Dy(U))) - Dx(U) - 0.5 * Dy(U) + 1, [u(t, LEFT) ~ 0, u(t, RIGHT) ~ 1])
# A nonlinear diffusion and advection term: `f` and `tangent`.
nonlinear() = system(
    Dt(U) ~ Dx((1 + U^2) * Dx(U)) + Dy((1 + U^2) * Dy(U)) - U * Dx(U) + exp(x) * sin(y) / 2,
    [u(t, LEFT) ~ sin(y) / 2]
)
# `NaNMath.pow` for a constant non-integer and a symbolic exponent, and NaNMath's log and sqrt.
powers() = system(
    Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + (1 + x^2)^1.5 + (2 + y)^a + log(2 + x) + sqrt(1 + y) + t^0.25,
    [u(t, LEFT) ~ 0]; params = [a => 0.7]
)
# Time-dependent Dirichlet data with a fractional power.
dirichlet() = system(
    Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)), [u(t, LEFT) ~ sin(t) * (1 + y)^0.5 + k * y, u(t, TOP) ~ 2.0 * t];
    params = [k => 0.3]
)
registered(fn) = system(Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + fn(x + y), [u(t, LEFT) ~ 0])

const CASES = [
    "heat" => (heat, Galerkin()),
    "supg" => (advection, SUPG((hK, b) -> hK / (2 * sqrt(b[1]^2 + b[2]^2)))),
    "nonlinear" => (nonlinear, Galerkin()),
    "powers" => (powers, Galerkin()),
    "dirichlet" => (dirichlet, Galerkin()),
    "goodf" => (() -> registered(goodf), Galerkin()),
]

# One `(role, Expr, host function, signature)` per kernel function of `pdesys`. Signatures: `:coeffs` is
# `(X, T, N, P)`, `:cell` adds `hK`, `:full` is `(U, G, Ud, V, W, X, T, N, P)` and `:dirichlet` is `(X, T, P)`.
function kernel_rows(pdesys, test_space)
    disc = FEMDiscretization(Ω, t; test_space)
    an, wf = analysis(pdesys, disc)
    rows = Tuple{String, Expr, Any, Symbol}[]
    for (e, f) in zip(emit_kernels(wf; expression = Val(true)), emit_kernels(wf))
        tag = string(e.kind, e.piece === nothing ? "" : "@" * string(e.piece.name))
        if e.coeffs isa CellCoeffs
            push!(rows, (tag * ".cellcoeffs", e.coeffs.f, f.coeffs.f, :cell))
        elseif e.coeffs !== nothing
            push!(rows, (tag * ".coeffs", e.coeffs, f.coeffs, :coeffs))
        end
        e.f === nothing || push!(rows, (tag * ".f", e.f, f.f, :full))
        e.tangent === nothing || push!(rows, (tag * ".tangent", e.tangent, f.tangent, :full))
    end
    for (i, ((_, _, ex), (_, _, g))) in enumerate(zip(emit_dirichlet(an; expression = Val(true)), emit_dirichlet(an)))
        push!(rows, ("dirichlet$i", ex, g, :dirichlet))
    end
    return rows, length(an.slots.P)
end

# Every `(case, role, Expr, host function, signature, number of parameters)` of `CASES`.
function all_rows(cases = CASES)
    out = []
    for (name, (pdesys, test_space)) in cases
        rows, np = kernel_rows(pdesys(), test_space)
        append!(out, [(name, role, ex, g, sig, np) for (role, ex, g, sig) in rows])
    end
    return out
end
