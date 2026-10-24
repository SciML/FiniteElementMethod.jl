"""
    KernelSpec(eq, kind, piece, f, tangent)

One per-quadrature-point integrand of equation `eq`, called as `f(U, G, Ud, V, W, X, T, N, P)` with state `U`,
field-major gradient `G[(f-1)*dim+d] = ∂u_f/∂x_d`, time derivative `Ud`, test function value `V` and gradient
`W`, coordinates `X`, time `T`, outward normal `N` and parameters `P`. `kind` is `:mass`, `:bilinear`, `:source`
or `:nonlinear`; `piece` is `nothing` for a volume term or the boundary piece of a facet term. `tangent` has the
same arguments and returns `∂f/∂[U; G]` (length `nf + nf*dim`) for `:nonlinear` kernels and is `nothing` otherwise.
The residual convention is `Σ∫vol + Σ∮facet = 0`, i.e. `M u̇ + K u − F + r(u) = 0` with `F = −∫source`.
"""
struct KernelSpec{F, T}
    eq::Int
    kind::Symbol
    piece
    f::F
    tangent::T
end
