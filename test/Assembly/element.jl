# The element routines `_linear_element!` and `_nonlinear_element!` called with caller-owned buffers, as a device
# kernel would call them: plain arrays, and views into 3D arrays with one slab per worker. Gathered over every cell
# they must reproduce what the driver assembles, and with views they must not allocate.
using FiniteElementMethod, Ferrite, LinearAlgebra, Random, Test, AllocCheck, Symbolics
using FiniteElementMethod: AssemblyContext, weak_form, emit_kernels, assemble_linear, assemble_nonlinear, _split,
    _linear_element!, _nonlinear_element!
include(joinpath(@__DIR__, "..", "shared", "strongform_seam.jl"))
using .StrongFormSeam: analyse

# Two coupled fields with cross-diffusion and a nonlinear reaction, volume terms only so the driver assembles
# nothing the element loop below leaves out.
function two_field_problem()
    @variables t x y Du Dw χ a b
    @variables u(..) w(..)
    Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
    U, Wf = u(t, x, y), w(t, x, y)
    eqs = [
        Dt(U) ~ Dx(Du * Dx(U) + χ * Dx(Wf)) + Dy(Du * Dy(U) + χ * Dy(Wf)) + a * x * y - U + U^2 * Wf,
        Dt(Wf) ~ Dx(Dw * Dx(Wf)) + Dy(Dw * Dy(Wf)) + b - U^2 * Wf,
    ]
    wf = weak_form(analyse(eqs, []; depvars = [U, Wf], t, sivs = [x, y], params = [Du, Dw, χ, a, b]))
    grid = generate_grid(Triangle, (6, 6))
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :w, ip)
    close!(dh)
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
    return wf, AssemblyContext(dh, [:u, :w], cv, fv, Dict{String, String}())
end

const P = [1.0, 20.0, 0.3, 0.1, 0.9]
const T = 0.25

# The element loop of a device kernel, on the host: worker `w` of `nw` owns slab `w` of the 3D buffers.
function gathered_linear(ctx, volume; nw = 1)
    dh = ctx.dh
    M, K, F = allocate_matrix(dh), allocate_matrix(dh), zeros(ndofs(dh))
    aM, aK = start_assemble(M), start_assemble(K, F)
    ne = ndofs_per_cell(dh)
    M3, K3, F2 = zeros(ne, ne, nw), zeros(ne, ne, nw), zeros(ne, nw)
    cvs = values(ctx.cellvalues)
    ranges = (dof_range(dh, :u), dof_range(dh, :w))
    for (c, cell) in enumerate(CellIterator(dh))
        foreach(cv -> reinit!(cv, cell), unique(objectid, cvs))
        w = mod1(c, nw)
        Me, Ke, Fe = view(M3, :, :, w), view(K3, :, :, w), view(F2, :, w)
        fill!(Me, 0); fill!(Ke, 0); fill!(Fe, 0)
        for k in volume
            _linear_element!(Me, Ke, Fe, k, Val(k.eq), cvs, getcoordinates(cell), ranges, P, T)
        end
        assemble!(aK, celldofs(cell), Ke, Fe)
        assemble!(aM, celldofs(cell), Me)
    end
    return (; M, K, F)
end

function gathered_nonlinear(ctx, volume, u; jacobian = true)
    dh = ctx.dh
    J, r = allocate_matrix(dh), zeros(ndofs(dh))
    aJ = start_assemble(J, r)
    ne = ndofs_per_cell(dh)
    Je3, re2, u2 = zeros(ne, ne, 2), zeros(ne, 2), zeros(ne, 2)
    cvs = values(ctx.cellvalues)
    ranges = (dof_range(dh, :u), dof_range(dh, :w))
    for (c, cell) in enumerate(CellIterator(dh))
        foreach(cv -> reinit!(cv, cell), unique(objectid, cvs))
        w = mod1(c, 2)
        Je, re, u_elem = view(Je3, :, :, w), view(re2, :, w), view(u2, :, w)
        fill!(Je, 0); fill!(re, 0)
        u_elem .= u[celldofs(cell)]
        for k in volume
            _nonlinear_element!(jacobian ? Je : nothing, re, u_elem, k, Val(k.eq), cvs, getcoordinates(cell), ranges, P, T)
        end
        assemble!(aJ, celldofs(cell), Je, re)
    end
    return (; r, J)
end

element_allocations(f!, args...) = (f!(args...); @allocated f!(args...))

@testset "element routines with caller-owned buffers equal the driver" begin
    wf, ctx = two_field_problem()
    u = rand(Xoshiro(3), ndofs(ctx.dh))
    for kernels in (emit_kernels(wf), emit_kernels(wf; coeffs = false))   # `LinearBlock`s, and the `f` path
        volume, groups = _split(ctx, kernels, (:mass, :bilinear, :source))
        @test isempty(groups)
        A = assemble_linear(ctx, kernels, P; t = T)
        @test norm(A.K) > 0 && norm(A.M) > 0 && norm(A.F) > 0
        for G in (gathered_linear(ctx, volume), gathered_linear(ctx, volume; nw = 3))
            @test G.M == A.M && G.K == A.K && G.F == A.F
        end

        nonlinear, _ = _split(ctx, kernels, (:nonlinear,))
        N = assemble_nonlinear(ctx, kernels, u, P; t = T)
        G = gathered_nonlinear(ctx, nonlinear, u)
        @test norm(N.r) > 0 && G.r == N.r && G.J == N.J
        @test gathered_nonlinear(ctx, nonlinear, u; jacobian = false).r == N.r
    end
end

@testset "element routines on views into 3D buffers do not allocate" begin
    wf, ctx = two_field_problem()
    dh = ctx.dh
    ne = ndofs_per_cell(dh)
    cell = first(CellIterator(dh))
    cvs = values(ctx.cellvalues)
    foreach(cv -> reinit!(cv, cell), unique(objectid, cvs))
    xs = getcoordinates(cell)
    ranges = (dof_range(dh, :u), dof_range(dh, :w))
    A3, B3, f2, u2 = zeros(ne, ne, 4), zeros(ne, ne, 4), zeros(ne, 4), rand(ne, 4)
    Me, Ke, Fe, u_elem = view(A3, :, :, 2), view(B3, :, :, 2), view(f2, :, 2), view(u2, :, 2)
    for kernels in (emit_kernels(wf), emit_kernels(wf; coeffs = false))
        volume, _ = _split(ctx, kernels, (:mass, :bilinear, :source))
        nonlinear, _ = _split(ctx, kernels, (:nonlinear,))
        for k in volume
            args = (Me, Ke, Fe, k, Val(k.eq), cvs, xs, ranges, P, T)
            @test isempty(check_allocs(_linear_element!, map(typeof, args)))
            @test element_allocations(_linear_element!, args...) == 0
        end
        for k in nonlinear, Je in (Ke, nothing)
            args = (Je, Fe, u_elem, k, Val(k.eq), cvs, xs, ranges, P, T)
            @test isempty(check_allocs(_nonlinear_element!, map(typeof, args)))
            @test element_allocations(_nonlinear_element!, args...) == 0
        end
    end
end
