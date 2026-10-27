# Baseline of benchmark (d), spec §11.5: Ferrite's own tutorial `docs/src/literate-tutorials/ns_vs_diffeq.jl`
# (Ferrite 1.7, MIT licence), run as published apart from three changes that do not touch the numerics: the mesh
# comes from `dfg_channel_grid` of the example (the same geometry for both sides of the comparison; the tutorial's
# own Gmsh rectangle-minus-circle with `Mesh.MeshSizeMax = 0.05` is kept as `tutorial_grid`), the VTK output loop
# is replaced by `solve` with `saveat`, the progress bar and logging are off, and the linear solver is named as
# UMFPACK: the tutorial predates LinearSolve's current sparse default, whose supernodal LU rejects the change of
# sparsity pattern between Rodas5P's W matrices that UMFPACK, the default when the tutorial was written, accepts. Everything else
# (Q2Q1 Taylor–Hood with a vector velocity, ν = 1e-3, the inflow ramp `min(1.5t, 1.5)`, Rodas5P with the Ferrite
# Dirichlet limiter, `abstol = 1e-4`, `reltol = 1e-5`, `dt = 1e-3`, the free-dof error norm) is the tutorial's.
#
# Prints and returns `(; seconds, ndofs, steps, ...)`; `ns_cylinder.jl` calls it and writes the results file.
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra
using DiffEqBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using ADTypes: AutoFiniteDiff
using LinearSolve: UMFPACKFactorization

include(joinpath(@__DIR__, "..", "..", "examples", "dfg_cylinder", "mesh.jl"))

function tutorial_grid()
    Gmsh.initialize()
    try
        gmsh.option.set_number("General.Verbosity", 2)
        dim = 2
        rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 1.1, 0.41)
        circle_tag = gmsh.model.occ.add_circle(0.2, 0.2, 0, 0.05)
        circle_curve_tag = gmsh.model.occ.add_curve_loop([circle_tag])
        circle_surf_tag = gmsh.model.occ.add_plane_surface([circle_curve_tag])
        gmsh.model.occ.cut([(dim, rect_tag)], [(dim, circle_surf_tag)])
        gmsh.model.occ.synchronize()
        gmsh.model.model.add_physical_group(dim - 1, [6], -1, "bottom")
        gmsh.model.model.add_physical_group(dim - 1, [7], -1, "inlet")
        gmsh.model.model.add_physical_group(dim - 1, [8], -1, "outlet")
        gmsh.model.model.add_physical_group(dim - 1, [9], -1, "top")
        gmsh.model.model.add_physical_group(dim - 1, [5], -1, "cylinder")
        gmsh.option.setNumber("Mesh.Algorithm", 11)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
        gmsh.model.mesh.generate(dim)
        grid = FerriteGmsh.togrid()
        addfacetset!(grid, "walls", union(getfacetset(grid, "bottom"), getfacetset(grid, "top")))
        return grid
    finally
        Gmsh.finalize()
    end
end

function assemble_mass_matrix(cv::MultiFieldCellValues, M::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cv.v)
    n_basefuncs_p = getnbasefunctions(cv.p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cv, cell)
        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            for i in 1:n_basefuncs_v
                φᵢ = shape_value(cv.v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cv.v, q_point, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end
    return M
end

function assemble_stokes_matrix(cv::MultiFieldCellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cv.v)
    n_basefuncs_p = getnbasefunctions(cv.p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        Ferrite.reinit!(cv, cell)
        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cv.v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cv.v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            for j in 1:n_basefuncs_p
                ψ = shape_value(cv.p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cv.v, q_point, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += (divφ * ψ) * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
            end
        end
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end

struct RHSparams{KT, CH, DH, CV, UT}
    K::KT
    ch::CH
    dh::DH
    cv::CV
    u::UT
end

function ferrite_limiter!(u, _, p, t)
    update!(p.ch, t)
    return apply!(u, p.ch)
end

function navierstokes_rhs_element!(dvₑ, vₑ, cv)
    n_basefuncs = getnbasefunctions(cv.v)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        ∇v = function_gradient(cv.v, q_point, vₑ)
        v = function_value(cv.v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cv.v, q_point, j)
            dvₑ[j] -= v ⋅ ∇v' ⋅ φⱼ * dΩ
        end
    end
    return
end

function navierstokes!(du, u_uc, p::RHSparams, t)
    (; K, ch, dh, cv, u) = p
    u .= u_uc
    update!(ch, t)
    apply!(u, ch)
    mul!(du, K, u)
    v_range = dof_range(dh, :v)
    n_basefuncs = getnbasefunctions(cv.v)
    vₑ = zeros(n_basefuncs)
    duₑ = zeros(n_basefuncs)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        v_celldofs = @view celldofs(cell)[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(duₑ, 0.0)
        navierstokes_rhs_element!(duₑ, vₑ, cv)
        assemble!(du, v_celldofs, duₑ)
    end
    return
end

function navierstokes_jac_element!(Jₑ, vₑ, cv)
    n_basefuncs = getnbasefunctions(cv.v)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        ∇v = function_gradient(cv.v, q_point, vₑ)
        v = function_value(cv.v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cv.v, q_point, j)
            for i in 1:n_basefuncs
                φᵢ = shape_value(cv.v, q_point, i)
                ∇φᵢ = shape_gradient(cv.v, q_point, i)
                Jₑ[j, i] -= (φᵢ ⋅ ∇v' + v ⋅ ∇φᵢ') ⋅ φⱼ * dΩ
            end
        end
    end
    return
end

function navierstokes_jac!(J, u_uc, p, t)
    (; K, ch, dh, cv, u) = p
    u .= u_uc
    update!(ch, t)
    apply!(u, ch)
    nonzeros(J) .= nonzeros(K)
    assembler = start_assemble(J; fillzero = false)
    n_basefuncs = getnbasefunctions(cv.v)
    Jₑ = zeros(n_basefuncs, n_basefuncs)
    vₑ = zeros(n_basefuncs)
    v_range = dof_range(dh, :v)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        v_celldofs = @view celldofs(cell)[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(Jₑ, 0.0)
        navierstokes_jac_element!(Jₑ, vₑ, cv)
        assemble!(assembler, v_celldofs, Jₑ)
    end
    return apply!(J, ch)
end

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

"""
    ferrite_ns_vs_diffeq(grid; T = 6.0, ν = 1e-3, vmax = 1.5, Δt_save = 0.1)

Ferrite's tutorial problem on `grid` (facet sets `"inlet"`, `"outlet"`, `"walls"`, `"cylinder"`): build the
operators and integrate to `T`. Returns the solution, the `DofHandler` and the setup and solve times.
"""
function ferrite_ns_vs_diffeq(grid; T = 6.0, ν = 1.0e-3, vmax = 1.5, Δt_save = 0.1)
    setup = @elapsed begin
        dim = 2
        ip_v = Lagrange{RefQuadrilateral, 2}()^dim
        qr = QuadratureRule{RefQuadrilateral}(4)
        ip_p = Lagrange{RefQuadrilateral, 1}()
        ip_geo = Ferrite.geometric_interpolation(Ferrite.getcelltype(grid))
        cellvalues = MultiFieldCellValues(qr, (v = ip_v, p = ip_p), ip_geo)
        dh = DofHandler(grid)
        add!(dh, :v, ip_v)
        add!(dh, :p, ip_p)
        close!(dh)
        ch = ConstraintHandler(dh)
        ∂Ω_noslip = union(getfacetset(grid, "walls"), getfacetset(grid, "cylinder"))
        add!(ch, Dirichlet(:v, ∂Ω_noslip, (x, t) -> Vec((0.0, 0.0)), [1, 2]))
        vᵢₙ(t) = min(t * vmax, vmax)
        parabolic_inflow_profile(x, t) = Vec((4 * vᵢₙ(t) * x[2] * (0.41 - x[2]) / 0.41^2, 0.0))
        add!(ch, Dirichlet(:v, getfacetset(grid, "inlet"), parabolic_inflow_profile, [1, 2]))
        close!(ch)
        update!(ch, 0.0)
        M = assemble_mass_matrix(cellvalues, allocate_matrix(dh), dh)
        K = assemble_stokes_matrix(cellvalues, ν, allocate_matrix(dh), dh)
        u₀ = zeros(ndofs(dh))
        apply!(u₀, ch)
        jac_sparsity = sparse(K)
        apply!(M, ch)
        p = RHSparams(K, ch, dh, cellvalues, copy(u₀))
        rhs = ODEFunction(navierstokes!, mass_matrix = M; jac = navierstokes_jac!, jac_prototype = jac_sparsity)
        problem = ODEProblem(rhs, u₀, (0.0, T), p)
        timestepper = Rodas5P(
            autodiff = AutoFiniteDiff(), step_limiter! = ferrite_limiter!, linsolve = UMFPACKFactorization()
        )
    end
    solve_s = @elapsed sol = solve(
        problem, timestepper; initializealg = NoInit(), dt = 1.0e-3,
        adaptive = true, abstol = 1.0e-4, reltol = 1.0e-5, saveat = Δt_save,
        internalnorm = FreeDofErrorNorm(ch), d_discontinuities = [1.0]
    )
    return (; sol, dh, ch, setup_s = setup, solve_s, ndofs = ndofs(dh), steps = sol.stats.naccept + sol.stats.nreject)
end
