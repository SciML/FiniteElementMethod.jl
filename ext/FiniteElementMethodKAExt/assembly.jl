# Device assembly of the linear operators M, K and F (`FEMDiscretization(...; backend)`): the cells of each
# SubDofHandler are scattered by KernelAbstractions kernels that call the host driver's element routine on per-worker
# element buffers, and the result is copied back into the host operators.

FEM.is_device_backend(::KA.Backend) = true
FEM.supports_float64(::KA.Backend) = true
FEM.device_matrix_type(::KA.CPU, Tv, Ti) = SparseMatrixCSC{Tv, Ti}
FEM.device_matrix_type(backend::KA.Backend, Tv, Ti) = throw(
    ArgumentError(
        "no device sparse matrix type is known for $(nameof(typeof(backend))): load its vendor package " *
            "(AMDGPU for ROCBackend()), or GenericSparseArrays for a backend without a sparse matrix library"
    )
)

function check_ferrite_gpu(ext)
    ext === nothing && throw(
        ArgumentError(
            "device assembly needs a Ferrite with GPU support (the FerriteKAExt extension, Ferrite.jl PR #1496); " *
                "the loaded Ferrite $(pkgversion(Ferrite)) has none"
        )
    )
    return nothing
end

function working_precision(backend, precision)
    T = something(precision, Float64)
    T === Float64 && !FEM.supports_float64(backend) &&
        throw(ArgumentError("$(nameof(typeof(backend))) has no Float64; pass precision = Float32"))
    return T
end

# Workers per launch and the workgroup size. On the CPU a workgroup runs on one thread, so a worker is a workgroup.
launch_shape(::KA.CPU) = (; workers = 4 * Threads.nthreads(), groupsize = 1)
launch_shape(::KA.Backend) = (; workers = 2^14, groupsize = 64)

# A device matrix with the stored entries of `A`, so the copy back needs only the values. Vendor matrix types are
# built from device storage vectors, as Ferrite's `allocate_matrix` builds them.
FEM.device_matrix(::Type{SparseMatrixCSC{Tv, Ti}}, backend, A) where {Tv, Ti} = SparseMatrixCSC{Tv, Ti}(A)
function FEM.device_matrix(::Type{MT}, backend, A) where {MT}
    Tv, Ti = eltype(MT), eltype(rowvals(A))
    storage(v, S) = adapt(backend, Vector{S}(v))
    return MT(storage(SparseArrays.getcolptr(A), Ti), storage(rowvals(A), Ti), storage(nonzeros(A), Tv), size(A))
end

device_fn(::Nothing, T) = nothing
device_fn(f::RuntimeGeneratedFunction, T) = FEM.device_kernel(RuntimeGeneratedFunctions.get_expression(f), T)
device_fn(c::FEM.CellCoeffs, T) = FEM.CellCoeffs(device_fn(c.f, T), c.fields)
function device_block(b::FEM.LinearBlock, T)
    return FEM.LinearBlock(b.eq, device_fn(b.mass, T), device_fn(b.bilinear, T), device_fn(b.source, T))
end
device_block(k, T) = throw(
    ArgumentError("device assembly needs the coefficient form of every linear kernel, but the $(k.kind) kernel of equation $(k.eq) has none")
)

host_fns(b::FEM.LinearBlock) = filter(!isnothing, [b.mass, b.bilinear, b.source])
host_expr(f::RuntimeGeneratedFunction) = RuntimeGeneratedFunctions.get_expression(f)
host_expr(c::FEM.CellCoeffs) = host_expr(c.f)

function quadrature_in(::Type{T}, qr::Ferrite.QuadratureRule{shape}) where {T, shape}
    return Ferrite.QuadratureRule{shape}(T.(Ferrite.getweights(qr)), map(x -> Ferrite.Vec{length(x), T}(Tuple(x)), Ferrite.getpoints(qr)))
end
function cellvalues_in(::Type{T}, cv) where {T}
    qr = quadrature_in(T, Ferrite.get_quadrature_rule(cv))
    return Ferrite.CellValues(T, qr, Ferrite.function_interpolation(cv), Ferrite.geometric_interpolation(cv))
end

# The DofHandler on a copy of the grid with coordinates in precision T, with the host dof numbering.
dofhandler_in(::Type{T}, dh::Ferrite.DofHandler{dim, <:Ferrite.Grid{dim, <:Any, T}}) where {T, dim} = dh
function dofhandler_in(::Type{T}, dh::Ferrite.DofHandler{dim}) where {T, dim}
    nodes = [Ferrite.Node(Ferrite.Vec{dim, T}(Tuple(Ferrite.get_node_coordinate(n)))) for n in Ferrite.getnodes(dh.grid)]
    dhT = Ferrite.DofHandler(Ferrite.Grid(Ferrite.getcells(dh.grid), nodes))
    for sdh in dh.subdofhandlers
        sub = Ferrite.SubDofHandler(dhT, sdh.cellset)
        foreach((name, ip) -> Ferrite.add!(sub, name, ip), sdh.field_names, sdh.field_interpolations)
    end
    Ferrite.close!(dhT)
    dhT.cell_dofs_offset == dh.cell_dofs_offset && length(dhT.cell_dofs) == length(dh.cell_dofs) ||
        error("the DofHandler rebuilt in $T has a different layout from the host DofHandler")
    copyto!(dhT.cell_dofs, dh.cell_dofs)
    return dhT
end

# One SubDofHandler on the device: its dof range per field, the cells of each launch (one per color, or all cells
# for atomic assembly), and per worker a CellCache, the CellValues of every field and the element buffers.
struct DeviceSubdomain{R, L, CC, CV, A3, A2}
    ranges::R
    launches::L
    cells::CC
    values::CV
    Me::A3
    Ke::A3
    Fe::A2
end

struct DeviceAssemblyContext{B, S}
    backend::B
    subdomains::S
    atomic::Bool
end

function DeviceAssemblyContext(backend, ctx::FEM.AssemblyContext{NF}, ::Type{T}; atomic = false) where {NF, T}
    dh = dofhandler_in(T, ctx.dh)
    device_dh = adapt(backend, dh)
    values = map(cv -> cellvalues_in(T, cv), Tuple(ctx.cellvalues))
    shape = launch_shape(backend)
    subdomains = map(enumerate(ctx.dh.subdofhandlers)) do (i, sdh)
        ranges = ntuple(f -> FEM._field_range(sdh, ctx.fields[f]), Val(NF))
        groups = atomic ? [collect(Int, sdh.cellset)] : Ferrite.create_coloring(ctx.dh.grid, sdh.cellset)
        nw = min(maximum(length, groups), shape.workers)
        ne = Ferrite.ndofs_per_cell(sdh)
        DeviceSubdomain(
            ranges, [adapt(backend, g) for g in groups],
            Ferrite.distribute_to_workers(backend, Ferrite.CellCache(device_dh.subdofhandlers[i]), nw),
            map(cv -> Ferrite.distribute_to_workers(backend, cv, nw), values),
            KA.zeros(backend, T, nw, ne, ne), KA.zeros(backend, T, nw, ne, ne), KA.zeros(backend, T, nw, ne)
        )
    end
    return DeviceAssemblyContext(backend, Tuple(subdomains), atomic)
end

workers(s::DeviceSubdomain) = size(s.Me, 1)

@kernel function linear_cells!(aM, aK, @Const(cellids), cells, values, Mes, Kes, Fes, kernels, ranges, p, t)
    w = @index(Global, Linear)
    stride = prod(@ndrange())
    cc, vals = cells[w], map(v -> v[w], values)
    am, ak = aM[w], aK[w]
    Me, Ke, Fe = view(Mes, w, :, :), view(Kes, w, :, :), view(Fes, w, :)
    op = FEM.LinearOp(nothing, nothing, Fe, Me, Ke, Fe)
    for task in w:stride:length(cellids)
        Ferrite.reinit!(cc, cellids[task])
        FEM._reinit_all!(vals, cc)
        fill!(Me, 0)
        fill!(Ke, 0)
        fill!(Fe, 0)
        FEM._kernels!(op, kernels, vals, Ferrite.getcoordinates(cc), ranges, p, t)
        dofs = Ferrite.celldofs(cc)
        Ferrite.assemble!(ak, dofs, Ke, Fe)
        Ferrite.assemble!(am, dofs, Me)
    end
end

# JLArrays 0.3.3, the last release for KernelAbstractions 0.9, defines no `synchronize`; its launches are synchronous.
sync(backend) = hasmethod(KA.synchronize, Tuple{typeof(backend)}) ? KA.synchronize(backend) : nothing

# Adds the device `kernels` (`LinearBlock`s of prepared functions) into `Md`, `Kd` and `Fd`.
function assemble_device!(Md, Kd, Fd, dctx::DeviceAssemblyContext, kernels, p, t; callees = String[])
    backend = dctx.backend
    for s in dctx.subdomains
        nw = workers(s)
        aM = Ferrite.distribute_to_workers(backend, Ferrite.start_assemble(Md; fillzero = false, atomic = dctx.atomic), nw)
        aK = Ferrite.distribute_to_workers(backend, Ferrite.start_assemble(Kd, Fd; fillzero = false, atomic = dctx.atomic), nw)
        kernel! = linear_cells!(backend, launch_shape(backend).groupsize)
        for cellids in s.launches
            launch() = kernel!(aM, aK, cellids, s.cells, s.values, s.Me, s.Ke, s.Fe, kernels, s.ranges, p, t; ndrange = min(length(cellids), nw))
            launch_checked(launch, backend, callees)
            sync(backend)
        end
    end
    return Md, Kd, Fd
end

function FEM.device_assemble_linear!(backend::KA.Backend, precision, assembly, M, K, F, ctx::FEM.AssemblyContext, kernels, p, t)
    check_ferrite_gpu(Base.get_extension(Ferrite, :FerriteKAExt))
    T = working_precision(backend, precision)
    volume, groups = FEM._split(ctx, kernels, FEM.LINEAR_KINDS)
    isempty(groups) || throw(ArgumentError("device assembly does not support facet kernels yet"))
    dkernels = map(k -> device_block(k, T), volume)
    callees = sort!(unique!(reduce(vcat, [foreign_callees(host_expr(f)) for b in volume for f in host_fns(b)]; init = String[])))
    dctx = DeviceAssemblyContext(backend, ctx, T; atomic = assembly === :atomic)
    MT = FEM.device_matrix_type(backend, T, eltype(rowvals(K)))
    zero_pattern = SparseMatrixCSC(size(K)..., SparseArrays.getcolptr(K), rowvals(K), zeros(T, length(nonzeros(K))))
    Md, Kd = FEM.device_matrix(MT, backend, zero_pattern), FEM.device_matrix(MT, backend, zero_pattern)
    Fd = KA.zeros(backend, T, length(F))
    assemble_device!(Md, Kd, Fd, dctx, dkernels, Tuple(T.(p)), T(t); callees)
    nonzeros(M) .= Array(nonzeros(Md))
    nonzeros(K) .= Array(nonzeros(Kd))
    F .= Array(Fd)
    return M, K, F
end
