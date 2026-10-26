"""
    FieldInfo(dv, name::Symbol, unknown, ip::Ferrite.Interpolation)

One dependent variable of a FEM discretization: the `PDESystem` dependent variable `dv`
(for example `u(t, x, y)`), the name of its field in the `DofHandler`, the array unknown
of the discretized system (for example `u(t)[1:n]`) and the field's interpolation.
"""
struct FieldInfo
    dv::Any
    name::Symbol
    unknown::Any
    ip::Ferrite.Interpolation
end

"""
    FEMMetadata(space, time, x̄, fields, pieces, pdesys; u0 = Float64[])

Discretization metadata passed as the `problem_type` of the `DAEProblem` (or, for a steady
problem with `time === nothing`, the `NonlinearProblem`), so that `solve` returns a
`SciMLBase.PDETimeSeriesSolution` (or `SciMLBase.PDENoTimeSolution`) that is indexed and
called in the variables of the `PDESystem`.

  - `space`: the discrete space; only `space.dh::Ferrite.DofHandler` is used.
  - `time`: the time independent variable, or `nothing` for a steady problem.
  - `x̄`: the spatial independent variables in the order of the `(x, y) ∈ Ω` pairing;
    position `i` is coordinate component `i` of the mesh.
  - `fields`: one [`FieldInfo`](@ref) per dependent variable.
  - `pieces`: maps each boundary piece to the name of its Ferrite facet set.
  - `pdesys`: the `PDESystem` that was discretized.
"""
struct FEMMetadata{hasTime, DH, PDE, Sp} <: SciMLBase.AbstractDiscretizationMetadata{hasTime}
    space::Sp
    time::Any
    x̄::Vector{Any}
    fields::Vector{FieldInfo}
    pieces::Dict{Any, String}
    pdesys::PDE
    metadata::Ref{Any}
    u0::Vector
end

function FEMMetadata(space, time, x̄, fields, pieces, pdesys; u0 = Float64[])
    hasTime = time === nothing ? Val(false) : Val(true)
    return FEMMetadata{hasTime, typeof(space.dh), typeof(pdesys), typeof(space)}(
        space, time, collect(Any, x̄), collect(FieldInfo, fields),
        Dict{Any, String}(pieces), pdesys, Ref{Any}(nothing), u0
    )
end

const FEMTimeSolution = SciMLBase.PDETimeSeriesSolution{T, N, S, <:FEMMetadata} where {T, N, S}
const FEMSteadySolution = SciMLBase.PDENoTimeSolution{T, N, S, <:FEMMetadata} where {T, N, S}
const FEMSolution = Union{FEMTimeSolution, FEMSteadySolution}

"""
    FEMInterpolant(original_sol, md::FEMMetadata)

The `interp` of a FEM solution. It caches the `Ferrite.PointEvalHandler` of the last point
set it evaluated, because building one searches the mesh.
"""
struct FEMInterpolant{S, M}
    sol::S
    md::M
    cache::Ref{Any}
end
FEMInterpolant(sol, md) = FEMInterpolant(sol, md, Ref{Any}(nothing))

function _handler(interp::FEMInterpolant, pts)
    cached = interp.cache[]
    cached !== nothing && cached.pts == pts && return cached.ph
    ph = _locate(_grid(interp.md), pts)
    interp.cache[] = (; pts = copy(pts), ph)
    return ph
end

# Ferrite searches only the cells around the few nearest nodes, which misses points on graded
# meshes, so every point it does not find is searched again among the cells whose bounding
# box contains it.
function _locate(grid, pts)
    ph = Ferrite.PointEvalHandler(grid, pts; warn = false)
    lost = findall(isnothing, ph.cells)
    isempty(lost) && return ph
    boxes = map(1:Ferrite.getncells(grid)) do c
        xs = Ferrite.getcoordinates(grid, c)
        ntuple(d -> extrema(x[d] for x in xs), length(first(xs)))
    end
    for i in lost
        p = pts[i]
        candidates = findall(boxes) do box
            all(d -> (slack = sqrt(eps()) * (box[d][2] - box[d][1]); box[d][1] - slack <= p[d] <= box[d][2] + slack), eachindex(box))
        end
        isempty(candidates) && continue
        found = Ferrite.PointEvalHandler(
            grid, [p]; cellset = candidates, search_nneighbors = Ferrite.getnnodes(grid), warn = false
        )
        ph.cells[i] = found.cells[1]
        ph.local_coords[i] = found.local_coords[1]
    end
    return ph
end

_dh(md::FEMMetadata) = md.space.dh
_grid(md::FEMMetadata) = _dh(md).grid
# Ferrite's FE values default to linear geometry, which breaks on quadratic cells.
_geometry(md::FEMMetadata) = Ferrite.geometric_interpolation(Ferrite.getcelltype(_grid(md)))
_vertex_coordinates(md, d) = [Ferrite.get_node_coordinate(n)[d] for n in Ferrite.getnodes(_grid(md))]

function _wrap_parts(sol, md::FEMMetadata, states)
    nt = length(states)
    T = eltype(first(states))
    umap = Dict{Any, Matrix{T}}()
    for f in md.fields
        out = Matrix{T}(undef, nt, Ferrite.getnnodes(_grid(md)))
        for (k, U) in enumerate(states)
            out[k, :] .= Ferrite.evaluate_at_grid_nodes(_dh(md), U, f.name)
        end
        umap[Num(f.dv)] = out
    end
    xs = ntuple(d -> _vertex_coordinates(md, d), length(md.x̄))
    dvs = [Num(f.dv) for f in md.fields]
    return umap, xs, dvs, FEMInterpolant(sol, md)
end

SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.PDETimeSeriesSolution, ::FEMMetadata) = sol

function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractTimeseriesSolution{T}, md::FEMMetadata) where {T}
    umap, xs, dvs, interp = _wrap_parts(sol, md, sol.u)
    ivdomain = (sol.t, xs...)
    ivs = [Num(md.time), Num.(md.x̄)...]
    return SciMLBase.PDETimeSeriesSolution{
        T, length(dvs), typeof(umap), typeof(md), typeof(sol), typeof(sol.errors), typeof(sol.t),
        typeof(ivdomain), typeof(ivs), typeof(dvs), typeof(sol.prob), typeof(sol.alg),
        typeof(interp), typeof(sol.stats),
    }(
        umap, sol, sol.errors, sol.t, ivdomain, ivs, dvs, md, sol.prob, sol.alg, interp,
        sol.dense, sol.tslocation, sol.retcode, sol.stats
    )
end

SciMLBase.PDENoTimeSolution(sol::SciMLBase.PDENoTimeSolution, ::FEMMetadata) = sol

function SciMLBase.PDENoTimeSolution(sol::SciMLBase.NonlinearSolution{T}, md::FEMMetadata) where {T}
    umap, xs, dvs, interp = _wrap_parts(sol, md, [sol.u])
    umap = Dict{Any, Vector{eltype(sol.u)}}(k => vec(v) for (k, v) in umap)
    ivs = Num.(md.x̄)
    return SciMLBase.PDENoTimeSolution{
        T, length(dvs), typeof(umap), typeof(md), typeof(sol), typeof(xs), typeof(ivs),
        typeof(dvs), typeof(sol.prob), typeof(sol.alg), typeof(interp), typeof(sol.stats),
    }(umap, sol, xs, ivs, dvs, md, sol.prob, sol.alg, interp, sol.retcode, sol.stats)
end

# `u` is a Dict of field => array, not a vector of states, as in MethodOfLines.
Base.length(A::FEMTimeSolution) = length(A.t)
Base.size(A::FEMTimeSolution) = (length(A.t),)
Base.ndims(::FEMTimeSolution) = 1
Base.axes(A::FEMTimeSolution) = (Base.OneTo(length(A.t)),)
Base.firstindex(::FEMTimeSolution) = 1
Base.lastindex(A::FEMTimeSolution) = length(A.t)
function Base.lastindex(::FEMSolution, ::Integer)
    throw(ArgumentError("`end` cannot be resolved in `sol[dv, inds...]`; index the field instead: `sol[dv][end, :]`."))
end

_states(A::FEMTimeSolution) = A.original_sol.u
_states(A::FEMSteadySolution) = [A.original_sol.u]
_stack(::FEMTimeSolution, rows) = permutedims(reduce(hcat, rows))
_stack(::FEMSteadySolution, rows) = only(rows)
_columns(values::AbstractMatrix, idxs) = values[:, idxs]
_columns(values::AbstractVector, idxs) = values[idxs]

function _fieldinfo(md, s)
    SymbolicUtils.iscall(s) || return nothing
    i = findfirst(f -> isequal(SymbolicUtils.operation(Symbolics.unwrap(f.dv)), SymbolicUtils.operation(s)), md.fields)
    return i === nothing ? nothing : md.fields[i]
end

# Trace arguments come back from `arguments` wrapped as constants.
function _piece_arg(s)
    args = map(SymbolicUtils.unwrap_const, SymbolicUtils.arguments(s))
    i = findfirst(a -> a isa DomainSets.Domain, args)
    return i === nothing ? nothing : args[i]
end

function _facetset(md, Γ)
    haskey(md.pieces, Γ) ||
        throw(ArgumentError("$Γ is not a boundary piece of this mesh; the pieces are $(collect(keys(md.pieces)))"))
    return Ferrite.getfacetset(_grid(md), md.pieces[Γ])
end

# `FacetIndex` has no `isless`.
_sorted_facets(md, Γ) = sort!(collect(_facetset(md, Γ)); by = fi -> (fi[1], fi[2]))

"""
    piece_nodes(sol, Γ)

Sorted vertex ids of the boundary piece `Γ`. They are the columns of `sol[u(t, Γ)]` and
`sol(t, Γ; dv)`, and they index `sol[x]`, so `sol[x][piece_nodes(sol, Γ)]` are the
coordinates of the trace values.
"""
piece_nodes(A::FEMSolution, Γ) = piece_nodes(A.disc_data, Γ)
function piece_nodes(md::FEMMetadata, Γ)
    grid = _grid(md)
    # The geometric interpolation's facet dofs are the facet's local nodes, mid-edge nodes included.
    facet_nodes = Ferrite.dirichlet_facetdof_indices(_geometry(md))
    ids = Int[]
    for fi in _sorted_facets(md, Γ)
        cell_nodes = Ferrite.getcells(grid, fi[1]).nodes
        append!(ids, (cell_nodes[i] for i in facet_nodes[fi[2]]))
    end
    return sort!(unique!(ids))
end

function _facet_normal_flux(md, f::FieldInfo, Γ, U)
    dh = _dh(md)
    grid = _grid(md)
    r = Ferrite.dof_range(dh, f.name)
    fv = Ferrite.FacetValues(Ferrite.FacetQuadratureRule{Ferrite.getrefshape(f.ip)}(2), f.ip, _geometry(md))
    return map(_sorted_facets(md, Γ)) do fi
        Ferrite.reinit!(fv, Ferrite.getcoordinates(grid, fi[1]), fi[2])
        u_cell = U[Ferrite.celldofs(dh, fi[1])[r]]
        flux = zero(eltype(U))
        area = 0.0
        for q in 1:Ferrite.getnquadpoints(fv)
            dΓ = Ferrite.getdetJdV(fv, q)
            flux += dot(Ferrite.function_gradient(fv, q, u_cell), Ferrite.getnormal(fv, q)) * dΓ
            area += dΓ
        end
        flux / area
    end
end

"""
    normal_flux(sol, dv, Γ)

Facet-mean outward normal derivative `∇u·n` of the field `dv` on every facet of the boundary
piece `Γ`, `nt × nfacets` (a vector for a steady solution). Facets are ordered by
`(cell, local facet)`. This is a post-processed flux, not the imposed boundary datum.
"""
function normal_flux(A::FEMSolution, dv, Γ)
    md = A.disc_data
    f = _fieldinfo(md, Symbolics.unwrap(dv))
    f === nothing && throw(ArgumentError("$dv is not a dependent variable of this solution; they are $(A.dvs)"))
    return _stack(A, [_facet_normal_flux(md, f, Γ, U) for U in _states(A)])
end

# L2 projection of the cellwise gradient onto the field's own space, assembled and factorised once
# per field. Order 2p integrates both the mass matrix and the (p - 1)-degree gradient against
# p-degree tests.
function _gradient_projector(md, f::FieldInfo)
    qr = Ferrite.QuadratureRule{Ferrite.getrefshape(f.ip)}(2 * _order(f.ip))
    proj = Ferrite.L2Projector(f.ip, _grid(md); qr_lhs = qr)
    return (; proj, qr, cv = Ferrite.CellValues(qr, f.ip, _geometry(md)))
end

# Gradient component `d` of field `f` for dof vector `U`, projected and read at the vertices.
function _nodal_gradient(md, f::FieldInfo, d, U, projector)
    (; proj, qr, cv) = projector
    dh = _dh(md)
    r = Ferrite.dof_range(dh, f.name)
    qpvals = [zeros(eltype(U), Ferrite.getnquadpoints(cv)) for _ in 1:Ferrite.getncells(_grid(md))]
    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        u_cell = U[Ferrite.celldofs(cell)[r]]
        for q in 1:Ferrite.getnquadpoints(cv)
            qpvals[Ferrite.cellid(cell)][q] = Ferrite.function_gradient(cv, q, u_cell)[d]
        end
    end
    return _project_at_nodes(proj, qpvals, qr)
end

_project_at_nodes(proj, qpvals, qr) = Ferrite.evaluate_at_grid_nodes(proj, Ferrite.project(proj, qpvals, qr))
# Ferrite's projection assembles into real buffers, so complex data is projected part by part.
function _project_at_nodes(proj, qpvals::Vector{<:Vector{<:Complex}}, qr)
    re = _project_at_nodes(proj, [real.(v) for v in qpvals], qr)
    im = _project_at_nodes(proj, [imag.(v) for v in qpvals], qr)
    return complex.(re, im)
end

# `Differential(x)(u(t, x, y))` gives `(field, d)` with `x == x̄[d]`; anything else `nothing`.
# Symbolics folds `Dy(Dy(u))` into `Differential(y, 2)`, so the order is read, not the nesting.
function _gradkey(md, s)
    (SymbolicUtils.iscall(s) && SymbolicUtils.operation(s) isa Symbolics.Differential) || return nothing
    D = SymbolicUtils.operation(s)
    D.order == 1 || return nothing
    inner = only(SymbolicUtils.arguments(s))
    f = _fieldinfo(md, inner)
    (f === nothing || _piece_arg(inner) !== nothing) && return nothing
    d = findfirst(x -> isequal(Symbolics.unwrap(x), Symbolics.unwrap(D.x)), md.x̄)
    return d === nothing ? nothing : (f, d)
end

function _key_error(A, sym)
    throw(
        ArgumentError(
            "$sym is not a key of this solution. Index with an independent variable $(A.ivs), " *
                "a dependent variable $(A.dvs), a trace u(t, Γ) on a boundary piece, or a first " *
                "spatial derivative such as Differential(x)(u(t, x, y)); higher derivatives are not " *
                "accepted. Facet normal fluxes come from sol[Dn_Γ(u(t, Γ))] or normal_flux(sol, dv, Γ), and the dof " *
                "coefficients from sol.original_sol[<array unknown>]."
        )
    )
end

function Base.getindex(A::FEMSolution, sym::Num)
    md = A.disc_data
    s = Symbolics.unwrap(sym)
    i = findfirst(iv -> isequal(Symbolics.unwrap(iv), s), A.ivs)
    i === nothing || return A.ivdomain[i]
    haskey(A.u, sym) && return A.u[sym]
    g = _gradkey(md, s)
    if g !== nothing
        projector = _gradient_projector(md, g[1])
        return _stack(A, [_nodal_gradient(md, g..., U, projector) for U in _states(A)])
    end
    if SymbolicUtils.iscall(s) && _is_normal_derivative(SymbolicUtils.operation(s))
        trace = Symbolics.unwrap(only(SymbolicUtils.arguments(s)))
        f = _fieldinfo(md, trace)
        Γ = _piece_arg(trace)
        (f === nothing || !isequal(Γ, SymbolicUtils.operation(s).piece)) && _key_error(A, sym)
        return normal_flux(A, f.dv, Γ)
    end
    f = _fieldinfo(md, s)
    f === nothing && _key_error(A, sym)
    Γ = _piece_arg(s)
    Γ === nothing && _key_error(A, sym)
    return _columns(A.u[Num(f.dv)], piece_nodes(md, Γ))
end

Base.getindex(A::FEMSolution, sym::Num, args...) = A[sym][args...]

function _dofs_at(A::FEMTimeSolution, t::Real)
    lo, hi = minmax(A.t[1], A.t[end])
    lo <= t <= hi || throw(DomainError(t, "outside the saved time span $(A.t[1]) .. $(A.t[end])"))
    return A.original_sol(t)
end

_axis(a) = a isa AbstractVector ? collect(a) : [a]

function _point_gradient(ph, md, f, d, U)
    dh = _dh(md)
    r = Ferrite.dof_range(dh, f.name)
    pv = Ferrite.PointValues(f.ip, _geometry(md))
    out = fill(convert(eltype(U), NaN), length(ph.cells))
    for (i, loc) in enumerate(Ferrite.PointIterator(ph))
        loc === nothing && continue
        Ferrite.reinit!(pv, loc)
        out[i] = Ferrite.function_gradient(pv, 1, U[Ferrite.celldofs(dh, Ferrite.cellid(loc))[r]])[d]
    end
    return out
end

# `states` holds one dof vector per requested time; `tshape` is `(nt,)` or `()`.
function _evaluate(A::FEMSolution, states, tshape, xs, dv, outside)
    md = A.disc_data
    dv === nothing && return map(d -> _evaluate(A, states, tshape, xs, d, outside), A.dvs)
    s = Symbolics.unwrap(dv)
    g = _gradkey(md, s)
    f = g === nothing ? _fieldinfo(md, s) : g[1]
    f === nothing && throw(ArgumentError("dv = $dv is not a dependent variable or a first spatial derivative of one; the dependent variables are $(A.dvs)"))
    if length(xs) == 1 && only(xs) isa DomainSets.Domain
        g === nothing || throw(ArgumentError("dv = $dv: traces of derivatives are not supported; use normal_flux(sol, dv, Γ)"))
        nodes = piece_nodes(md, only(xs))
        rows = [Ferrite.evaluate_at_grid_nodes(_dh(md), U, f.name)[nodes] for U in states]
        return isempty(tshape) ? only(rows) : permutedims(reduce(hcat, rows))
    end
    if length(xs) == 1 && only(xs) isa AbstractVector{<:Ferrite.Vec}
        pts = collect(only(xs))
        pshape = (length(pts),)
    else
        length(xs) == length(md.x̄) ||
            throw(ArgumentError("expected $(length(md.x̄)) spatial arguments $(md.x̄), got $(length(xs))"))
        axes_ = map(_axis, xs)
        pts = vec([Ferrite.Vec{length(xs)}(Float64.(Tuple(c))) for c in Iterators.product(axes_...)])
        pshape = Tuple(length(a) for (a, x) in zip(axes_, xs) if x isa AbstractVector)
    end
    ph = _handler(A.interp, pts)
    if outside === :error
        i = findfirst(isnothing, ph.cells)
        i === nothing || throw(DomainError(pts[i], "$(count(isnothing, ph.cells)) point(s) are outside the mesh"))
    elseif outside !== :nan
        throw(ArgumentError("outside must be :nan or :error, got $(repr(outside))"))
    end
    out = Matrix{eltype(first(states))}(undef, length(states), length(pts))
    for (k, U) in enumerate(states)
        out[k, :] .= g === nothing ? Ferrite.evaluate_at_points(ph, _dh(md), U, f.name) :
            _point_gradient(ph, md, f, g[2], U)
    end
    shp = (tshape..., pshape...)
    return isempty(shp) ? out[1] : reshape(out, shp)
end

const _CallArg = Union{Number, AbstractArray, DomainSets.Domain}

# Documented in docs/src/solution.md: a docstring on a call of SciMLBase's type has no
# binding of this package for Documenter to check against.
function (A::FEMTimeSolution)(t::Union{Real, AbstractVector{<:Real}}, xs::Vararg{_CallArg}; dv = nothing, outside::Symbol = :nan)
    states = [_dofs_at(A, tk) for tk in _axis(t)]
    return _evaluate(A, states, t isa AbstractVector ? (length(t),) : (), xs, dv, outside)
end

function (A::FEMSteadySolution)(xs::Vararg{_CallArg}; dv = nothing, outside::Symbol = :nan)
    return _evaluate(A, _states(A), (), xs, dv, outside)
end

_xml_escape(str) = replace(str, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&apos;")

# A hand-written .pvd keeps WriteVTK, which Ferrite does not re-export publicly, out of the deps.
"""
    write_vtk(sol, base::AbstractString) -> String

Write one `.vtu` file per saved time, `"\$(base)_0001.vtu"` and so on, with every field as
point data, plus a ParaView collection `"\$(base).pvd"` that maps them to `sol.t`. Returns the
path of the `.pvd` file.
"""
function write_vtk(A::FEMTimeSolution, base::AbstractString)
    dh = _dh(A.disc_data)
    entries = String[]
    for (k, U) in enumerate(_states(A))
        name = "$(base)_$(lpad(k, 4, '0'))"
        Ferrite.VTKGridFile(name, dh) do vtk
            Ferrite.write_solution(vtk, dh, U)
        end
        push!(entries, "    <DataSet timestep=\"$(A.t[k])\" part=\"0\" file=\"$(_xml_escape(basename(name))).vtu\"/>")
    end
    pvd = base * ".pvd"
    write(
        pvd, """
        <?xml version="1.0"?>
        <VTKFile type="Collection" version="1.0" byte_order="LittleEndian">
          <Collection>
        $(join(entries, '\n'))
          </Collection>
        </VTKFile>
        """
    )
    return pvd
end
