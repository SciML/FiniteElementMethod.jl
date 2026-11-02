# Hand-assembled FEM heat problems for the solution-interface tests: u_t = Δu with u = sin(t)
# on the facet set "inlet" (x = 0) and natural zero flux elsewhere, lowered by hand to a
# DAEProblem whose problem_type is a FEMMetadata. No PDESystem front end is involved.
using Ferrite, SciMLBase, SparseArrays, LinearAlgebra, Symbolics
using FiniteElementMethod: FiniteElementMethod, FEMMetadata, FieldInfo
const SII = SciMLBase.SymbolicIndexingInterface
const DomainSets = FiniteElementMethod.DomainSets

struct TestPiece <: DomainSets.Domain{Vector{Float64}}
    name::String
end

square_grid(n) = generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0))

# The unit square without the quadrant (0.5, 1) × (0.5, 1); n must be even.
function lshape_grid(n)
    full = square_grid(n)
    keep = filter(1:getncells(full)) do c
        m = sum(getcoordinates(full, c)) / 3
        !(m[1] > 0.5 && m[2] > 0.5)
    end
    cells = getcells(full)[keep]
    used = sort!(unique!(reduce(vcat, [collect(c.nodes) for c in cells])))
    newid = Dict(old => new for (new, old) in enumerate(used))
    grid = Grid([Triangle(map(i -> newid[i], c.nodes)) for c in cells], getnodes(full)[used])
    return with_inlet(grid)
end

function with_inlet(grid)
    addfacetset!(grid, "inlet", x -> x[1] ≈ 0.0)
    return grid
end

# Mass and stiffness for uncoupled fields in one field-wise numbered DofHandler, with the rows
# of the dofs on `dirichlet_sets` replaced by the algebraic identity.
function assemble_heat(grid, ips::NamedTuple, dirichlet_sets)
    dh = DofHandler(grid)
    for (name, ip) in pairs(ips)
        add!(dh, name, ip)
    end
    close!(dh)
    renumber!(dh, DofOrder.FieldWise())
    ch = ConstraintHandler(dh)
    for name in keys(ips)
        for set in dirichlet_sets
            add!(ch, Dirichlet(name, getfacetset(grid, set), (x, t) -> 0.0))
        end
    end
    close!(ch)
    K = allocate_matrix(dh)
    M = allocate_matrix(dh)
    Kasm, Masm = start_assemble(K), start_assemble(M)
    nd = ndofs_per_cell(dh)
    Ke, Me = zeros(nd, nd), zeros(nd, nd)
    geometry = geometric_interpolation(getcelltype(grid))
    cvs = map(ip -> CellValues(QuadratureRule{RefTriangle}(4), ip, geometry), ips)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        for name in keys(ips)
            cv = cvs[name]
            r = dof_range(dh, name)
            Ferrite.reinit!(cv, cell)
            for q in 1:getnquadpoints(cv), (i, I) in enumerate(r), (j, J) in enumerate(r)
                dΩ = getdetJdV(cv, q)
                Ke[I, J] += (shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j)) * dΩ
                Me[I, J] += shape_value(cv, q, i) * shape_value(cv, q, j) * dΩ
            end
        end
        assemble!(Kasm, celldofs(cell), Ke)
        assemble!(Masm, celldofs(cell), Me)
    end
    pd = copy(ch.prescribed_dofs)
    for d in pd
        K[d, :] .= 0
        M[d, :] .= 0
        K[d, d] = 1
    end
    dropzeros!(K)
    dropzeros!(M)
    return dh, K, M, ch
end

# Symbols and metadata shared by the transient and the steady problem.
function heat_metadata(dh, ips; steady = false)
    @variables t x y u(..) w(..)
    funs = Dict(:u => u, :w => w)
    unknowns = map(collect(keys(ips))) do name
        n = length(dof_range_global(dh, name))
        only(@variables $(Symbol(name, :h))(t)[1:n])
    end
    dvs = [steady ? funs[name](x, y) : funs[name](t, x, y) for name in keys(ips)]
    Γ = TestPiece("inlet")
    fields = [FieldInfo(dvs[i], name, unknowns[i], ips[name]) for (i, name) in enumerate(keys(ips))]
    md = FEMMetadata((; dh), steady ? nothing : t, [x, y], fields, Dict{Any, String}(Γ => "inlet"), nothing)
    return md, (; t, x, y, u, w, dvs, unknowns, Γ, dh)
end

# u = sin(t) on "inlet" from u0 = 0.
function heat_problem(grid, ips::NamedTuple; tend = 1.0)
    dh, K, M, ch = assemble_heat(grid, ips, ("inlet",))
    pd = ch.prescribed_dofs
    md, syms = heat_metadata(dh, ips)
    function resid!(res, du, U, p, t)
        mul!(res, M, du)
        mul!(res, K, U, 1, 1)
        res[pd] .-= sin(t)
        return nothing
    end
    jac!(J, du, U, p, γ, t) = (J .= K .+ γ .* M; nothing)
    N = ndofs(dh)
    u0 = zeros(N)
    # Consistent du0: du = cos(0) = 1 on the inlet, M du = -K u0 elsewhere.
    A0 = copy(M)
    rhs = -K * u0
    for d in pd
        A0[d, d] = 1
        rhs[d] = 1
    end
    du0 = A0 \ rhs
    dvmask = trues(N)
    dvmask[pd] .= false
    elems = reduce(vcat, [collect(Symbolics.unwrap(u)) for u in syms.unknowns])
    sys = SII.SymbolCache(elems, [], Symbolics.unwrap(syms.t))
    fn = DAEFunction{true}(resid!; jac = jac!, jac_prototype = K + M, sys)
    prob = DAEProblem(fn, du0, u0, (0.0, tend), SciMLBase.NullParameters(), md; differential_vars = dvmask)
    return prob, syms
end

# Laplace with u = 1 on "inlet" (x = 0) and u = 0 on "right" (x = 1): u = 1 - x, exact in P1.
function steady_solution(grid, ips::NamedTuple)
    dh, K, _, ch = assemble_heat(grid, ips, ("inlet", "right"))
    md, syms = heat_metadata(dh, ips; steady = true)
    F = zeros(ndofs(dh))
    inlet = ConstraintHandler(dh)
    add!(inlet, Dirichlet(only(keys(ips)), getfacetset(grid, "inlet"), x -> 1.0))
    close!(inlet)
    F[inlet.prescribed_dofs] .= 1
    f!(res, U, p) = (res .= K * U .- F; nothing)
    prob = NonlinearProblem{true}(NonlinearFunction{true}(f!), zeros(ndofs(dh)), SciMLBase.NullParameters(), md)
    U = K \ F
    raw = SciMLBase.build_solution(prob, nothing, U, K * U - F; retcode = SciMLBase.ReturnCode.Success)
    return SciMLBase.wrap_sol(raw), syms
end

# A steady solution holding the given dof vector, for tests of evaluation alone.
function steady_from_dofs(dh, ips::NamedTuple, U)
    md, syms = heat_metadata(dh, ips; steady = true)
    prob = NonlinearProblem{true}(NonlinearFunction{true}((res, u, p) -> nothing), U, SciMLBase.NullParameters(), md)
    raw = SciMLBase.build_solution(prob, nothing, U, zero(U); retcode = SciMLBase.ReturnCode.Success)
    return SciMLBase.wrap_sol(raw), syms
end

# Global dof numbers of field `name`; contiguous after field-wise renumbering.
function dof_range_global(dh, name)
    return sort!(unique!(reduce(vcat, [celldofs(dh, c)[dof_range(dh, name)] for c in 1:getncells(Ferrite.get_grid(dh))])))
end

# du = 0 from the dof vector U over `tspan`, which may run backwards.
function constant_problem(dh, ips::NamedTuple, U, tspan)
    md, syms = heat_metadata(dh, ips)
    resid!(res, du, u, p, t) = (res .= du; nothing)
    prob = DAEProblem(DAEFunction{true}(resid!), zero(U), U, tspan, SciMLBase.NullParameters(), md)
    return prob, syms
end
