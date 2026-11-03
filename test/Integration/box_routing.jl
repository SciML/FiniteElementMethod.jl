# The facet sets left, right, bottom and top are read as faces of a box only when the chain is given its `domain`;
# without one a condition on them is analysed against the mesh's own facets, so a grid that merely has a facet set
# called `left` (a quarter annulus, say) is not taken for the box [-1, 1]².
include(joinpath(@__DIR__, "..", "shared", "references.jl"))

@variables t x y
@variables u(..)
Dx, Dy, Dt = Differential(x), Differential(y), Differential(t)
U = u(t, x, y)

# The square [0, 2]²: its facet sets named left and bottom lie at x = 0 and y = 0, not at -1.
const SQUARE = mesh_setup(Triangle, 1, 4; corners = (Vec(0.0, 0.0), Vec(2.0, 2.0)))
const EQ = Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U))
const REF = [("left", :u, (X, t) -> X[1] + 3 * X[2]), ("bottom", :u, (X, t) -> X[1] + 3 * X[2])]
prescribed(c) = dirichlet_values(constraints(SQUARE, c.dirichlet, Float64[]))

@testset "without a domain, left and bottom are the mesh's facets" begin
    c = chain([EQ], [BC("left", U ~ x + 3y), BC("bottom", U ~ x + 3y)]; depvars = [U], t, sivs = [x, y])
    @test prescribed(c) ≈ dirichlet_values(constraints_reference(SQUARE, REF)) rtol = 1.0e-15
    # A Neumann condition on bottom reaches the facet terms with its own normal, not as the face y = -1.
    n = chain([EQ], [BC("bottom", -Dy(U) ~ x, [0.0, -1.0])]; depvars = [U], t, sivs = [x, y])
    @test any(term -> term.piece == "bottom", n.wf.facet)
end

@testset "with a domain they are the faces at its ends" begin
    box = [(0.0, 2.0), (0.0, 2.0)]
    c = chain([EQ], [BC("left", U ~ x + 3y), BC("bottom", U ~ x + 3y)]; depvars = [U], t, sivs = [x, y], domain = box)
    @test prescribed(c) ≈ dirichlet_values(constraints_reference(SQUARE, REF)) rtol = 1.0e-15
    # A domain that does not match the mesh moves the faces: left is x = 5 whatever the grid says.
    far = chain([EQ], [BC("left", U ~ x)]; depvars = [U], t, sivs = [x, y], domain = [(5.0, 6.0), (5.0, 6.0)])
    @test all(==(5.0), filter(!iszero, prescribed(far)))
end
