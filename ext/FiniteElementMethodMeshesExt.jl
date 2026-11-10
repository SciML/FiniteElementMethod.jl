module FiniteElementMethodMeshesExt

using FiniteElementMethod: FiniteElementMethod
using Ferrite: Ferrite
using LinearAlgebra: cross, dot
using Meshes: Meshes

function FiniteElementMethod.ferrite_grid(mesh::Meshes.Mesh)
    Meshes.crs(mesh) <: Meshes.Cartesian ||
        throw(ArgumentError("ferrite_grid needs a mesh in Cartesian coordinates, got $(nameof(Meshes.crs(mesh)))"))
    dim = Meshes.embeddim(mesh)
    nodes = [Ferrite.Node(Ferrite.Vec{dim}(Tuple(Meshes.ustrip.(Meshes.to(p))))) for p in Meshes.vertices(mesh)]
    x = [node.x for node in nodes]
    cells = [_cell(Meshes.pltype(c), Meshes.indices(c), x) for c in Meshes.elements(Meshes.topology(mesh))]
    return Ferrite.Grid(cells, nodes)
end

# Ferrite needs cells with a positive Jacobian, while Meshes.jl allows either orientation.
function _cell(::Type{<:Meshes.Segment}, (i, j), x)
    reversed = length(x[i]) == 1 && x[j][1] < x[i][1]
    return Ferrite.Line(reversed ? (j, i) : (i, j))
end
function _cell(::Type{<:Meshes.Triangle}, (i, j, k), x)
    clockwise = length(x[i]) == 2 && _cross2(x[j] - x[i], x[k] - x[i]) < 0
    return Ferrite.Triangle(clockwise ? (i, k, j) : (i, j, k))
end
function _cell(::Type{<:Meshes.Quadrangle}, (i, j, k, l), x)
    clockwise = length(x[i]) == 2 && _cross2(x[k] - x[i], x[l] - x[j]) < 0
    return Ferrite.Quadrilateral(clockwise ? (i, l, k, j) : (i, j, k, l))
end
function _cell(::Type{<:Meshes.Tetrahedron}, (i, j, k, l), x)
    inverted = dot(cross(x[j] - x[i], x[k] - x[i]), x[l] - x[i]) < 0
    return Ferrite.Tetrahedron(inverted ? (i, j, l, k) : (i, j, k, l))
end
function _cell(::Type{<:Meshes.Hexahedron}, v, x)
    up = sum(x[v[n]] for n in 5:8) - sum(x[v[n]] for n in 1:4)
    inverted = dot(cross(x[v[3]] - x[v[1]], x[v[4]] - x[v[2]]), up) < 0
    return Ferrite.Hexahedron(inverted ? (v[5], v[6], v[7], v[8], v[1], v[2], v[3], v[4]) : v)
end
function _cell(PL, inds, x)
    throw(ArgumentError("ferrite_grid supports segments, triangles, quadrangles, tetrahedra and hexahedra, got a $(nameof(PL))"))
end

_cross2(a, b) = a[1] * b[2] - a[2] * b[1]

end # module
