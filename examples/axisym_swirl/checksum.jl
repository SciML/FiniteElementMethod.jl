# FNV-1a checksum of the files the committed figures of the axisymmetric swirl page are rendered from. The
# checksum is committed next to the images (docs/src/assets/axisym_swirl.fnv1a) and test/NavierStokes/axisym_swirl.jl
# fails when these files change without a re-render with `julia --project=docs examples/axisym_swirl/figures.jl`.
const FIGURE_SOURCES = ("mesh.geo", "manufactured.jl", "axisym_swirl.jl", "figures.jl")

function figure_sources_hash(directory = @__DIR__)
    checksum = 0xcbf29ce484222325
    for file in FIGURE_SOURCES, byte in read(joinpath(directory, file))
        checksum = (checksum ⊻ byte) * 0x00000100000001b3
    end
    return string(checksum; base = 16, pad = 16)
end
