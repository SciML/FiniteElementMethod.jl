# FNV-1a checksums of the files the committed images of the axisymmetric swirl page are rendered from. Each checksum
# is committed next to its images (docs/src/assets/axisym_swirl.fnv1a for the two figures, axisym_swirl_animation.fnv1a
# for the animation) and test/NavierStokes/axisym_swirl.jl fails when these files change without a re-render with
# `julia --project=docs examples/axisym_swirl/figures.jl` or `animation.jl`.
const FIGURE_SOURCES = ("mesh.geo", "manufactured.jl", "axisym_swirl.jl", "figures.jl")
const ANIMATION_SOURCES = ("mesh.geo", "manufactured.jl", "axisym_swirl.jl", "animation.jl")

function sources_hash(files, directory = @__DIR__)
    checksum = 0xcbf29ce484222325
    for file in files, byte in read(joinpath(directory, file))
        checksum = (checksum ⊻ byte) * 0x00000100000001b3
    end
    return string(checksum; base = 16, pad = 16)
end

figure_sources_hash(directory = @__DIR__) = sources_hash(FIGURE_SOURCES, directory)
animation_sources_hash(directory = @__DIR__) = sources_hash(ANIMATION_SOURCES, directory)
