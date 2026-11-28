# Manufactured H(curl) solutions of the driven curl-curl problem on the unit cube with PEC walls, in units with
# ε0 = μ0 = 1 and ω = 1, so the equation solved is (1/μr) curl curl E + (iσ - εr) E = -i J.
using FiniteElementMethod, Ferrite, Test
include(joinpath(@__DIR__, "..", "shared", "maxwell.jl"))

function mms_errors(F, material; ns = (4, 8, 16))
    return map(ns) do n
        grid = unit_cube(n)
        addcellset!(grid, "body", x -> true)
        model = MaxwellModel(grid; materials = Dict("body" => material), pec = CUBE_FACES, eps0 = 1.0, mu0 = 1.0)
        (; eps_r, mu_r, sigma) = material
        f(x) = curlcurl_exact(F, x) ./ mu_r .+ (im * sigma - eps_r) .* F(x)
        sol = solve_driven(model, 1.0, x -> im .* f(x))
        (; hcurl_errors(model, sol.u, F)..., ndofs = ndofs(model.dh))
    end
end

@testset "curl curl E - E = f, lossless" begin
    errs = mms_errors(E_smooth, MaxwellMaterial())
    hs = 1 ./ (4, 8, 16)
    @info "Maxwell MMS lossless" errs rates(getfield.(errs, :hcurl), hs) rates(getfield.(errs, :l2), hs)
    @test all(>=(0.9), rates(getfield.(errs, :hcurl), hs))
    @test all(>=(0.9), rates(getfield.(errs, :l2), hs))
end

@testset "complex lossy material" begin
    material = MaxwellMaterial(; eps_r = 2.0 - 0.5im, mu_r = 2.0, sigma = 1.0)
    errs = mms_errors(E_complex, material)
    hs = 1 ./ (4, 8, 16)
    @info "Maxwell MMS lossy" errs rates(getfield.(errs, :hcurl), hs)
    @test all(>=(0.9), rates(getfield.(errs, :hcurl), hs))
    @test all(>=(0.9), rates(getfield.(errs, :l2), hs))
end
