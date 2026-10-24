using FiniteElementMethod
using Test

@testset "Package loads" begin
    @test FiniteElementMethod isa Module
end
