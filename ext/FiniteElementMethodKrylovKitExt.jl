module FiniteElementMethodKrylovKitExt

using FiniteElementMethod: FiniteElementMethod, KrylovKitEigensolver
using KrylovKit: eigsolve

function FiniteElementMethod._eigsolve(::KrylovKitEigensolver, op, x0, nev; krylovdim, tol, maxiter)
    vals, vecs, info = eigsolve(op, x0, nev, :LM; ishermitian = false, krylovdim, tol, maxiter)
    info.converged < nev && @warn "KrylovKit.eigsolve converged $(info.converged) of $nev eigenpairs"
    return vals, vecs
end

end # module
