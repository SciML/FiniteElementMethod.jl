# Supported equation classes

This page states which classes of equations the package supports and on what evidence. The
table is reproduced verbatim from the design specification (§1.1). Its evidence column cites
the specification's verification runs: "report N" is a numbered prototype or library report,
"red cell" is the independent re-run of those prototypes, and `heat_1d1`, `convdiff2` and
similar names are systems of the PDESystemLibrary corpus. "VERIFIED" means a run supports the
claim; "INFERRED" means the claim was made by reading code or by extrapolation, without a run.

The library run behind most rows is narrower than its headline suggests: 392 of its 427
systems are one random-coefficient advection family. The honest figures are 9 of the library's
12 distinct PDE families accepted by the front end and 7 of 7 attempted systems solved end to
end at nodal convergence rate about 2. The library exercises only time-dependent problems,
mostly 1D, with MethodOfLines endpoint-form conditions on boxes; it has no Robin condition, no
mesh domain, no second time derivative, no integral term, nothing in 3D and nothing steady.

| Tier | Class | Evidence | Status |
|---|---|---|---|
| 1, supported | Second-order parabolic equations in divergence or non-divergence form, scalar or coupled multi-field | `heat_1d1`, `pdesys`, `sph` rates 1.99–2.00 (report 6); non-divergence `a(x,y)Δu` rates 1.96–2.01 and two-field reaction–diffusion to 1.9e-16 (report 5, red cell) | VERIFIED |
| 1 | Their elliptic steady counterparts | steady `a(x,y)Δu + f = 0` and steady nonlinear `k(u)` solved by assembly plus linear or Newton solve at rate 2 (red cell `test_redcell_nondiv.out`, `test_redcell_nl_nonharmonic.out`) | VERIFIED at assembly level; INFERRED through the package's `PDENoTimeSolution` path |
| 1 | Nonlinear through state-dependent coefficients and sources: `k(u)`, `f(u)`, viscous Burgers | `k = 1+u²` rates 1.71–1.94 (report 5, red cell); `burgers_2d` two-field, rates 2.26, 2.43 (report 6) | VERIFIED |
| 1 | Convection–diffusion at moderate cell Péclet number | `convdiff2` rates 2.01, 2.00 against MethodOfLines' 4.75e-2, 2.54e-2, 1.32e-2 (rate about 1) (report 6) | VERIFIED |
| 1 | Curvilinear radial Laplacians through the product rule | `sph`, `4/r² ∂ᵣ(r² ∂ᵣu)`, rate 2.00 on four refinements (report 6) | VERIFIED |
| 1 | Dirichlet (constant and time-dependent), Neumann, Robin | time-dependent Dirichlet in `burgers_2d` and all `funcconv`; Neumann in `pdesys`, `sph` (report 6); Robin in cases (i), (ii) and the 2D DAE prototype (reports 5, 4) | VERIFIED; Robin from reports 4–5 only |
| 1 | 1D and 2D on boxes | `Line` and `Quadrilateral` grids end to end (report 6); 2D DAE with three condition kinds (report 4); two-field P1/P2 solution interface (report 7) | VERIFIED |
| 1 | Unstructured and Gmsh meshes | pieces resolved to facet sets on an L-shaped Ferrite grid and a Gmsh import probed (reports 2, 4); no solve on an unstructured mesh has been run | INFERRED; M2's API-example exit makes it VERIFIED |
| 1 | Mixed saddle-point problems with an inf-sup-stable unequal-order pair chosen by the user: Stokes and incompressible Navier–Stokes with Taylor–Hood Q2/Q1 or P2/P1 | DFG 2D-1 cylinder through the PDESystem front end (#31; test group NavierStokes, test/NavierStokes/dfg_cylinder.jl): tolerance stated before the runs = Schäfer–Turek intervals cD ∈ [5.57, 5.59], cL ∈ [0.0104, 0.0110], Δp ∈ [0.1172, 0.1176]; 16 430 unknowns give cD 5.579436, cL 0.010583, Δp 0.117584 and agree with an independent hand-written Ferrite solver on the same mesh to 1e-7; 148 986 unknowns: relative errors −4.5e-7, −5.1e-6, −5.3e-5 (benchmark (d)); Taylor–Hood Stokes rates 3 and 2 (#28) | VERIFIED for the example's formulation: continuity paired with p explicitly, pressure pinned on a boundary piece, outflow ν∂ₙu = 0, index-2 DAE started from rest; no inf-sup check is made, the user chooses a stable pair |
| 2, accepted but not reliable until SUPG (M4) | Pure advection, advection-dominated problems, inviscid hyperbolic problems | `inviscid_burgers_monotonic` exact to 1e-10 only because its solution lies in the P1 space, so it says nothing about shocks or steep fronts; `convdiff4` at cell Péclet > 1 dropped to rate 1.44 on the coarse mesh (report 6) | VERIFIED limitation |
| 3, rejected with a clean error | Periodic conditions (until M4); third- and fourth-order spatial operators (mixed pre-pass is an M4 candidate); `∂ₜ²` (write a first-order system); integral terms; state-dependent mass; non-conormal natural conditions | Appendix A messages; 16 library rejections all periodic, 7 of them also higher order (report 6) | VERIFIED |
| not evidenced | 3D; steady-only systems through the package path; discontinuous coefficients across interfaces | nothing in any corpus or prototype | neither claimed nor rejected |

The "Unstructured and Gmsh meshes" row is reproduced as the specification states it. Its exit
condition, a solve on a Gmsh mesh, is the API example of [Getting started](@ref), which the
test suite runs (`test/Frontend/api_example.jl`).

!!! warning "Unstabilised Galerkin is centred differencing"
    The Galerkin method without stabilisation behaves like a centred finite-difference
    scheme: it is second-order accurate and non-dissipative. For Tier 2 problems (pure
    advection, advection-dominated and inviscid hyperbolic problems) it therefore produces
    oscillations at steep fronts and loses accuracy when the cell Péclet number exceeds one.
    The front end accepts these problems, but their results are not reliable until
    streamline-upwind Petrov–Galerkin stabilisation (`SUPG(τ)`) ships in milestone M4.

The "not evidenced" row is neither a claim nor a rejection: nothing in any test or prototype
covers 3D, steady-only systems or discontinuous coefficients across interfaces. A row moves
only when a test covers it.

Some rows are wider than this release. The table describes the design, so it lists classes
that the front end of this release still rejects with an error naming the milestone that adds
them: steady problems (M4), nonlinear coefficients such as `k(u)` and time-dependent mass or
stiffness coefficients (M3). [Boundary conditions and limitations](@ref) lists what this
release accepts.
