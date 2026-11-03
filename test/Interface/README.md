# Interface tests

End-to-end tests written the way a user writes a script, in the style of MethodOfLines.jl's interface
tests: build a `PDESystem`, choose a `FEMDiscretization`, `discretize`, `solve`, and compare with the exact
(manufactured) solution. They use the exported and documented API only: no internals such as
`sol.disc_data` or `FiniteElementMethod.rectangle_grid`. Meshes come from `Ferrite.generate_grid`, as in
the documentation.

Run the group with `GROUP=Interface julia --project=. -e 'using Pkg; Pkg.test()'`. Every `*.jl` file in this
directory runs in its own `@safetestset`, so each file is self-contained.

## Conventions

1. **One narrated testset per case.** The name reads as a sentence stating the result
   (`"P1 elements on a line converge at rate 2 for ..."`).
2. **The comment block opens every testset** and states, in this order: `PDE`, `Domain`, `BCs` (boundary and
   initial conditions), `Exact` (the manufactured solution) and `Shows` (what the case demonstrates).
   A testset that loops over variants uses `$name` in its title and one comment block for all of them.
3. **Short.** A case is about 40 lines at most. If it needs more, move the repeated PDE setup into a small
   function above the testset (see `heat_1d_error` in `heat_1d.jl`) or split the case.
4. **Cheap.** Meshes and time spans are small; a case takes seconds after compilation.
5. **Exact comparison.** Choose a solution inside the element space (error at solver tolerance, tolerance
   `1e-7`) or compare against a smooth one with a stated tolerance, or check the convergence rate with
   `observed_rate` over a refinement sequence. Compare through `sol(t, x, y; dv)` using `l2_error`.
6. **Unsupported inputs** go in `rejections.jl` with `@test_throws "message text" expr`. Do not add `@test_broken`.
7. Shared code is in `test/shared/interface_helpers.jl` (`sample_points`, `l2_error`, `observed_rate`); add to it
   only what two files need.

## Adding a case

Copy the nearest testset, change the PDE, boundary conditions and exact solution, update the comment block,
and add a row to the table below. A new theme is a new file in this directory; nothing needs registering.
The mesh of a box is `MeshDomain(generate_grid(Triangle, (n, n), Vec(0.0, 0.0), Vec(1.0, 1.0)))`; its
boundary pieces are `boundarypiece.(Ω, (:left, :right, :bottom, :top))` in 2D and `(:left, :right)` in 1D.

## Cases

| File | Case | PDE | Dim | Order | Boundary conditions |
|------|------|-----|-----|-------|---------------------|
| `heat_1d.jl` | P1 convergence rate 2 | heat | 1 | P1 | constant Dirichlet |
| `heat_1d.jl` | P2 convergence rate 3 | heat | 1 | P2 | constant Dirichlet |
| `heat_2d.jl` | eigenmode decay on triangles and quadrilaterals | heat | 2 | P1, P2 | zero Dirichlet |
| `heat_2d.jl` | exact reproduction by P2 | heat | 2 | P2 | time-dependent Dirichlet |
| `heat_2d.jl` | P1 convergence rate 2 | heat | 2 | P1 | time-dependent Dirichlet |
| `neumann_robin.jl` | Neumann on three sides | heat | 2 | P2 | Dirichlet and Neumann |
| `neumann_robin.jl` | Robin with parameters `k`, `α` | heat | 2 | P2 | Dirichlet and Robin |
| `neumann_robin.jl` | Dirichlet, Neumann and Robin together | heat | 2 | P2 | all three |
| `neumann_robin.jl` | insulated box | heat | 2 | P1 | zero Neumann |
| `solution_access.jl` | `sol[u]`, `sol[x]`, `sol[y]` shapes | heat with parameter `D` | 2 | P2 | Dirichlet |
| `solution_access.jl` | `sol(t, x, y; dv)` points, grids, outside the mesh | heat | 2 | P2 | Dirichlet |
| `solution_access.jl` | derivative and trace keys | heat | 2 | P2 | Dirichlet |
| `solution_access.jl` | parameter values set the diffusivity | heat | 2 | P2 | Dirichlet |
| `advection_diffusion.jl` | exact reproduction with constant velocity | advection-diffusion with source | 1 | P2 | time-dependent Dirichlet |
| `advection_diffusion.jl` | P2 convergence rate 3 with manufactured source | advection-diffusion with source | 1 | P2 | zero Dirichlet |
| `advection_diffusion.jl` | exact reproduction with velocity vector on triangles and quadrilaterals | advection-diffusion with source | 2 | P2 | time-dependent Dirichlet |
| `advection_diffusion.jl` | P2 convergence rate 3 with manufactured source | advection-diffusion with source | 2 | P2 | zero Dirichlet |
| `reaction_diffusion.jl` | linear reaction with source on triangles and quadrilaterals | reaction-diffusion | 2 | P2 | zero Dirichlet |
| `reaction_diffusion.jl` | decay reaction, P2 convergence rate 3 | reaction-diffusion | 1 | P2 | zero Dirichlet |
| `reaction_diffusion.jl` | two linearly coupled fields, exact | coupled reaction-diffusion | 1 | P2 | time-dependent Dirichlet |
| `reaction_diffusion.jl` | nonlinear reaction rejected (M3) | logistic reaction-diffusion | 2 | P1 | zero Dirichlet |
| `variable_coefficient_diffusion.jl` | `Dx(a Dx(u))`, rate 3 | divergence-form diffusion | 1 | P2 | zero Dirichlet |
| `variable_coefficient_diffusion.jl` | `a Dxx(u)`, rate 3 | non-divergence-form diffusion | 1 | P2 | zero Dirichlet |
| `variable_coefficient_diffusion.jl` | `Dx(a Dx(u)) + Dy(a Dy(u))`, rate 3, triangles and quadrilaterals | divergence-form diffusion | 2 | P2 | zero Dirichlet |
| `variable_coefficient_diffusion.jl` | `a (Dxx(u) + Dyy(u))`, rate 3, triangles and quadrilaterals | non-divergence-form diffusion | 2 | P2 | zero Dirichlet |
| `source_terms.jl` | manufactured source, rates 2 and 3 | heat with source | 1 | P1, P2 | time-dependent Dirichlet |
| `source_terms.jl` | manufactured source, rate 3, triangles and quadrilaterals | heat with source | 2 | P2 | time-dependent Dirichlet |
| `source_terms.jl` | polynomial solution, exact | heat with source | 2 | P2 | time-dependent Dirichlet |
| `rejections.jl` | rejected boundary conditions, equations, options, domains, shared facets | heat | 1 to 2 | | |

## Not covered on this branch

Steady problems, time-last variables, observed expressions, SUPG and Navier-Stokes are not on `master`.
