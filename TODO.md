# TODO

## Done

- match_eltype() -- the redundant ternary is gone (`convert` is already the
identity when the type matches). The function stays: it maps real→real and
complex→complex, and does not change the *array* type.

- MersenneTwister -- no reason; tests use StableRNG now.

- ≈ instead of your own norm -- assertions are `≈ … rtol=`; reldiff() survives
only for the @info line, which is the one thing `≈` cannot do.

- Strided = "^2.5" -- already so: the caret is implied in compat syntax.

- PencilArray special cases in pixel_window!(), calc_velocity_component!(),
multiply_by_pkG!(), scale_by_pk!() -- all gone. Separable ops broadcast for
every backend via `broadcast_dim()`, which knows a PencilArray broadcasts in
memory order; gathers use `similar_local()`. New testset "PencilFFTs matches
FFTW in k-space" covers the permutation.

- like_arrays() vs convert()/typeof(arr)(x) -- neither works; src/arrays.jl now
says why. match_eltype() is merged into it: precision and device now change
together, which is what a user-supplied array needs. Handing a host Float64 3-d
pk array to a GPU pipeline used to stack-overflow; there is a test for it now.

- rfftplanner / nmesh -- the core simulate_galaxies() takes `deltar` and derives
the mesh from it; the others are wrappers. `rfftplanner` has to stay: PencilFFTs
builds the plan before it can allocate a matching distributed input.

- (found on the way) `deltar` was documented as `MtlArray{Float32}(undef, ...)`,
i.e. uninitialized memory used as white noise. It *is* the noise and is never
drawn into; docstring and test now pass `randn`.

- (found on the way) test/metal.jl shared one rng *object* between the GPU and
CPU runs, so the second continued the first's stream. With one each they agree
exactly.

- The ARM64 PencilFFTs skips are gone. MPI was never the problem -- MPICH_jll
has an aarch64-apple-darwin build and an RFFT round-trip is exact on 1, 2 and 4
ranks. What failed was `mean_global()`/`var_global()`/`extrema_global()`: a
PencilArray reduces *collectively*, building a user-defined MPI.Op that aarch64
cannot construct (JuliaParallel/MPI.jl#404). They reduce over `local_data()`
now, which also fixes a multi-rank bug -- they paired a local `length(arr)` with
an already-global `mean(arr)`, so `var_global()` rescaled a global variance.
New testset "*_global() matches the host array".


## Open

- Multi-rank PencilFFTs is untested. Single-rank exercises the permutation but
not the local ranges; would need an `mpiexec -n 2` harness.

- scale_by_pk!(pk::AbstractArray{T,2}) sets the DC mode at the local [1,1,1],
which is k⃗ = 0 only on one rank. Pre-existing; see the FIXME there.

- scale_by_pk!(array) does `√(pkG * vol)` on a complex pkG, which for a slightly
negative pkG sits on the branch cut: FFTW and Metal then disagree by a factor
-1 on a handful of modes (18 of 2304 at n=16). The magnitudes agree. Should
probably be a real sqrt with the sign handled explicitly.
