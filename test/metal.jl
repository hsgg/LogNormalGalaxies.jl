# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# Runs the pipeline on an Apple GPU in Float32.
#
# Opt-in, and never run by CI: see the guard in runtests.jl. CI is
# ubuntu-latest/x64, and GitHub's macOS runners have no usable Metal GPU either,
# so this can only be exercised on a developer machine:
#
#     LNG_TEST_GPU=1 julia --project -e 'using Pkg; Pkg.test(test_args=["metal"])'
#
# The stages are checked one at a time and each builds its own inputs, so that a
# failure names the function responsible instead of just "simulate_galaxies",
# and so one broken stage does not hide the others.
#
# Note there is no Float64 anywhere: Apple GPUs do not have it. That is why
# element-type genericity had to come first.

using Test
using LogNormalGalaxies
using Metal
using AbstractFFTs
using LinearAlgebra: norm
using Random

const LNG = LogNormalGalaxies


if !Metal.functional()
    @testset "Metal GPU" begin
        @test_skip "Metal is not functional on this machine"
    end
else

@testset "Metal GPU (Float32)" begin

    n = 16
    nxyz = (n, n, n)
    L = 100.0
    kF = (2π / L) .* (1, 1, 1)
    Volume = L^3

    keq = 2e-2
    c = 3 * keq^4
    a = 2e4 * 4 * keq^3
    pkfn(k) = a * k / (c + k^4)
    pk1d = pkfn.((2π / L) .* (0:(n - 1)))

    # One white-noise field, shared by every backend below, so that differences
    # are the pipeline's and not the random number generator's.
    noise = randn(MersenneTwister(20260826), n, n, n)
    host32() = Float32.(noise)
    dev32() = MtlArray(Float32.(noise))

    plan = plan_rfft(MtlArray{Float32}(undef, nxyz...))
    hostplan = LNG.plan_with_fftw(nxyz, Float32)

    @testset "Metal's own FFT contract" begin
        # Everything below rests on these, and none of them is documented as
        # guaranteed, so check them explicitly rather than discovering a
        # regression as a mysterious failure further down.
        x = dev32()
        y = plan * x
        @test y isa MtlArray{ComplexF32,3}
        @test size(y) == (n ÷ 2 + 1, n, n)
        # `\` is not defined by Metal.jl; it comes from AbstractFFTs via
        # plan_inv(), and the whole pipeline uses it six times per simulation.
        z = plan \ y
        @test z isa MtlArray{Float32,3}
        @test Array(z) ≈ Array(x)
        # and it must be cached, or every use rebuilds an MPSGraph plan
        @test inv(plan) === inv(plan)
        # a Float64 normalisation would push Float64 through the device rmul!
        @test inv(plan).scale isa Float32
    end

    @testset "draw_phases" begin
        dk = LNG.draw_phases(plan; deltar=dev32())
        @test dk isa MtlArray{ComplexF32,3}
        @test all(isfinite, Array(dk))

        # same noise, same answer -- the seam must not touch the rng
        @test Array(dk) == Array(LNG.draw_phases(plan; deltar=dev32()))
    end

    # Each stage, GPU against CPU, on identical Float32 inputs. These are one or
    # two operations deep, so they should agree closely; the tolerance is set
    # from what CPU Float32 vs CPU Float64 already costs (see float32.jl, worst
    # case 1.2e-6), since the GPU cannot do better than the arithmetic allows.
    @testset "$name matches the CPU" for (name, op!) in [
            "scale_by_pk!(callable)" =>
                (dk, p) -> LNG.scale_by_pk!(dk, pkfn, 1.5, kF, Volume; rfftplan=p),
            "scale_by_pk!(array)" =>
                (dk, p) -> LNG.scale_by_pk!(dk, pk1d, 1.5, kF, Volume; rfftplan=p),
            "pixel_window!" =>
                (dk, p) -> LNG.pixel_window!(dk, nxyz; voxel_window_correction=1),
            "calc_velocity_component!" =>
                (dk, p) -> LNG.calc_velocity_component!(dk, kF, 3),
            "set_fixed_phase!" =>
                (dk, p) -> LNG.set_fixed_phase!(dk, 0.3),
        ]
        gpu = op!(LNG.draw_phases(plan; deltar=dev32()), plan)
        cpu = op!(LNG.draw_phases(hostplan; deltar=host32()), hostplan)

        @test gpu isa MtlArray
        @test all(isfinite, Array(gpu))
        @test all(isfinite, cpu)

        err = norm(ComplexF64.(Array(gpu)) .- ComplexF64.(cpu)) / norm(ComplexF64.(cpu))
        @info "Metal vs CPU: $name" err
        @test err < 1e-4
    end

    @testset "lognormal transform and global reductions" begin
        dr = plan \ LNG.draw_phases(plan; deltar=dev32())
        @. dr = exp(dr)
        @test LNG.mean_global(dr) isa Float32
        @test isfinite(LNG.mean_global(dr))
        @test LNG.var_global(dr) isa Float32
        @test all(isfinite, LNG.extrema_global(dr))
    end

    # End to end. Judged on global metrics, not element by element: between the
    # shared noise and the output lie two forward and four inverse transforms, a
    # log1p, a complex sqrt, an exp, and a normalisation by the global mean that
    # cancels catastrophically wherever the density contrast is near zero. Metal
    # and FFTW each round differently at every one of those, so a per-element
    # bound would fire on correct code.
    @testset "simulate_galaxies end to end" begin
        common = (; nmesh=n, bias=1.5, f=1, velocity_assignment=0,
                    voxel_window_power=1, rng=MersenneTwister(99))

        xg, vg = simulate_galaxies(3e-4, L, pkfn; common..., deltar=dev32())
        xc, vc = simulate_galaxies(3e-4, L, pkfn; common..., deltar=host32())

        # results come back on the host, at the field's precision
        @test xg isa Matrix{Float32}
        @test vg isa Matrix{Float32}
        @test all(isfinite, xg)
        @test all(isfinite, vg)
        @test size(xg, 2) > 0

        # The two catalogues are *not* expected to have the same size. Poisson
        # sampling turns a 1e-7 difference in the density into a different draw
        # as soon as it straddles a boundary, so the counts can only agree to
        # within shot noise, which is √N for N galaxies. Allow 4σ.
        ng, nc = size(xg, 2), size(xc, 2)
        @info "Metal vs CPU: galaxy count" gpu=ng cpu=nc tolerance=4√nc
        @test abs(ng - nc) <= 4√nc

        # Aggregate statistics must agree even though individual galaxies need
        # not: positions fill the box, velocities have a physical scale.
        for (label, g, cpu) in (("positions", xg, xc), ("velocities", vg, vc))
            rg = sqrt(sum(abs2, Float64.(g)) / size(g, 2))
            rc = sqrt(sum(abs2, Float64.(cpu)) / size(cpu, 2))
            @info "Metal vs CPU: rms $label" gpu=rg cpu=rc
            @test isapprox(rg, rc; rtol=0.05)
        end
    end

    @testset "the one-keyword GPU interface" begin
        # What a user is meant to write. No planner, no Metal-specific method.
        x, v = simulate_galaxies(3e-4, L, pkfn; nmesh=n, bias=1.5, f=1,
                                 deltar=MtlArray{Float32}(undef, nxyz...))
        @test x isa Matrix{Float32}
        @test v isa Matrix{Float32}
        @test all(isfinite, x)
        @test size(x, 2) > 0
    end

end

end  # if Metal.functional()


# vim: set sw=4 et sts=4 :
