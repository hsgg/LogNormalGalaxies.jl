# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# Tests that the pipeline really runs at the requested precision.
#
# The element-type assertions are the substance here. A Float64 scalar
# multiplying a Float32 array silently promotes the result back to Float64,
# which produces correct-looking output on the CPU while defeating the whole
# point of T=Float32 -- and is fatal on a GPU with no Float64 at all. Only
# checking eltype at every stage catches that.
#
# Float64-vs-Float32 agreement is checked per operation on *identical* inputs,
# because that is deterministic. Comparing two full simulations is not: randn!()
# consumes a different number of random bits for Float32 than for Float64, so
# the same seed gives different noise, different Poisson draws, and a different
# number of galaxies.

using Test
using LogNormalGalaxies
using LinearAlgebra: norm
using Random

const LNG = LogNormalGalaxies


# relative difference, computed in Float64 so the metric itself is not the
# limiting precision
reldiff(a, b) = norm(ComplexF64.(a) .- ComplexF64.(b)) / norm(ComplexF64.(a))


function f32_testpk()
    keq = 2e-2
    c = 3 * keq^4
    a = 2e4 * 4 * keq^3
    return k -> a * k / (c + k^4)
end


@testset "Float32/Float64 genericity" begin

    n = 16
    L = 100.0
    nxyz = (n, n, n)
    kF = (2π / L) .* (1, 1, 1)
    Volume = L^3
    pkfn = f32_testpk()

    @testset "eltypes are preserved: T=$T" for T in (Float64, Float32)
        plan = LNG.plan_with_fftw(nxyz, T)
        @test eltype(plan) === T

        deltar = LNG.allocate_input(plan)
        @test eltype(deltar) === T

        deltak = LNG.draw_phases(plan; rng=MersenneTwister(7))
        @test eltype(deltak) === Complex{T}

        # a callable pk goes through pk_to_pkG(), which is Float64 internally;
        # the conversion must happen at the array boundary
        dk = copy(deltak)
        LNG.scale_by_pk!(dk, pkfn, 1.5, kF, Volume; rfftplan=plan)
        @test eltype(dk) === Complex{T}

        # a user-supplied Float64 pk array must not drag the field to Float64
        pk1d = pkfn.((2π / L) .* (0:(n - 1)))
        @test eltype(pk1d) === Float64
        dk = copy(deltak)
        LNG.scale_by_pk!(dk, pk1d, 1.5, kF, Volume; rfftplan=plan)
        @test eltype(dk) === Complex{T}

        pk2d = [pk1d zero(pk1d) 0.1 .* pk1d]
        dk = copy(deltak)
        LNG.scale_by_pk!(dk, pk2d, 1.5, kF, Volume; rfftplan=plan)
        @test eltype(dk) === Complex{T}

        pk3d = pkfn.([√((2π / L)^2 * (i^2 + j^2 + k^2))
                      for i in 0:(n ÷ 2), j in 0:(n - 1), k in 0:(n - 1)])
        dk = copy(deltak)
        LNG.scale_by_pk!(dk, pk3d, 1.5, kF, Volume; rfftplan=plan)
        @test eltype(dk) === Complex{T}

        dk = copy(deltak)
        @test eltype(LNG.pixel_window!(dk, nxyz; voxel_window_correction=1)) === Complex{T}
        @test eltype(LNG.calc_velocity_component!(dk, kF, 1)) === Complex{T}
        @test eltype(LNG.set_fixed_phase!(dk, 0.3)) === Complex{T}

        # back to real space, and on through the lognormal transform
        @test eltype(plan \ dk) === T
    end


    # Deterministic per-operation comparison: give both precisions the identical
    # input field and check they agree to Float32 precision.
    @testset "Float32 agrees with Float64 per operation" begin
        plan64 = LNG.plan_with_fftw(nxyz, Float64)
        plan32 = LNG.plan_with_fftw(nxyz, Float32)

        deltak64 = LNG.draw_phases(plan64; rng=MersenneTwister(11))
        deltak32 = ComplexF32.(deltak64)
        # the inputs differ only by the initial rounding
        @test reldiff(deltak64, deltak32) < 1e-6

        pk1d = pkfn.((2π / L) .* (0:(n - 1)))

        for (name, op!) in [
                "scale_by_pk!(callable)" =>
                    (dk, plan) -> LNG.scale_by_pk!(dk, pkfn, 1.5, kF, Volume; rfftplan=plan),
                "scale_by_pk!(array)" =>
                    (dk, plan) -> LNG.scale_by_pk!(dk, pk1d, 1.5, kF, Volume; rfftplan=plan),
                "pixel_window!" =>
                    (dk, plan) -> LNG.pixel_window!(dk, nxyz; voxel_window_correction=1),
                "calc_velocity_component!" =>
                    (dk, plan) -> LNG.calc_velocity_component!(dk, kF, 3),
            ]
            a = op!(copy(deltak64), plan64)
            b = op!(copy(deltak32), plan32)
            err = reldiff(a, b)
            @info "Float32 vs Float64: $name" err
            @test all(isfinite, a)
            @test all(isfinite, b)
            # A handful of Float32 roundings, plus an rfft round trip over n^3
            # points for scale_by_pk!(array). Measured at n=16:
            #   scale_by_pk!(callable)     3.4e-8
            #   scale_by_pk!(array)        1.2e-6
            #   pixel_window!              7.4e-8
            #   calc_velocity_component!   3.3e-8
            # These are the numbers a GPU comparison should be judged against:
            # a GPU running the same Float32 arithmetic cannot do better.
            @test err < 1e-4
        end
    end


    # The `deltar` seam lets two backends run on *identical* white noise, which
    # is the only way to compare fields deterministically across precisions.
    # This is the reference shape for the eventual CPU/GPU comparison.
    @testset "identical noise gives matching fields across precisions" begin
        noise64 = randn(MersenneTwister(2024), n, n, n)

        function field_from_noise(::Type{T}) where {T}
            plan = LNG.plan_with_fftw(nxyz, T)
            deltar = T.(noise64)
            deltak = LNG.draw_phases(plan; deltar)
            LNG.scale_by_pk!(deltak, pkfn, 1.5, kF, Volume; rfftplan=plan)
            deltarg = plan \ deltak
            @. deltarg = exp(deltarg * (n^3 / T(Volume)))
            return deltarg
        end

        # the seam must not consume the rng at all: same noise, same answer
        @test field_from_noise(Float64) == field_from_noise(Float64)

        a = field_from_noise(Float64)
        b = field_from_noise(Float32)
        @test eltype(a) === Float64
        @test eltype(b) === Float32
        @test all(isfinite, a)
        @test all(isfinite, b)
        err = norm(Float64.(b) .- a) / norm(a)
        @info "Float32 vs Float64: density field from identical noise" err
        @test err < 1e-4
    end


    # The precision commit narrowed scalars in these option paths, which the
    # reproducibility gate never reaches (it uses the defaults throughout).
    @testset "option paths stay in T: $label" for (label, opts) in [
            "defaults"            => (;),
            "fixed_phase"         => (; fixed_phase=true),
            "fixed_amplitude"     => (; fixed_amplitude=true),
            "phase_shift"         => (; phase_shift=π / 3),
            "sigma_psi"           => (; sigma_psi=1.0),
            "minimize_shotnoise"  => (; minimize_shotnoise=true),
            "voxel_window_corr"   => (; voxel_window_correction=1),
            "voxel_window_power"  => (; voxel_window_power=3),
        ]
        x⃗, Ψ = simulate_galaxies(3e-4, L, pkfn; nmesh=n, bias=1.5, f=1,
                                 T=Float32, opts...)
        @test eltype(x⃗) === Float32
        @test eltype(Ψ) === Float32
        @test all(isfinite, x⃗)
        @test all(isfinite, Ψ)
        @test size(x⃗, 2) > 0
    end


    @testset "velocity_assignment=$va stays in T" for va in 0:6
        x⃗, Ψ = simulate_galaxies(3e-4, L, pkfn; nmesh=n, bias=1.5, f=1,
                                 T=Float32, velocity_assignment=va,
                                 voxel_window_power=2)
        @test eltype(x⃗) === Float32
        @test eltype(Ψ) === Float32
        @test all(isfinite, x⃗)
        @test all(isfinite, Ψ)
    end


    # Requesting Float32 from a planner that cannot supply it must fail loudly
    # rather than silently returning Float64.
    @testset "legacy one-argument planner" begin
        legacy(nxyz) = LNG.plan_with_fftw(nxyz, Float64)
        @test_throws ArgumentError simulate_galaxies(3e-4, L, pkfn; nmesh=n,
                                                     rfftplanner=legacy, T=Float32)
        x⃗, Ψ = simulate_galaxies(3e-4, L, pkfn; nmesh=n, rfftplanner=legacy)
        @test eltype(x⃗) === Float64
    end

end


# vim: set sw=4 et sts=4 :
