# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


#!/usr/bin/env julia

using Revise

using Test
using LogNormalGalaxies
using MySplines
using DelimitedFiles
using Random
using StableRNGs
using BenchmarkTools
using PkSpectra


# Run a subset of the suite by name, e.g.
#     julia --project -e 'using Pkg; Pkg.test(test_args=["reproducibility"])'
# With no arguments the whole suite runs.
selected(name) = isempty(ARGS) || name in ARGS

include("testutils.jl")


@testset verbose=true "LogNormalGalaxies" begin

    selected("reproducibility") && include("reproducibility.jl")
    selected("float32") && include("float32.jl")

    # GPU tests are opt-in. CI is ubuntu-latest/x64, and GitHub's macOS runners
    # have no usable Metal GPU either, so this only ever runs on a developer
    # machine. `using` cannot appear inside an `if`, hence the guarded include.
    if selected("metal") && Sys.isapple() && Sys.ARCH == :aarch64 &&
            get(ENV, "LNG_TEST_GPU", "0") == "1"
        include("metal.jl")
    end

    selected("compile") && @testset "Compile and load $rfftplanner" for rfftplanner=[LogNormalGalaxies.plan_with_fftw,LogNormalGalaxies.plan_with_pencilffts]
        @show rfftplanner
        if Sys.ARCH == :aarch64 && rfftplanner == LogNormalGalaxies.plan_with_pencilffts
            @test_skip "Skipping PencilFFTs on ARM64"
            continue
        end

        bias = 1.8
        f = 0.71
        D = 0.82

        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
        pk(k) = D^2 * _pk(k)

        nbar = 3e-4
        L = 1e1
        ΔL = 50.0  # buffer for RSD
        n = 32
        #Random.seed!(8143083339)

        # generate catalog
        @time x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias, f=1, rfftplanner)
        @show size(x⃗), size(Ψ)
        # Float64 is still the default; see float32.jl for the general case.
        @test typeof(x⃗) <: Array{Float64}
        @test typeof(Ψ) <: Array{Float64}
        @test typeof(x⃗) <: Array{<:AbstractFloat}
        @test typeof(Ψ) <: Array{<:AbstractFloat}
    end


    selected("phases") && @testset "Random phases" begin
        function create_randn(n, rfftplanner)
            rfftplan = rfftplanner([n,n,n])
            deltar = LogNormalGalaxies.allocate_input(rfftplan)
            randn!(parent(deltar))
            d = parent(deltar)[:]
            μ = LogNormalGalaxies.mean(d)
            v = LogNormalGalaxies.var(d)
            return μ, v, d
        end

        n = 64
        seed = rand(UInt128)
        Random.seed!(seed)
        @show seed
        μ, v, d = create_randn(n, LogNormalGalaxies.plan_with_fftw)
        Random.seed!(seed)
        μ2, v2, d2 = create_randn(n, LogNormalGalaxies.plan_with_pencilffts)
        @test all(@. d2 - d == 0)
    end


    # The k-space operations are one broadcast for every backend now, and a
    # PencilArray broadcasts in *memory* order, so `broadcast_dim()` has to place
    # each factor at its permuted axis. Nothing else checks that: run identical
    # noise through both backends and demand the same field. Single-rank -- all
    # `Pkg.test` gives us -- but a single rank already carries a
    # Permutation(3,2,1), which is the part that needs checking. Multi-rank
    # local ranges remain untested. (Hence also no ARM64 skip here.)
    selected("pencil_kspace") && @testset "PencilFFTs matches FFTW in k-space" begin
        begin
            n = 16
            nxyz = (n, n, n)
            L = 100.0
            kF = (2π / L) .* (1, 1, 1)
            Volume = L^3

            keq = 2e-2
            pkfn(k) = 2e4 * 4 * keq^3 * k / (3 * keq^4 + k^4)
            pk1d = pkfn.((2π / L) .* (0:(n - 1)))

            noise = randn(StableRNG(4711), nxyz...)

            # A host array in logical index order, for both array types.
            # `CartesianIndices(::PencilArray)` yields logical indices but
            # *iterates* in memory order, so a comprehension over it would come
            # out permuted; hence the explicit loops. Single-rank only.
            to_plain(u) = [u[i,j,k] for i in axes(u,1), j in axes(u,2), k in axes(u,3)]

            function kspace_field(rfftplanner, op!)
                rfftplan = rfftplanner(nxyz)
                deltar = LogNormalGalaxies.allocate_input(rfftplan)
                for I in CartesianIndices(deltar)
                    deltar[I] = noise[I]
                end
                deltak = LogNormalGalaxies.draw_phases(rfftplan; deltar)
                return to_plain(op!(deltak, rfftplan))
            end

            @testset "$name" for (name, op!) in [
                    "draw_phases" =>
                        (dk, p) -> dk,
                    "pixel_window!" =>
                        (dk, p) -> LogNormalGalaxies.pixel_window!(dk, nxyz; voxel_window_correction=1),
                    "calc_velocity_component!" =>
                        (dk, p) -> LogNormalGalaxies.calc_velocity_component!(dk, kF, 2),
                    "scale_by_pk!(callable)" =>
                        (dk, p) -> LogNormalGalaxies.scale_by_pk!(dk, pkfn, 1.5, kF, Volume; rfftplan=p),
                    "scale_by_pk!(array)" =>
                        (dk, p) -> LogNormalGalaxies.scale_by_pk!(dk, pk1d, 1.5, kF, Volume; rfftplan=p),
                ]
                a = kspace_field(LogNormalGalaxies.plan_with_fftw, op!)
                b = kspace_field(LogNormalGalaxies.plan_with_pencilffts, op!)
                @test size(a) == size(b)
                @test all(isfinite, a)
                # both call FFTW on the same noise: reassociation apart only
                @test a ≈ b rtol=1e-10
            end
        end
    end


    selected("pk_to_pkG") && @testset "pk_to_pkG(D²=$D²)" for D²=[0.1,1.0]
        @show D²
        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        println("data read")
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
        println("data splined")
        pk(k) = D² * _pk(k)
        println("D^2 multiplied")
        k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
        @show D²,pk.([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        @show D²,pkG.([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        @test pkG(0) == 0
    end


    selected("zero_pk") && @testset "Zero pk" begin
        pk(k) = 0.0
        #k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
        k = 10.0 .^ (-3:0.01:0)
        pkG = LogNormalGalaxies.Spline1D(k, pk.(k), extrapolation=MySplines.powerlaw)
        @test pkG(0) == 0
        @test pkG(0.0) == 0
        @test pkG(0.1) == 0

        nbar = 3e-4
        L = 100.0
        n = 64
        b = 1.0
        f = 1
        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    end


    selected("cutoff_pk") && @testset "Cutoff pk" begin
        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
        k0 = 5e-2
        pk(k) = 0.5^2 * _pk(k) * exp(-(k/k0)^2)
        k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
        @test pkG(0) == 0
    end


    selected("draw_galaxies") && @testset "draw_galaxies_with_velocities()" begin
        # The function 'draw_galaxies_with_velocities()' is a performance bottleneck.
        nnn = 128, 128, 128
        deltar = randn(nnn...)
        vx = randn(nnn...) / 10
        vy = randn(nnn...) / 10
        vz = randn(nnn...) / 10
        Ngalaxies = 100_000
        Δx = [1.0, 1.0, 1.0]
        Navg = Ngalaxies / prod(nnn)
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        #@btime LogNormalGalaxies.draw_galaxies_with_velocities($deltar, $vx, $vy, $vz, $Navg, $Ngalaxies, $Δx, Val(true), Val(2), Val(6))
    end


    selected("deepcopy") && @testset "Array deepcopy" begin
        nxyz = (2, 2, 2)
        rfftplan = LogNormalGalaxies.plan_with_pencilffts(nxyz)
        x = LogNormalGalaxies.allocate_input(rfftplan)
        randn!(parent(x))
        y = deepcopy(x)
        @show typeof(x) typeof(y)
        @show x y
    end


    selected("any_spline") && @testset verbose=true "Any spline" begin
        println("Test any typed spline:")
        # Someone may give other data types than Float64 to the module. Let's be
        # able to handle that.
        data = readdlm((@__DIR__)*"/matterpower.dat")
        pk0 = Spline1D(data[2:end,1], data[2:end,2], extrapolation=MySplines.powerlaw)
        pk1 = PkDefault()

        # Test any callable
        struct SomeTypeSomeWhere end
        (::SomeTypeSomeWhere)(k) = pk1(k)
        pk2 = SomeTypeSomeWhere()

        @testset "$(typeof(pk))" for pk in [pk0, pk1, pk2]
            @show typeof(data)
            @show typeof(pk)
            @show pk(0.01)

            k, pkG = LogNormalGalaxies.pk_to_pkG(pk)

            nbar = 3e-4
            L = 10.0
            n = 32
            b = 1.5
            f = 1
            x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
            x⃗, Ψ = simulate_galaxies(nbar, [L,L,L], pk; nmesh=[n,n,n], bias=b, f=true, rfftplanner=LogNormalGalaxies.plan_with_fftw)
        end
    end


    selected("array_pk") && @testset "3D-Array pk" begin
        println("Test array typed pk:")
        nbar = 3e-4
        L = 100.0
        n = 64
        b = 1.5
        f = 1

        pk = rand(n ÷ 2 + 1, n, n)
        @show typeof(pk)

        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=false, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    end


    selected("array_pk") && @testset "2D-Array pk" begin
        println("Test array typed pk:")
        nbar = 3e-4
        L = 100.0
        n = 64
        b = 1.5
        f = 1
        lmax = 4

        pk = rand(n, lmax + 1)
        @show typeof(pk)

        x⃗, Ψ = simulate_galaxies(nbar, [L,L,L], pk; nmesh=[n,n,n], bias=b, f=false, rfftplanner=LogNormalGalaxies.plan_with_fftw)
        x⃗, Ψ = simulate_galaxies(nbar, [L,L,L], pk; nmesh=[n,n,n], bias=b, f=true, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    end


    selected("array_pk") && @testset "1D-Array pk" begin
        println("Test array typed pk:")
        nbar = 3e-4
        L = 100.0
        n = 64
        b = 1.5
        f = 1
        lmax = 4

        pk = rand(n)
        @show typeof(pk)

        x⃗, Ψ = simulate_galaxies(nbar, [L,L,L], pk; nmesh=[n,n,n], bias=b, f=true, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    end


    selected("apply_rsd") && include("apply_rsd.jl")
    selected("iterate_kspace") && include("iterate_kspace.jl")
    selected("pencil_arraytype") && include("pencil_arraytype.jl")


    ## This meant to be used more interactively:
    #include("lognormals_50sims.jl")

    selected("example") && @testset "example.jl" begin
        include("example.jl")
    end

    # @testset "complenetary_sims.jl" begin
    #     include("complenetary_sims.jl")
    # end
end


# vim: set sw=4 et sts=4 :
