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


@testset verbose=true "LogNormalGalaxies" begin

    @testset "Compile and load $rfftplanner" for rfftplanner=[LogNormalGalaxies.plan_with_fftw,LogNormalGalaxies.plan_with_pencilffts]
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
        @test typeof(x⃗) <: Array{Float64}
        @test typeof(Ψ) <: Array{Float64}
    end


    @testset "Random phases" begin
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


    @testset "pk_to_pkG(D²=$D²)" for D²=[0.1,1.0]
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


    @testset "Zero pk" begin
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


    @testset "Cutoff pk" begin
        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
        k0 = 5e-2
        pk(k) = 0.5^2 * _pk(k) * exp(-(k/k0)^2)
        k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
        @test pkG(0) == 0
    end


    @testset "draw_galaxies_with_velocities()" begin
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


    @testset "Array deepcopy" begin
        nxyz = (2, 2, 2)
        rfftplan = LogNormalGalaxies.plan_with_pencilffts(nxyz)
        x = LogNormalGalaxies.allocate_input(rfftplan)
        randn!(parent(x))
        y = deepcopy(x)
        @show typeof(x) typeof(y)
        @show x y
    end


    @testset verbose=true "Any spline" begin
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


    @testset "3D-Array pk" begin
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


    @testset "2D-Array pk" begin
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


    @testset "1D-Array pk" begin
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


    @testset "Reproducible catalog, rsd=$rsd" for rsd=[false,true]
        nbar = 1e-8
        L = 1e3
        nmesh = 32
        bias = 1.5

        keq = 2e-2
        c = 3 * keq^4
        a = 2e4 * 4 * keq^3
        pk(k) = a * k / (c + k^4)

        # Create separate random number generator, because task creation also uses
        # the global RNG, so using that depends on the task-creation scheme.
        #rng = Random.Xoshiro()  # not stable across Julia versions
        rng = StableRNG(981670238674)

        x⃗old = Float64[-335.9692397515812 -157.67807790772844 -72.68579550665544 -269.6279704155078 -491.17708179876234 -365.72719760869035 -224.59392943221292 -95.4013877730365 345.3605001013826 -226.13352716021592 117.6963824483073 93.71545127205661; -180.8600883273986 220.7798770583439 109.99973965725985 -244.36442326834518 -201.83880847661288 453.6280628916106 235.70606340167456 453.34764628135554 -356.39776839402646 -263.5108300272305 -274.49624776463884 347.296242103321; -377.20300791807426 -265.12202516834606 -156.12206587366097 -36.14651372879746 -33.719191328648435 -60.55004264254137 53.63170729565536 42.4431323030625 133.39265941194697 181.867222289673 497.4939329019023 481.6793904866439]
        if rsd
            # Ψold = Float64[-3.6046158060568043 -1.2473745889669638 -0.5267783512164292 7.559762043549412 -2.2893585427941026 -4.109193450538838 -2.391853412800177 0.3292149097252126 -1.48609726716212 2.141554349879806 -2.706566703856816 -5.4986996776576795; 1.1556700567515272 -5.386680707657309 -0.29200492354859553 0.4961305897075565 3.9457922088356314 -4.4949924227674884 0.5157226080949375 -2.6783119499320387 -3.4522859920956344 2.060164626729195 0.4898185766272014 1.494258898730236; 4.146205213131446 -1.239451327305363 -4.625577225978314 -1.5716119688975145 -9.7564603786325 5.3319551731469925 5.484297450308572 -0.551959727063615 3.453837888093186 0.48423961453566144 2.154348830736598 -1.2155629198989812]
            Ψold = Float64[-3.6045840198752472 -1.2473635893827395 -0.5267737059856432 7.559695380142889 -2.289338354798983 -4.109157214896512 -2.391832320985978 0.3292120066460913 -1.4860841624763261 2.1415354652499206 -2.706542836841505 -5.498651189087557; 1.1556598658351513 -5.386633206891577 -0.2920023485941454 0.4961262147345087 3.945757414096252 -4.494952785077113 0.5157180603557805 -2.6782883320724684 -3.4522555491870874 2.060146459809227 0.48981425731467376 1.494245722073638; 4.146168651112423 -1.2394403975899562 -4.625536436766531 -1.5715981101269614 -9.756374344326037 5.3319081549613365 5.484249088740005 -0.5519548597787105 3.4538074314997282 0.484235344419471 2.1543298332825676 -1.215552200836052]
        else
            Ψold = fill(Float64(0), size(x⃗old))
        end

        @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh, bias, f=rsd, rng, voxel_window_power=1, velocity_assignment=0, sigma_psi=0)
        x⃗ = LogNormalGalaxies.concatenate_mpi_arr(x⃗)
        Ψ = LogNormalGalaxies.concatenate_mpi_arr(Ψ)

        @show size(x⃗) size(Ψ) x⃗ x⃗old Ψ Ψold
        @show x⃗[:,2]
        @show Ψ[:,2]
        @test size(x⃗) == size(x⃗old)
        @test size(Ψ) == size(Ψold)

        for i=1:size(x⃗,2)
            @test x⃗[:,i] ≈ x⃗old[:,i]  rtol=eps(Float32(L/2))
            @test Ψ[:,i] ≈ Ψold[:,i]  rtol=eps(10f0)
        end
    end


    include("apply_rsd.jl")
    include("iterate_kspace.jl")


    ## This meant to be used more interactively:
    #include("lognormals_50sims.jl")

    # @testset "example.jl" begin
    #     include("example.jl")
    # end

    # @testset "complenetary_sims.jl" begin
    #     include("complenetary_sims.jl")
    # end
end


# vim: set sw=4 et sts=4 :
