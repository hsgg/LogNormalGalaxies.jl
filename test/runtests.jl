#!/usr/bin/env julia

using Revise

using Test
using LogNormalGalaxies
using Splines
using DelimitedFiles
using Random
using BenchmarkTools


@testset "LogNormalGalaxies" begin

    @testset "Compile and load" for rfftplanner=[LogNormalGalaxies.plan_with_fftw,LogNormalGalaxies.plan_with_pencilffts]
        @show rfftplanner
        if Sys.ARCH == :aarch64 && rfftplanner == LogNormalGalaxies.plan_with_pencilffts
            @test_skip "Skipping PencilFFTs on ARM64"
            continue
        end

        bias = 1.8
        f = 0.71
        D = 0.82

        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
        pk(k) = D^2 * _pk(k)

        nbar = 3e-4
        L = 2e3
        ΔL = 50.0  # buffer for RSD
        n = 64
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
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
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
        pkG = LogNormalGalaxies.Spline1D(k, pk.(k), extrapolation=Splines.powerlaw)
        @test pkG(0) == 0
        @test pkG(0.0) == 0
        @test pkG(0.1) == 0

        nbar = 3e-4
        L = 1000.0
        n = 64
        b = 1.0
        f = 1
        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    end


    @testset "Cutoff pk" begin
        data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
        _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
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
        Ngalaxies = 1_000_000
        Δx = [1.0, 1.0, 1.0]
        Navg = Ngalaxies / prod(nnn)
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Navg, Ngalaxies, Δx, Val(true), Val(2), Val(6))
        @btime LogNormalGalaxies.draw_galaxies_with_velocities($deltar, $vx, $vy, $vz, $Navg, $Ngalaxies, $Δx, Val(true), Val(2), Val(6))
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


    @testset "Any spline" begin
        println("Test any typed spline:")
        # Someone may give other data types than Float64 to the module. Let's be
        # able to handle that.
        data = readdlm((@__DIR__)*"/matterpower.dat")
        pk = Spline1D(data[2:end,1], data[2:end,2], extrapolation=Splines.powerlaw)
        @show typeof(data)
        @show typeof(pk)
        @show pk(0.01)

        k, pkG = LogNormalGalaxies.pk_to_pkG(pk)

        nbar = 3e-4
        L = 1000.0
        n = 64
        b = 1.0
        f = 1
        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
        x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=false, rfftplanner=LogNormalGalaxies.plan_with_fftw)
        x⃗, Ψ = simulate_galaxies(nbar, [L,L,L], pk; nmesh=[n,n,n], bias=b, f=false, rfftplanner=LogNormalGalaxies.plan_with_fftw)
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
        rng = Random.Xoshiro()

        Random.seed!(rng, 981670238674)
        #x⃗old = Float64[90.12323 -239.76099 -111.01846 207.76068 169.9992 -413.45624 -98.743225 321.521 -409.5943 -222.11697; 379.68872 -1.3737488 446.7375 -203.58994 206.53577 405.7047 444.58038 90.98395 317.6936 -431.44104; -494.18585 -379.7971 -395.80817 -359.86798 -358.05563 -337.06317 -336.2911 -165.63074 19.286865 195.81134]  # for fixed sim
        x⃗old = Float64[90.12321974755253 -239.76098618443064 -111.01847650944575 207.76069325248034 169.99919833094225 -413.456244563397 -98.74321484548818 321.52102394899373 -345.50796415332684 249.1035805306675 305.01836468482884 -472.1169598142895 -51.13714531768477 57.278828083530925; 379.6887456088716 -1.3737516498867421 446.73747903348874 -203.58994395380898 206.5357878046358 405.7046885864876 444.58039946521956 90.98397469207168 37.69200862629327 -43.457955013855496 192.6431169864627 -431.44104424115875 -14.038089263114216 -288.42223127474676; -494.1858486717646 -379.79708480125373 -395.8081783007257 -359.8679723383981 -358.0556302613436 -337.06317757383476 -336.29110332091375 -165.6307367285409 -47.21860542409644 -0.8617971996176266 53.84224331280575 195.81136157578237 384.8446145325869 483.57593037360164]
        if rsd
            #Ψold = Float64[0.4778329 3.219249 -1.6333185 -0.42805248 3.4687948 -4.5293393 -2.486782 0.5010775 0.012112802 1.8666419; -4.4531307 -1.2860316 2.3900015 -1.4083495 4.585397 10.149197 1.5276932 -4.7637467 -3.0302913 1.0205473; -2.4395022 5.5024977 0.15911977 0.015944915 -6.7924957 -2.0523155 -0.005055556 -2.3137286 3.6153123 -2.6358268]  # for fixed sim
            Ψold = Float64[1.2522930214209542 -0.2960562830024809 -2.746209419954654 -0.5513976412953523 3.4090224112102425 -2.493058374495547 -2.6393046195469534 -0.6056407939672763 -1.080296103484205 1.6303296888341368 -0.0375084014677467 -4.086029645875958 -1.5520402862679314 1.0967066933548097; -4.354852411351535 -0.3564794594056764 2.6398282378002 -2.0724864726639165 2.7949413805516095 11.094013933458307 4.123073665421489 -3.436828957799673 1.8685814410082595 2.887465062807249 -5.712741790729609 4.992010387109773 -2.983512126663954 3.4143316383234463; -3.609848397661667 6.556227673252665 -2.5999433652556467 0.24728764621536925 -4.713890271168711 -1.9885979698229455 1.4974347925110034 -5.3979345492974 -3.7711720179248105 4.9617116546234366 2.9718511217257397 -0.3783596863135963 -2.257422781679364 -4.195788310631852]
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


    ## This meant to be used more interactively:
    #include("lognormals_50sims.jl")
end


# vim: set sw=4 et sts=4 :
