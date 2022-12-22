#!/usr/bin/env julia

using Revise

module TestLogNormalGalaxies

using Test
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using DelimitedFiles
using Random
using BenchmarkTools


function compile_and_load()
    b = 1.8
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
    @time x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    @time x⃗2, Ψ2 = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_pencilffts)
    @show size(x⃗), size(Ψ)
    @show size(x⃗2), size(Ψ2)
    @test typeof(x⃗) <: Array{Float32}
    @test typeof(Ψ) <: Array{Float32}
    @test typeof(x⃗2) <: Array{Float32}
    @test typeof(Ψ2) <: Array{Float32}
end


function create_randn(n, rfftplanner)
    rfftplan = rfftplanner([n,n,n])
    deltar = LogNormalGalaxies.allocate_input(rfftplan)
    randn!(parent(deltar))
    d = parent(deltar)[:]
    μ = LogNormalGalaxies.mean(d)
    v = LogNormalGalaxies.var(d)
    return μ, v, d
end


function test_random_phases()
    # test random phases
    n = 64
    seed = rand(UInt128)
    Random.seed!(seed)
    @show seed
    μ, v, d = create_randn(n, LogNormalGalaxies.plan_with_fftw)
    Random.seed!(seed)
    μ2, v2, d2 = create_randn(n, LogNormalGalaxies.plan_with_pencilffts)
    @test all(@. d2 - d == 0)
end


function test_pk_to_pkG(D²)
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


function test_zero_pk()
    pk(k) = 0.0
    #k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
    k = 10.0 .^ (-3:0.01:0)
    pkG = LogNormalGalaxies.Spline1D(k, pk.(k), extrapolation=LogNormalGalaxies.Splines.powerlaw)
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


function test_cutoff_pk()
    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    k0 = 5e-2
    pk(k) = 0.5^2 * _pk(k) * exp(-(k/k0)^2)
    k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
    @test pkG(0) == 0
end


function test_draw_galaxies_with_velocities()
    # The function 'draw_galaxies_with_velocities()' is a performance bottleneck.
    nnn = 128, 128, 128
    deltar = randn(nnn...)
    vx = randn(nnn...) / 10
    vy = randn(nnn...) / 10
    vz = randn(nnn...) / 10
    Ngalaxies = 1_000_000
    Δx = [1.0, 1.0, 1.0]
    @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx)
    @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx)
    @time xyzv = LogNormalGalaxies.draw_galaxies_with_velocities(deltar, vx, vy, vz, Ngalaxies, Δx)
    @btime LogNormalGalaxies.draw_galaxies_with_velocities($deltar, $vx, $vy, $vz, $Ngalaxies, $Δx)
end


function test_array_deepcopy()
    nxyz = (2, 2, 2)
    rfftplan = LogNormalGalaxies.plan_with_pencilffts(nxyz)
    x = LogNormalGalaxies.allocate_input(rfftplan)
    randn!(parent(x))
    y = deepcopy(x)
    @show typeof(x) typeof(y)
    @show x y
end


function test_anyspline()
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
end


function test_reproducible_catalog(; rsd=true)
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
    x⃗old = Float32[90.12323 -239.76099 -111.01846 207.76068 169.9992 -413.45624 -98.743225 321.521 -409.5943 -222.11697; 379.68872 -1.3737488 446.7375 -203.58994 206.53577 405.7047 444.58038 90.98395 317.6936 -431.44104; -494.18585 -379.7971 -395.80817 -359.86798 -358.05563 -337.06317 -336.2911 -165.63074 19.286865 195.81134]
    if rsd
        Ψold = Float32[0.4778329 3.219249 -1.6333185 -0.42805248 3.4687948 -4.5293393 -2.486782 0.5010775 0.012112802 1.8666419; -4.4531307 -1.2860316 2.3900015 -1.4083495 4.585397 10.149197 1.5276932 -4.7637467 -3.0302913 1.0205473; -2.4395022 5.5024977 0.15911977 0.015944915 -6.7924957 -2.0523155 -0.005055556 -2.3137286 3.6153123 -2.6358268]
    else
        Ψold = fill(Float32(0), size(x⃗old))
    end

    @time x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh, bias, f=rsd, rng)
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


function main()
    @testset "unit tests" begin
        test_pk_to_pkG(0.1)
        test_pk_to_pkG(1.0)
        test_random_phases()
        test_zero_pk()
        test_cutoff_pk()
        test_draw_galaxies_with_velocities()
        test_array_deepcopy()
        test_anyspline()
        compile_and_load()
        test_reproducible_catalog(; rsd=false)
        test_reproducible_catalog(; rsd=true)
    end
end


end


using Test
@testset "LogNormalGalaxies" begin
    TestLogNormalGalaxies.main()
    include("lognormals_50sims.jl")
end


# vim: set sw=4 et sts=4 :
