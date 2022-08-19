#!/usr/bin/env julia


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
    x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    x⃗2, Ψ2 = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_pencilffts)
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



function main()
    @testset "unit tests" begin
        test_pk_to_pkG(0.1)
        test_pk_to_pkG(1.0)
        compile_and_load()
        test_random_phases()
        test_zero_pk()
        test_cutoff_pk()
        test_draw_galaxies_with_velocities()
        test_array_deepcopy()
        test_anyspline()
    end
end


end


using Test
@testset "LogNormalGalaxies" begin
    TestLogNormalGalaxies.main()
    include("lognormals_50sims.jl")
end


# vim: set sw=4 et sts=4 :
