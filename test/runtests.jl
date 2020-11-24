#!/usr/bin/env julia


module TestLogNormalGalaxies

using Test
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using DelimitedFiles
using Random


function compile_and_load()
    b = 1.8
    f = 0.71
    D = 0.82

    data = readdlm((@__DIR__)*"/rockstar_matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)

    nbar = 3e-4
    L = 2e3
    ΔL = 50.0  # buffer for RSD
    n = 16
    #Random.seed!(8143083339)

    # generate catalog
    x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_fftw)
    x⃗2, Ψ2 = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1, rfftplanner=LogNormalGalaxies.plan_with_pencilffts)
    @show size(x⃗), size(Ψ)
    @show size(x⃗2), size(Ψ2)
    return true
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
    seed = rand(UInt64)
    Random.seed!(seed)
    @show seed
    μ, v, d = create_randn(n, LogNormalGalaxies.plan_with_fftw)
    Random.seed!(seed)
    μ2, v2, d2 = create_randn(n, LogNormalGalaxies.plan_with_pencilffts)
    @test all(@. d2 - d == 0)
end


function test_pk_to_pkG()
    data = readdlm((@__DIR__)*"/rockstar_matterpower.dat", comments=true)
    pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    k, pkG = LogNormalGalaxies.pk_to_pkG(pk)
    @show pkG.(k[1:20])
    @test pkG(0) == 0
end


function main()
    test_pk_to_pkG()
    @test compile_and_load()
    test_random_phases()
end


end


TestLogNormalGalaxies.main()


# vim: set sw=4 et sts=4 :
