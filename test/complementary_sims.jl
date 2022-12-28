using Revise
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using DelimitedFiles
using Statistics
using PyPlot
using Test

@testset "Complimentary simulations" begin
    nbar = 1e-4
    L = 1e3
    nmesh = 64
    bias = 1.5
    f = 0.7

    LLL = L .* [1,1,1]
    nnn = nmesh .* [1,1,1]
    box_center = [0,0,0]
    opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=1)

    #keq = 2e-2
    #c = 3 * keq^4
    #a = 2e4 * 4 * keq^3
    #pk(k) = a * k / (c + k^4)
    D = 0.82
    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)




    Random.seed!(reinterpret(UInt64, time()))
    seed = rand(UInt64, 4)
    @show seed
    Random.seed!(seed)

    @time (x⃗1, Ψ1), (x⃗2, Ψ2) = simulate_galaxies(nbar, L, pk; nmesh, bias, f=true, extra_phases=π)
    #@time x⃗1, Ψ1 = simulate_galaxies(nbar, L, pk; nmesh, bias, f=true)
    #@time x⃗2, Ψ2 = simulate_galaxies(nbar, L, pk; nmesh, bias, f=true)


    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn, box_center; opts...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn, box_center; opts...)

    figure()
    title("Real space")
    plot(k1, bias^2 .* pk.(k1), "k-")
    plot(k1, pk1[:,:], "-")
    gca().set_prop_cycle(nothing)
    plot(k2, pk2[:,:], "--")
    plot(k1, middle.(pk1, pk2), "k:")
    xlim(0, 0.15)


    function add_RSD!(x⃗, Ψ, f, los)
        Ngals = size(x⃗,2)
        for i=1:Ngals
            x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
        end
        return x⃗
    end
    los = [0, 0, 1]
    add_RSD!(x⃗1, Ψ1, f, los)
    add_RSD!(x⃗2, Ψ2, f, los)
    x⃗1 = MeasurePowerSpectra.periodic_boundaries(x⃗1, LLL, box_center)
    x⃗2 = MeasurePowerSpectra.periodic_boundaries(x⃗2, LLL, box_center)
    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn, box_center; opts...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn, box_center; opts...)


    figure()
    title("Redshift space")
    plot(k1, pk1, "-")
    gca().set_prop_cycle(nothing)
    plot(k2, pk2, "--")
end
