using Revise
using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using DelimitedFiles
using Statistics
using PythonPlot
using Random
using Test

@testset "Complimentary simulations" begin
    include("lib.jl")

    nbar = 1e-1
    L = 0.5e3
    n_sim = 256
    n_est = 256
    bias = 1.5
    f = 0.7
    fixed = false

    LLL = L .* [1,1,1]
    nnn_est = n_est .* [1,1,1]
    box_center = [0,0,0]
    vox = 1
    opts = (nbar=nbar, lmax=4, do_mu_leakage=true, subtract_shotnoise=true, voxel_window_power=vox)

    id = "vox=$vox, nsim=$n_sim, nest=$n_est"

    #keq = 2e-2
    #c = 3 * keq^4
    #a = 2e4 * 4 * keq^3
    #pk(k) = a * k / (c + k^4)
    D = 0.2
    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)


    Random.seed!(reinterpret(UInt64, time()))
    seed = rand(UInt64, 4)
    #seed = UInt64[0x01ca4eddf1fdc165, 0xd87d80f08f1e36f8, 0x0ecc4300d7f0a61c, 0xeb0ab21155f56174]
    @show seed

    Random.seed!(seed)
    @time x⃗1, Ψ1 = simulate_galaxies(nbar, L, pk; nmesh=n_sim, bias, f=true, fixed)

    Random.seed!(seed)
    @time x⃗2, Ψ2 = simulate_galaxies(nbar, L, pk; nmesh=n_sim, bias, f=true, fixed, phase_shift=π)


    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn_est, box_center; opts...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn_est, box_center; opts...)
    k1 = k1[2:end]
    k2 = k2[2:end]
    pk1 = pk1[2:end,:]
    pk2 = pk2[2:end,:]

    #figure()
    #title("Real space")
    #hlines(1/nbar, extrema(k1)..., color="0.75")
    #plot(k1, bias^2 .* pk.(k1), "k-")
    #plot(k1, pk1[:,:], "-")
    #gca().set_prop_cycle(nothing)
    #plot(k2, pk2[:,:], "--")
    #plot(k1, middle.(pk1, pk2), "k:")
    #xlim(0, 0.25)

    figure()
    title("Real space: $id")
    hlines(1, extrema(k1)..., color="0.75")
    plot(k1, pk1[:,1] ./ (bias^2 .* pk.(k1)), "-", label=L"\phi=0")
    plot(k2, pk2[:,1] ./ (bias^2 .* pk.(k1)), "-", label=L"\phi=\pi")
    plot(k2, (pk1 .+ pk2)[:,1] ./ 2 ./ (bias^2 .* pk.(k1)), "k:", label="mean")
    xlim(0, 0.25)
    ylim(0.6, 1.4)
    legend()

    #return

    #### to redshift space

    los = [0, 0, 1]
    apply_rsd!(x⃗1, Ψ1, f, los)
    apply_rsd!(x⃗2, Ψ2, f, los)
    x⃗1 = MeasurePowerSpectra.periodic_boundaries!(x⃗1, LLL, box_center)
    x⃗2 = MeasurePowerSpectra.periodic_boundaries!(x⃗2, LLL, box_center)
    k1, pk1, nmodes1 = xgals_to_pkl_planeparallel(x⃗1, LLL, nnn_est, box_center; opts...)
    k2, pk2, nmodes2 = xgals_to_pkl_planeparallel(x⃗2, LLL, nnn_est, box_center; opts...)

    β = f / bias
    pkm_kaiser = @. bias^2 * Arsd_Kaiser(β, (0:4)') * pk(k1)

    figure()
    title("Redshift space: $id")
    hlines(1/nbar, extrema(k1)..., color="0.75")
    plot(k1, pkm_kaiser[:,:], "-", lw=0.5)
    gca().set_prop_cycle(nothing)
    plot(k1, pk1[:,:], "-")
    gca().set_prop_cycle(nothing)
    plot(k2, pk2[:,:], "--")
    plot(k1, middle.(pk1, pk2), "k:")
    xlim(0, 0.25)

    figure()
    title("Redshift space: $id")
    hlines(1, extrema(k1)..., color="0.75")
    plot(k1, pk1[:,1] ./ pkm_kaiser[:,1], "-")
    plot(k2, pk2[:,1] ./ pkm_kaiser[:,1], "-")
    plot(k2, (pk1 .+ pk2)[:,1] ./ 2 ./ pkm_kaiser[:,1], "k:")
    xlim(0, 0.25)
    ylim(0.6, 1.4)
end
