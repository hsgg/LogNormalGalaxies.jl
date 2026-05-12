# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


#!/usr/bin/env julia

using Revise

module test_lognormal

using LogNormalGalaxies
using MySplines
using MeasurePowerSpectra
using Random
using DelimitedFiles
#using QuadGK
#using PlaneParallelRedshiftSpaceDistortions

do_plot = isinteractive()

if do_plot
    using PythonPlot
end


function Arsd_Kaiser(β, ℓ)
    if ℓ == 0
        return 1 + 2/3*β + 1/5*β^2
    elseif ℓ == 2
        return 4/3*β + 4/7*β^2
    elseif ℓ == 4
        return 8/35*β^2
    else
        return 0
    end
end


function main()
    b = 1.5
    f = 0.5
    D = 1.0

    nbar = 3e-4
    L = 1e3
    n = 64
    n_sim = 128
    LLL = [L, L, L]
    nnn = [n, n, n]
    box_center = [0,0,1e8]
    Random.seed!(8143083339)
    Random.seed!(1234567)
    alpha = 0.1

    kNy_sim = π / L * min(n, n_sim)

    voxel_window_power_sim = 1
    voxel_window_correction_sim = 0

    grid_assignment = 4
    interlacing_order = 1
    voxel_window_power = 4

    #estimator = :xgals_pp
    estimator = :Fr_ep

    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
    pkfn(k) = D^2 * _pk(k)

    ## choose
    # pk = pkfn
    kin = (2π/L) * (0:n_sim)
    pk = pkfn.(kin)

    # generate catalog
    x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n_sim, bias=b, f=(f!=0), voxel_window_power=voxel_window_power_sim, voxel_window_correction=voxel_window_correction_sim)

    # add RSD
    los = [0, 0, 1]
    Ngals = size(x⃗,2)
    for i=1:Ngals
        x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
    end

    # cut the possibly incomplete (due to RSD) boundaries
    sel = @. -L/2 <= x⃗[1,:] <= L/2
    @. sel &= -L/2 <= x⃗[2,:] <= L/2
    @. sel &= -L/2 <= x⃗[3,:] <= L/2
    x⃗ = x⃗[:,sel] .+ box_center
    x⃗ = MeasurePowerSpectra.periodic_boundaries!(x⃗, LLL, box_center)

    # Measure power spectrum
    if estimator == :xgals_pp
        km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn, box_center;
            subtract_shotnoise=false, grid_assignment, voxel_window_power)
    elseif estimator == :Fr_ep
        Ngalaxies = size(x⃗, 2)
        Nrandoms = round(Int, nbar * prod(LLL) / alpha)
        #nbar_actual = Ngalaxies / prod(LLL)
        #alpha_actual = Ngalaxies / Nrandoms
        nbar_actual = nbar
        alpha_actual = alpha
        xgals = x⃗
        xrand = LLL .* (rand(3, Nrandoms) .- 1 // 2) .+ box_center
        ng = calc_number_density(xgals, LLL, nnn, box_center; grid_assignment, interlacing_order)
        nr = calc_number_density(xrand, LLL, nnn, box_center; grid_assignment, interlacing_order)
        A = calc_pkl_normalization(ng, nr; alpha1=alpha_actual)
        dVol = prod(LLL ./ nnn)
        @show Ngalaxies Nrandoms
        @show sum(ng)*dVol
        @show sum(nr)*dVol
        @show nbar_actual alpha_actual
        Fr = calc_deltar(ng, nr, nbar_actual, nbar_actual / alpha_actual)
        km, pkm, nmodes = Fr_to_pkl_endpoint(Fr, Fr, LLL, box_center; ell1=0:4, ell2=0, do_mu_leakage=true, voxel_window_power)
        pkm .*= nbar^2 / (A * prod(LLL))
        pkm[:,2:2:end] .*= -im
    else
        error("Estimator $estimator not defined.")
    end

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pkfn(km)
    pkm_kaiser[:,1] .+= 1 / nbar


    if do_plot
        n = 0

        plotclose("all")  # close previous plots
        figure()
        hlines(1/nbar, extrema(km)..., color="0.75")
        axvline(kNy_sim, color="0.75")
        #plot(km, b^2 .* km.^n.*(pkfn.(km) .+ 1/nbar), "k", label="input \$k^{$n}\\,P(k)\$")
        for m=1:size(pkm,2)
            plot(km, km.^n.*pkm[:,m], "C$(m-1)-", label="\$k^{$n}\\,P_{$(m-1)}(k)\$")
            plot(km, km.^n.*pkm_kaiser[:,m], "C$(m-1)--")
        end
        xlabel("\$k\$")
        ylabel("\$k\\,P_\\ell(k)\$")
        #xscale("log")
        #xlim(right=0.6)
        legend(fontsize="small")
        savefig((@__DIR__)*"/lognormal.pdf")
    end
end


end


test_lognormal.main()


# vim: set sw=4 et sts=4 :
