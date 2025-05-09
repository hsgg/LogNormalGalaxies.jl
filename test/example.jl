# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


#!/usr/bin/env julia

using Revise

module test_lognormal

using LogNormalGalaxies
using MySplines
using MeasurePowerSpectra
using PythonPlot
using Random
using DelimitedFiles
#using QuadGK
#using PlaneParallelRedshiftSpaceDistortions


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
    b = 1.8
    f = 0.0
    D = 1.0

    nbar = 3e-4
    L = 2e3
    n = 256
    LLL = [L, L, L]
    nnn = [n, n, n]
    box_center = [0,0,0]
    #Random.seed!(8143083339)

    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=MySplines.powerlaw)
    pkfn(k) = D^2 * _pk(k)

    ## choose
    # pk = pkfn
    kin = (2π/L) * (0:n)
    pk = pkfn.(kin)

    # generate catalog
    x⃗, Ψ = simulate_galaxies(nbar, L, pk; nmesh=n, bias=b, f=(f!=0), voxel_window_power=2)

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
    x⃗ = x⃗[:,sel]
    x⃗ = MeasurePowerSpectra.periodic_boundaries!(x⃗, LLL, box_center)
    km, pkm, nmodes = xgals_to_pkl_planeparallel(x⃗, LLL, nnn, box_center; voxel_window_power=3)

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pkfn(km)

    n = 0

    # plot
    plotclose("all")  # close previous plots
    figure()
    plot(km, b^2 .* km.^n.*pkfn.(km), "k", label="input \$k^{$n}\\,P(k)\$")
    for m=1:size(pkm,2)
        plot(km, km.^n.*pkm[:,m], "C$(m-1)-", label="\$k^{$n}\\,P_{$(m-1)}(k)\$")
        plot(km, km.^n.*pkm_kaiser[:,m], "C$(m-1)--")
    end
    xlabel(L"k")
    ylabel(L"k\,P_\ell(k)")
    xscale("log")
    xlim(right=0.6)
    legend(fontsize="small")
    savefig((@__DIR__)*"/lognormal.pdf")
end


end


test_lognormal.main()


# vim: set sw=4 et sts=4 :
