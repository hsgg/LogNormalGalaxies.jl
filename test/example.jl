#!/usr/bin/env julia

using Revise

module test_lognormal

using LogNormalGalaxies
using LogNormalGalaxies.Splines
using MeasurePowerSpectra
using PyPlot
using Random
using DelimitedFiles
using QuadGK
using PlaneParallelRedshiftSpaceDistortions


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
    f = 0.71
    D = 0.82

    data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    _pk = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)
    pk(k) = D^2 * _pk(k)

    nbar = 3e-4
    L = 2e3
    ΔL = 50.0  # buffer for RSD
    n = 512
    Random.seed!(8143083339)

    # generate catalog
    x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1)

    # add RSD
    los = [0.0, 0.0, 1.0]
    Ngals = size(x⃗,2)
    for i=1:Ngals
        x⃗[:,i] .+= f * (Ψ[:,i]' * los) * los
    end

    # cut the possibly incomplete (due to RSD) boundaries
    sel = @. -L/2 <= x⃗[1,:] <= L/2
    @. sel &= -L/2 <= x⃗[2,:] <= L/2
    @. sel &= -L/2 <= x⃗[3,:] <= L/2
    x⃗ = x⃗[:,sel]

    # measure multipoles
    km, pkm, Mlm, Ngalaxies = x⃗gals_to_pkm([L,L,L], [n,n,n], x⃗; lmax=4)

    # theory
    β = f / b
    pkm_kaiser = @. b^2 * Arsd_Kaiser(β, (0:4)') * pk(km)

    # plot
    close("all")  # close previous plots
    figure()
    plot(km, b^2 .* km.*pk.(km), "k", label="input \$k\\,P(k)\$")
    for m=1:size(pkm,2)
        plot(km, km.*pkm[:,m], "C$(m-1)-", label="\$k\\,P_{$(m-1)}(k)\$")
        plot(km, km.*pkm_kaiser[:,m], "C$(m-1)--")
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
