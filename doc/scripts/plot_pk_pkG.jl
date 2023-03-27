#!/usr/bin/env julia

using Revise

# load current LogNormalGalaxies version:
using Pkg
Pkg.activate((@__DIR__)*"/../..")
using LogNormalGalaxies

# load other dependencies:
Pkg.activate(@__DIR__)
using PyPlot
using DelimitedFiles
using Splines


function plot_pk_pkG()
    #data = readdlm((@__DIR__)*"/../../test/matterpower.dat", comments=true)
    data = readdlm((@__DIR__)*"/../../test/matterpower_zeff=0.38.dat", comments=true)
    pkfn = Spline1D(data[:,1], data[:,2], extrapolation=Splines.powerlaw)

    k = 10.0 .^ (-4:0.01:0)
    #DD = [0.1, 0.2, 0.4, 0.8]
    DD = [1.0, 0.5, 0.25, 0.125]
    pk = fill(NaN, length(k), length(DD))
    pkG = fill(NaN, length(k), length(DD))
    lab_delta = []
    lab_G = []
    for i=1:length(DD)
        D = DD[i]
        kG, pkGD = LogNormalGalaxies.pk_to_pkG(k -> D^2 * pkfn(k))
        pk[:,i] .= D^2 .* pkfn.(k)
        pkG[:,i] .= pkGD.(k)
        push!(lab_delta, L"P_{\delta}")
        push!(lab_G, "\$P_G\$ with \$D=$D\$")
    end

    figure()
    loglog()
    plot(k, pk, "--", c="0.75", label=lab_delta)
    plot(k, pkG, label=lab_G)
    legend(ncols=2)
    xlabel(L"$k$ in [$h$/Mpc]")
    ylabel(L"$P(k)$ in [Mpc/$h$]$^3$")
    mkpath((@__DIR__)*"/../figs/")
    savefig((@__DIR__)*"/../figs/pk_pkG.pdf")
    println("Saved '$((@__DIR__)*"/../figs/pk_pkG.pdf'.")")
end


plot_pk_pkG()
