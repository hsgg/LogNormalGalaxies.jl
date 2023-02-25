using LogNormalGalaxies
using PyPlot
using Test
using DelimitedFiles
using Splines


@testset "pk_to_pkG()" begin
    #data = readdlm((@__DIR__)*"/matterpower.dat", comments=true)
    #in_k = data[:,1]
    #in_pk = data[:,2]

    in_k = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/kh_camb_z_eff=0.38.csv")[:]
    in_pk = readdlm(homedir() * "/MeasurePowerSpectra.jl/inputs/matter_power_spectrum_pk_camb_z_eff=0.38.csv")[:]

    pkfn = Spline1D(in_k, in_pk, extrapolation=Splines.powerlaw)

    k = 10.0 .^ (-4:0.01:0)
    DD = [0.1, 0.2, 0.4, 0.8, 1.6]
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
    xlabel(L"k")
    ylabel(L"P(k)")
end
