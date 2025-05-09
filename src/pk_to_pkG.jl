# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# In this file we put functions related to the transform from the power
# spectrum pk of the density field δ to the power spectrum pkG of the lognormal
# density field G=ln(1+\delta)-μ.


#using PythonPlot


function xicalc00_quadosc(fn, r)
    # Note: The default series acceleration, the Wynn eps algorithm, somehow
    # chokes on this.
    I,E = quadosc(k -> k^2 * fn(k) * j0(k*r), 0, Inf, n->π*n/r,
        accelerator=QuadOsc.accel_cohen_villegas_zagier)
    #I,E = quadgk(k -> k^2 * fn(k) * j0(k*r), 0, Inf)
    I,E = (I,E) ./ (2 * π^2)
    #@show r,I
    return I
end


#################### calculate P_G(k) ###########################
function pk_to_pkG(pkfn)
    ## Notes: xicalc is fast here, quadosc is too slow for large r. However,
    ## xicalc has some noise at large r. Therefore, later we need to use
    ## quadosc as it is more robust in dealing with that noise.

    r1, xi1 = xicalc(pkfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-5, N=2^15, q=2.0)
    #r2, xi2 = xicalc(pkfn, 0, 0; kmin=1e-25, kmax=1e25, r0=1e-25, N=4096, q=2.0)
    #r3 = 10.0 .^ range(-4, 3.7, length=1000)
    #xi3 = xicalc00_quadosc.(pkfn, r3)
    #@show "xicalc finished"

    r, xi = r1, xi1
    #r, xi = r3, xi3

    #@show r[1],xi[1]
    #@show r[end],xi[end]

    #close("all")

    #figure()
    #plot(r1,xi1, "b", label=L"\xi(r)")
    #plot(r1,-xi1, "b--")
    #plot(r3,xi3, "g", label=L"\xi(r)")
    #plot(r3,-xi3, "g--")
    #xscale("log")
    #yscale("log")
    #legend()

    sel = @. 1e-5 <= r <= 1e4
    r = r[sel]
    xi = xi[sel]

    xiG = @. log1p(xi)

    # ensure that it goes towards zero
    #@show r[1],xiG[1:2]
    #@show r[end],xiG[end-4:end]
    while abs(xiG[end]) > abs(xiG[end-1])
        #@show r[end],xiG[end]
        r = r[1:end-1]
        xiG = xiG[1:end-1]
    end
    #@show r[1],xiG[1:2]
    #@show r[end],xiG[end-4:end]

    xiGfn = Spline1D(r, xiG, extrapolation=MySplines.powerlaw)
    #xiG2 = @. log1p(xi2)
    #xiG3 = @. log1p(xi3)
    #xiG3fn = Spline1D(r3, xiG3, extrapolation=MySplines.powerlaw)

    #figure()
    ##plot(r,xi, "k", label=L"\xi(r)")
    ##plot(r,-xi, "k--")
    ##plot(r,xiG, "b", label=L"\xi_G(r)")
    ##plot(r,-xiG, "b--")
    #plot(r,r.^3 .* xi, "k", label=L"r^3 \xi(r)")
    #plot(r,-r.^3 .* xi, "k--")
    ##plot(r2,r2.^3 .* xi2, "0.75")
    ##plot(r2,-r2.^3 .* xi2, c="0.75", ls="--")
    #plot(r,r.^3 .* xiG, "b", label=L"r^3 \xi_G(r)")
    #plot(r,-r.^3 .* xiG, "b--")
    #plot(r3,r3.^3 .* xiG3, "g")
    #plot(r3,-r3.^3 .* xiG3, "g--")
    #xscale("log")
    #yscale("log")
    #xlabel(L"r")
    #legend()

    #k1, pkG1 = xicalc(xiGfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-10, N=2^18, q=1.5)
    #k2, pkG2 = xicalc(xiGfn, 0, 0; kmin=1e-10, kmax=1e10, r0=1e-5, N=2^18, q=1.5)
    k3 = 10.0 .^ range(-5, 2, length=200)
    pkG3 = xicalc00_quadosc.(xiGfn, k3)
    #pkG1 .*= (2π)^3
    #pkG2 .*= (2π)^3
    pkG3 .*= (2π)^3
    #@show "xicalc00_quadosc finished"

    k, pkG = k3, pkG3
    #@show pkG

    #figure()
    #plot(k, pkfn.(k), "k", L"P(k)")
    #plot(k1,pkG1, "b", label=L"$P_G(k)$")
    #plot(k1,-pkG1, "b--")
    ##plot(k2,pkG2, "g", label=L"$P_G(k)$")
    ##plot(k2,-pkG2, "g--")
    #plot(k3,pkG3, "r", label=L"$P_G(k)$")
    #plot(k3,-pkG3, "r--")
    #xscale("log")
    #yscale("log")
    ##ylim(1e-14, 1e5)
    #legend()

    #@show k[1], pkG[1]

    # the extremes lead to overflow
    sel = @. (pkG >= 0)
    k = k[sel][3:end-2]
    pkG = pkG[sel][3:end-2]
    #@show k[1], pkG[1]
    while pkG[end] > pkG[end-1]
        k = k[1:end-1]
        pkG = pkG[1:end-1]
    end
    #@show pkG

    sel = @. 1e-5 <= k <= 1e2
    k = k[sel]
    pkG = pkG[sel]
    #@show pkG

    #@show k[1], pkG[1]
    #@show k[end], pkG[end]

    pkGfn = Spline1D(k, pkG, extrapolation=MySplines.powerlaw)
    #@show pkGfn.([0.0, 1e-4])
    return k, pkGfn
end


# vim: set sw=4 et sts=4 :
