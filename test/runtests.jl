#!/usr/bin/env julia

# Just a compile test.

using LogNormalGalaxies
using LogNormalGalaxies.Splines
using DelimitedFiles


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
x⃗, Ψ = simulate_galaxies(nbar, L+ΔL, pk; nmesh=n, bias=b, f=1)



# vim: set sw=4 et sts=4 :
