using LogNormalGalaxies
using PyPlot
using Test


@testset "Draw phases" begin

    n = 4

    rfftplan = LogNormalGalaxies.plan_with_fftw([n,n,n])

    deltak = LogNormalGalaxies.draw_phases(rfftplan)

    LogNormalGalaxies.iterate_kspace(deltak) do ijk_local, ijk_global
        @show ijk_local,ijk_global,deltak[ijk_local...]
    end

end
