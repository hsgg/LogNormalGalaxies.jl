# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


using LogNormalGalaxies
using PythonPlot
using Test


@testset "Draw phases" begin

    n = 4

    rfftplan = LogNormalGalaxies.plan_with_fftw([n,n,n])

    deltak = LogNormalGalaxies.draw_phases(rfftplan)

    LogNormalGalaxies.iterate_kspace(deltak) do ijk_local, ijk_global
        @show ijk_local,ijk_global,deltak[ijk_local...]
    end

end
