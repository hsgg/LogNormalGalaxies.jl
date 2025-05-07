# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


using Test
using LogNormalGalaxies


@testset "iterate_kspace()" begin

    kgrid = rand(ComplexF64, 5, 8, 9)

    idx_center = div.((size(kgrid)...,), 2) .+ 1
    neg_idx_extreme = div.(((size(kgrid) .+ 1)...,), 2)
    @show idx_center


    # k-space
    LogNormalGalaxies.iterate_kspace(kgrid) do ijk_local, ijk_global
        if ijk_local == (1,1,1)
            @test ijk_global == (0,0,0)
        end

        if all(ijk_local .<= idx_center)
            @test ijk_global == ijk_local .- 1
        end

        @test ijk_global[1] == ijk_local[1] - 1


        if ijk_local == idx_center .+ 1
            @test ijk_global[1] == idx_center[1]
            @test ijk_global[2:end] == @. 1 - neg_idx_extreme[2:end]
        end

        if ijk_local == size(kgrid)
            @test ijk_global[1] == size(kgrid, 1) - 1
            @test ijk_global[2:end] == (-1,-1)
        end
    end


    # r-space
    LogNormalGalaxies.iterate_rspace(kgrid) do ijk_local, ijk_global
        if ijk_local == (1,1,1)
            @test ijk_global == (0,0,0)
        end

        @test ijk_global == ijk_local .- 1
    end

end
