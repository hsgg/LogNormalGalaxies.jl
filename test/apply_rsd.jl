using LogNormalGalaxies
using Test

@testset "apply_rsd!()" begin
    xyz = rand(3, 10)
    velo = rand(3, 10)

    apply_rsd!(xyz, velo)

    apply_rsd!(xyz, velo, los=:vlos)
    apply_rsd!(xyz, velo, los=[0,0,1])
    apply_rsd!(xyz, velo, los=(0,0,1))

    apply_rsd!(xyz, velo, :vlos)
    apply_rsd!(xyz, velo, [0,0,1])
    apply_rsd!(xyz, velo, (0,0,1))

    apply_rsd!(xyz, velo, 1, :vlos)
    apply_rsd!(xyz, velo, 1, [0,0,1])
    apply_rsd!(xyz, velo, 1, (0,0,1))
end
