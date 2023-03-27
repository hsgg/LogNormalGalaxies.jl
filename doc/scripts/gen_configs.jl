#!/usr/bin/env julia


dname = (@__DIR__)*"/../sim_configs"


config_head = raw"""
randomseed: "[0x659827fbbc75b78b, 0xe5aa8abfea05b963, 0x05caadbdc25751c9, 0xea636b29b0cfc8fb]"

pkfname: "./../../test/matterpower_zeff=0.38.dat"

bias: 1.455  # same as Agrawal+2017
D: 1.0       # already included in power spectrum
nbar: 3e-5   # non need to change between sims
L: 3e3       # let's keep it constant
nrlzs: 100   # can be the same for all

n_sim: 320
n_est: 320
"""


# realspace
for sim_vox in [1,2]
    for fxshift_sim in [0.0, 0.5]
        sim_assign = sim_vox == 1 ? "NGP" : "CIC"
        fname = dname * "/pkest_realspace_sim$(sim_assign)_estNGP_fxshift$(fxshift_sim).yml"
        println("Writing '$fname'...")
        open(fname, "w") do f
            write(f, config_head)
            write(f, "\n")
            write(f, "f: 0.0\n")
            write(f, "sigma_psi: 0.0\n")
            write(f, "\n")
            write(f, "sim_vox: $sim_vox\n")
            write(f, "sim_velo: 1\n")
            write(f, "est_vox: 0\n")
            write(f, "est_grid_assignment: 1\n")
            write(f, "fxshift_sim: $fxshift_sim\n")
            write(f, "fxshift_est: 0.0\n")
        end
    end
end


# redshift space
for sim_velo in [1,2,3,4,5,6]
    for fxshift_sim in [0.0, 0.5]
        fname = dname * "/pkest_redshiftspace_simCIC_estNGP_velo$(sim_velo)_fxshift$(fxshift_sim).yml"
        println("Writing '$fname'...")
        open(fname, "w") do f
            write(f, config_head)
            write(f, "\n")
            write(f, "f: 0.71\n")
            write(f, "sigma_psi: 0.0\n")
            write(f, "\n")
            write(f, "sim_vox: 2\n")
            write(f, "sim_velo: $sim_velo\n")
            write(f, "est_vox: 0\n")
            write(f, "est_grid_assignment: 1\n")
            write(f, "fxshift_sim: $fxshift_sim\n")
            write(f, "fxshift_est: 0.0\n")
        end
    end
end
